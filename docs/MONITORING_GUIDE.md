# Monitoring & Observability Guide

## Overview

The ML Pipeline has three layers of observability:

| Layer | Tool | Where |
|---|---|---|
| Application Logging | Python `logging` + RotatingFileHandler | `logs/` directory |
| Inference Monitoring | `PerformanceMonitor` | In-memory + `GET /metrics` |
| Production Observability | AWS CloudWatch (simulation) | CloudWatch Logs / Metrics |

---

## Log Files

All logs are written to the `logs/` directory with rotating files
(10 MB per file, 5 backup files kept).

| File | Content |
|---|---|
| `logs/pipeline.log` | Training pipeline events, model metrics |
| `logs/api.log` | Flask request/response logs, inference events |
| `logs/gunicorn_access.log` | Gunicorn access log (Docker production) |
| `logs/gunicorn_error.log` | Gunicorn error log (Docker production) |

### Log Format

```
2024-01-15 10:30:42 | INFO     | src.data.ingestion              | Synthetic dataset ready — (2000, 9), default_rate=28.50%
2024-01-15 10:30:43 | INFO     | src.models.train                | RandomForest — best_cv_f1=0.9187 (std=0.0124)
2024-01-15 10:30:51 | INFO     | src.monitoring.performance      | Inference — prediction=0 probability=0.0823 latency=12.45ms total_requests=1
2024-01-15 10:30:51 | WARNING  | src.monitoring.performance      | Extreme model confidence: probability=0.9723
```

Fields: `timestamp | level | module | message`

---

## What is Logged

### Training Pipeline (`pipeline.log`)

| Event | Level | Example |
|---|---|---|
| Pipeline start/end | INFO | `ML PIPELINE — START` |
| Data loaded | INFO | `Loaded 2,000 rows × 9 columns` |
| Validation pass/fail | INFO/ERROR | `Validation passed` / `Validation FAILED` |
| Feature engineering | INFO | `6 new columns: ['loan_to_income', ...]` |
| CV results | INFO | `RandomForest — best_cv_f1=0.9187` |
| Model saved | INFO | `Model artifact saved ➜ models/model_v1.pkl` |
| Timer blocks | INFO | `[timer] model_training completed in 8420.12ms` |

### API Server (`api.log`)

| Event | Level | Example |
|---|---|---|
| Request received | INFO | `REQUEST  POST /predict  content_type=application/json` |
| Response sent | INFO | `RESPONSE POST /predict  status=200  latency=14.2ms` |
| Validation error | WARNING | `Validation error: [{'type': 'value_error', ...}]` |
| Inference error | ERROR | `Inference error` + full traceback |
| Model loaded | INFO | `Predictor ready.` |

### Performance Monitor

| Event | Level | Trigger |
|---|---|---|
| Per-request metrics | INFO | Every `/predict` call |
| High latency | WARNING | Latency > 500ms |
| Extreme confidence | WARNING | probability < 0.05 or > 0.95 |
| Drift alert | WARNING | Positive-rate delta > 10 pp |

---

## GET /metrics — Live Dashboard

Query the running API for rolling statistics:

```bash
curl http://localhost:5000/metrics | python -m json.tool
```

Response:

```json
{
  "status": "success",
  "metrics": {
    "total_requests": 1523,
    "errors": 2,
    "window_size": 1000,
    "latency_ms": {
      "mean": 18.4,
      "median": 14.2,
      "p95": 52.1,
      "p99": 98.7,
      "max": 212.5
    },
    "prediction_distribution": {
      "positive_rate": 0.2841,
      "avg_confidence": 0.6213
    }
  },
  "drift": {
    "drift_detected": false,
    "current_positive_rate": 0.2841,
    "reference_positive_rate": 0.30,
    "delta": 0.0159,
    "alert": "No significant drift detected."
  }
}
```

---

## Drift Detection

### What is Monitored

The `PerformanceMonitor.check_drift()` method tracks:
- **Prediction distribution drift** — rolling positive-prediction rate vs training baseline.

### Thresholds (Configurable)

| Signal | Warning Threshold | Action |
|---|---|---|
| Positive-rate delta | > 10 percentage points | Log WARNING + CloudWatch Alarm |
| Latency p95 | > 500ms | Log WARNING |
| Error rate | > 1% of window | Log ERROR |
| Extreme confidence | < 5% or > 95% | Log WARNING |

### Full Production Drift Detection

In production, replace the rolling-window approach with:

```python
# Statistical drift tests
from scipy import stats

# KS-test on feature distributions
ks_stat, p_value = stats.ks_2samp(reference_feature, live_feature)
if p_value < 0.05:
    trigger_retraining_alert()

# PSI (Population Stability Index)
psi = sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))
if psi > 0.25:
    trigger_retraining_alert()
```

---

## Privacy & PII Policy

**The ML pipeline never logs PII.**

Rules:
- Input features logged as count and type only — not raw values.
- `sanitize_log()` in `src/utils/__init__.py` strips PII fields before any log call.
- Prediction probabilities logged as-is (they are model outputs, not user data).
- No user identifiers stored anywhere in the logging pipeline.

PII fields blocked from logging:
`name`, `email`, `phone`, `ssn`, `dob`, `address`, `ip_address`

---

## AWS CloudWatch (Production)

### Log Groups

| Log Group | Source |
|---|---|
| `/ml-pipeline/api` | Flask API (via ECS awslogs driver) |
| `/ml-pipeline/training` | Training job logs |
| `/ml-pipeline/predictions` | Inference events |

### CloudWatch Insights Query Examples

**Latency percentiles over last hour:**
```
fields @timestamp, @message
| filter @message like /Inference/
| parse @message "latency=*ms" as latency
| stats pct(latency, 50) as p50,
        pct(latency, 95) as p95,
        pct(latency, 99) as p99
    by bin(5m)
```

**Error rate:**
```
fields @timestamp, @message
| filter @message like /RESPONSE/
| parse @message "status=*" as status
| stats count() as total,
        count_if(status >= 500) as errors
    by bin(1h)
| fields errors / total * 100 as error_rate_pct
```

**Prediction distribution:**
```
fields @timestamp, @message
| filter @message like /prediction=/
| parse @message "prediction=* probability=*" as prediction, probability
| stats count() as n,
        avg(toFloat(probability)) as avg_confidence,
        sum(toInt(prediction)) / count() * 100 as positive_rate_pct
    by bin(1h)
```

### CloudWatch Alarms

| Alarm | Metric | Threshold | Action |
|---|---|---|---|
| High Latency | p95 latency | > 500ms for 5 min | SNS → PagerDuty |
| High Error Rate | 5xx / total | > 1% for 5 min | SNS → Slack |
| Model Drift | positive_rate_delta | > 0.10 for 1h | SNS → Slack + trigger retrain |
| No Requests | request_count | < 1 in 30 min | SNS → Slack (dead service?) |
