# API Documentation

## Overview

The ML Pipeline exposes a RESTful JSON API served by **Flask + Gunicorn** on port `5000`.

Base URL (local):  `http://localhost:5000`
Base URL (Docker): `http://localhost:5000`

All responses follow this envelope:

```json
{
  "status": "success | error",
  ...payload fields...
}
```

---

## Endpoints

### POST /predict

Run inference on a single borrower record.

**Request**

```
POST /predict
Content-Type: application/json
```

```json
{
  "income": 65000,
  "age": 35,
  "loan_amount": 15000,
  "credit_score": 720,
  "employment_years": 5.0,
  "debt_ratio": 0.25,
  "num_accounts": 3,
  "employment_type": "full_time"
}
```

**Request Field Reference**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `income` | float | Yes | > 0 | Annual income in USD |
| `age` | int | Yes | 18 – 120 | Borrower age |
| `loan_amount` | float | Yes | > 0 | Requested loan amount in USD |
| `credit_score` | int | Yes | 300 – 850 | FICO credit score |
| `employment_years` | float | Yes | ≥ 0 | Years at current employer |
| `debt_ratio` | float | Yes | 0.0 – 1.0 | Monthly debt / monthly income |
| `num_accounts` | int | Yes | ≥ 0 | Number of open bank accounts |
| `employment_type` | string | Yes | full_time \| part_time \| self_employed | Employment category |

**Success Response — 200**

```json
{
  "status": "success",
  "prediction": 0,
  "probability": 0.08,
  "model_version": "v1",
  "latency_ms": 12.4
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | int | 0 = no default, 1 = default |
| `probability` | float | Probability of default [0, 1] |
| `model_version` | string | Active model version identifier |
| `latency_ms` | float | End-to-end inference time in milliseconds |

**Error Responses**

```json
// 400 — non-JSON body
{ "status": "error", "message": "Request body must be JSON" }

// 422 — Pydantic validation failure
{
  "status": "error",
  "errors": [
    {
      "type": "value_error",
      "loc": ["employment_type"],
      "msg": "employment_type must be one of {'full_time', 'part_time', 'self_employed'}",
      "input": "unemployed"
    }
  ]
}

// 500 — inference engine failure
{ "status": "error", "message": "Inference failed" }
```

---

### GET /health

Liveness and readiness probe.  Used by Docker HEALTHCHECK, AWS ALB target groups,
and Kubernetes liveness probes.

**Request**

```
GET /health
```

**Success Response — 200**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Degraded Response — 503**

```json
{
  "status": "unhealthy",
  "model_loaded": false
}
```

---

### GET /metrics

Rolling performance statistics over the last 1,000 requests.

> **Note:** In production, gate this endpoint behind internal-network access
> or authentication to avoid leaking system internals.

**Request**

```
GET /metrics
```

**Response — 200**

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

### GET /info

Model registry metadata for the currently loaded model.

**Request**

```
GET /info
```

**Response — 200**

```json
{
  "status": "success",
  "model_info": {
    "model_version": "v1",
    "model_name": "RandomForest",
    "trained_at": "2024-01-15T10:30:00+00:00",
    "metrics": {
      "accuracy": 0.9325,
      "precision": 0.9301,
      "recall": 0.9325,
      "f1": 0.9308,
      "roc_auc": 0.9712
    },
    "cv_score": 0.9187,
    "best_params": {
      "max_depth": 20,
      "min_samples_leaf": 1,
      "min_samples_split": 2,
      "n_estimators": 200
    },
    "feature_count": 14,
    "training_samples": 1600,
    "model_file": "model_v1.pkl",
    "preprocessor_file": "preprocessor_v1.pkl"
  }
}
```

---

## Example cURL Commands

```bash
# Health check
curl http://localhost:5000/health

# Prediction — low-risk borrower
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income": 95000,
    "age": 42,
    "loan_amount": 10000,
    "credit_score": 780,
    "employment_years": 12.0,
    "debt_ratio": 0.15,
    "num_accounts": 5,
    "employment_type": "full_time"
  }'

# Prediction — high-risk borrower
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income": 22000,
    "age": 24,
    "loan_amount": 18000,
    "credit_score": 540,
    "employment_years": 0.5,
    "debt_ratio": 0.68,
    "num_accounts": 1,
    "employment_type": "part_time"
  }'

# Performance metrics
curl http://localhost:5000/metrics

# Model info
curl http://localhost:5000/info
```

---

## Python Client Example

```python
import requests

API_URL = "http://localhost:5000"

payload = {
    "income": 65000,
    "age": 35,
    "loan_amount": 15000,
    "credit_score": 720,
    "employment_years": 5.0,
    "debt_ratio": 0.25,
    "num_accounts": 3,
    "employment_type": "full_time",
}

response = requests.post(f"{API_URL}/predict", json=payload)
result = response.json()

print(f"Prediction  : {result['prediction']}")    # 0 or 1
print(f"Probability : {result['probability']:.2%}")
print(f"Latency     : {result['latency_ms']} ms")
```

---

## Error Code Reference

| HTTP Code | Meaning | When Triggered |
|---|---|---|
| 200 | OK | Successful inference |
| 400 | Bad Request | Non-JSON body |
| 404 | Not Found | Unknown route |
| 405 | Method Not Allowed | Wrong HTTP verb for endpoint |
| 422 | Unprocessable Entity | Pydantic validation error |
| 500 | Internal Server Error | Unhandled exception |
| 503 | Service Unavailable | Model not loaded (health check) |
