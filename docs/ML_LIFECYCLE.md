# ML Model Lifecycle

## Overview

The ML Model Lifecycle describes how a model moves from raw data to a
production serving endpoint — and how it is monitored, versioned, and
replaced when it degrades.

---

## Lifecycle Stages

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL LIFECYCLE                             │
│                                                                     │
│  1. DATA         2. TRAINING     3. VALIDATION   4. SELECTION      │
│  INGESTION    ──►  (CV + HPO)  ──►  (test set)  ──►  (best F1)    │
│                                                                     │
│  5. SERIAL-      6. DEPLOY-     7. MONITOR-      8. RETRAIN        │
│  IZATION      ──►  MENT       ──►  ING          ──►  TRIGGER      │
│                                                       │            │
│                              ◄────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Ingestion

**Module:** `src/data/ingestion.py`

**What happens:**
- Raw data is loaded from one of three sources:
  1. Synthetic generator (demo mode)
  2. CSV file (production mode)
  3. AWS RDS PostgreSQL query (production mode)
- Data is saved to `data/raw/raw_data.csv` as an immutable snapshot.
- A stratified train/test split (80/20) is created and returned.

**Key design decisions:**
- Stratification preserves class balance in both splits.
- The random seed is fixed via `config.model.random_state` for reproducibility.
- Raw data is always archived — enables debugging and auditing.

---

## Stage 2: Training

**Module:** `src/models/train.py`

**What happens:**
1. Feature engineering adds six domain-derived columns.
2. Preprocessing (imputation + scaling + encoding) is fitted on `X_train` only — preventing data leakage.
3. **GridSearchCV** searches hyperparameter grids for two model families:
   - `RandomForestClassifier` — 24 grid combinations
   - `LogisticRegression` — 4 grid combinations
4. **StratifiedKFold** (k=5) ensures each fold has representative class balance.
5. The best hyperparameter set for each model is retained.

**Hyperparameter grids:**

| Model | Parameters Searched |
|---|---|
| RandomForest | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| LogisticRegression | C (regularization strength), penalty |

**Parallelism:** `n_jobs=-1` uses all available CPU cores.

---

## Stage 3: Validation

**Module:** `src/data/validation.py` + `src/models/evaluate.py`

**What happens:**
- Data validation: schema, nulls, ranges, types checked before training.
- Model validation: the best estimator from each model family is scored on the held-out test set.

**Metrics computed:**
| Metric | Description |
|---|---|
| Accuracy | Fraction of correct predictions |
| Precision | True positives / (true positives + false positives) |
| Recall | True positives / (true positives + false negatives) |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the receiver operating characteristic curve |
| Confusion Matrix | TP, FP, TN, FN counts |

**Gate criteria:** If no model achieves F1 > 0.60 on the test set, the
pipeline should alert and block deployment (not implemented as hard gate
in demo — add via `if best_report['f1'] < THRESHOLD: sys.exit(1)`).

---

## Stage 4: Model Selection

**Module:** `src/models/train.py` + `src/models/evaluate.py`

**What happens:**
- All candidate models are ranked by **test-set F1-score**.
- The model with the highest F1 is selected as "best model".
- A comparison table is logged showing all models.

**Why F1 over Accuracy:**
For the loan default dataset, the positive class (default) is ~25-30%.
Accuracy can be misleadingly high if the model always predicts "no default".
F1 penalizes both false positives (bad loans approved) and false negatives
(good loans rejected), making it the right business metric here.

---

## Stage 5: Serialization

**Module:** `src/models/registry.py`

**What happens:**
```
models/
├── model_v1.pkl           ← best sklearn estimator (joblib)
├── preprocessor_v1.pkl    ← fitted ColumnTransformer (joblib)
└── metadata.json          ← version, metrics, params, timestamp
```

**metadata.json example:**
```json
{
  "model_version": "v1",
  "model_name": "RandomForest",
  "trained_at": "2024-01-15T10:30:00+00:00",
  "metrics": { "accuracy": 0.93, "f1": 0.91, "roc_auc": 0.97 },
  "cv_score": 0.90,
  "best_params": { "n_estimators": 200, "max_depth": 20 },
  "feature_count": 14,
  "training_samples": 1600,
  "model_file": "model_v1.pkl",
  "preprocessor_file": "preprocessor_v1.pkl"
}
```

**In production:** `metadata.json` is also stored in S3 alongside the
model artifact, and the model is registered in the MLflow Model Registry
with a "Staging" tag before promotion to "Production".

---

## Stage 6: Deployment

**Modules:** `Dockerfile`, `docker-compose.yml`, `src/api/app.py`

**Local deployment:**
```bash
docker build -t ml-pipeline .
docker run -p 5000:5000 ml-pipeline
```

**AWS EC2 deployment (simulated):**
1. `docker build` → `docker tag` → `docker push` to Amazon ECR.
2. EC2 user data script: `docker pull` + `docker run` on boot.
3. ALB health check targets `GET /health`.
4. Auto Scaling Group adjusts instance count based on CPU metrics.

**Zero-downtime update strategy:**
1. Build new Docker image with new model version.
2. Push to ECR.
3. Update ECS Task Definition to new image.
4. ECS rolling update deploys new containers while old ones drain.

---

## Stage 7: Monitoring

**Modules:** `src/monitoring/logger.py`, `src/monitoring/performance.py`

**What is tracked:**
| Signal | Where Logged |
|---|---|
| Request latency | api.log + PerformanceMonitor |
| Prediction class | pipeline.log + PerformanceMonitor |
| Model confidence | pipeline.log + PerformanceMonitor |
| Error count | pipeline.log + PerformanceMonitor |

**Drift detection (placeholder):**
The `check_drift()` method compares the rolling positive-prediction rate
against the training baseline. If the delta exceeds 10 percentage points,
a WARNING is logged and (in production) a CloudWatch Alarm fires.

**Full production monitoring stack:**
- **CloudWatch Logs** — ship all log files via CloudWatch Agent.
- **CloudWatch Metrics** — custom metrics for latency p95, error rate.
- **CloudWatch Alarms** — trigger SNS → Slack/PagerDuty on threshold breach.
- **X-Ray** — distributed tracing for API → model → response path.

---

## Stage 8: Retraining Trigger

**When retraining is triggered:**

| Trigger | Description |
|---|---|
| Scheduled | Weekly cron via AWS EventBridge → Lambda → CodeBuild |
| Drift alert | `check_drift()` returns `drift_detected=True` → CloudWatch Alarm → Lambda |
| Manual | Developer runs `python run_pipeline.py` or triggers CI/CD pipeline |
| Data volume | New labeled data batch exceeds configured threshold |

**Retraining promotion gate:**
```
New model F1 on fixed holdout set
          │
          ▼
  ≥ production F1 + 0.01?
       │          │
      YES         NO
       │          │
       ▼          ▼
  Register     Log rejection
  as v{N+1}   Keep v{N} live
  Deploy       Alert team
```

---

## Model Versioning Strategy

| Concern | Strategy |
|---|---|
| **Naming** | `model_v{N}.pkl` — monotonically increasing integer |
| **Metadata** | JSON file paired with every artifact |
| **Storage** | S3 with versioning enabled (never deletes old versions) |
| **Rollback** | Change `MODEL_VERSION` env var, restart containers |
| **A/B testing** | ALB weighted target groups: 90% v1 / 10% v2 |
| **Registry** | MLflow Model Registry: None → Staging → Production → Archived |
| **Promotion gate** | New model must beat current production F1 by ≥ 1% |
| **Shadow mode** | New model runs in parallel, predictions logged but not served |
