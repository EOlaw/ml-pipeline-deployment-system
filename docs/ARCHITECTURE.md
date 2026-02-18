# System Architecture

## Text-Based Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    END-TO-END ML PIPELINE SYSTEM                        ║
╚══════════════════════════════════════════════════════════════════════════╝

             ┌─────────────────────────────────────────┐
             │             DATA SOURCES                │
             │  CSV Files │ Synthetic │ Sklearn │ RDS  │
             └───────────────────┬─────────────────────┘
                                 │
                                 ▼
             ┌─────────────────────────────────────────┐
             │           DATA LAYER                    │
             │  ┌─────────────┐  ┌──────────────────┐  │
             │  │  Ingestion  │  │   Validation     │  │
             │  │ ingestion.py│  │  validation.py   │  │
             │  └──────┬──────┘  └───────┬──────────┘  │
             │         └────────┬─────────┘             │
             │                  ▼                        │
             │        ┌─────────────────┐               │
             │        │  Preprocessing  │               │
             │        │preprocessing.py │               │
             │        └────────┬────────┘               │
             └─────────────────│───────────────────────┘
                               │
                               ▼
             ┌─────────────────────────────────────────┐
             │         FEATURE ENGINEERING LAYER       │
             │       feature_engineering.py            │
             │                                         │
             │  loan_to_income │ credit_tier │         │
             │  age_bracket    │ high_risk_flag        │
             │  income_per_account │ stability         │
             └──────────────────┬──────────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────────┐
             │           MODEL TRAINING LAYER          │
             │              train.py                   │
             │                                         │
             │  ┌──────────────────────────────────┐   │
             │  │   GridSearchCV + StratifiedKFold  │   │
             │  │                                   │   │
             │  │  ┌─────────────┐ ┌─────────────┐ │   │
             │  │  │Random Forest│ │  Logistic   │ │   │
             │  │  │             │ │  Regression │ │   │
             │  │  └─────────────┘ └─────────────┘ │   │
             │  └──────────────────────────────────┘   │
             └──────────────────┬──────────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────────┐
             │         EVALUATION LAYER                │
             │             evaluate.py                 │
             │                                         │
             │  Accuracy │ Precision │ Recall          │
             │  F1-score │ ROC-AUC   │ Confusion Matrix│
             └──────────────────┬──────────────────────┘
                                │ (best model by F1)
                                ▼
             ┌─────────────────────────────────────────┐
             │           MODEL REGISTRY                │
             │             registry.py                 │
             │                                         │
             │  models/model_v1.pkl                    │
             │  models/preprocessor_v1.pkl             │
             │  models/metadata.json                   │
             │                                         │
             │  [AWS] S3 Bucket (model storage)        │
             └──────────────────┬──────────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────────┐
             │       INFERENCE API LAYER               │
             │       Flask + Gunicorn (app.py)         │
             │       Containerized via Docker          │
             │                                         │
             │  POST /predict  ← Pydantic validation   │
             │  GET  /health   ← load balancer probe   │
             │  GET  /metrics  ← rolling stats         │
             │  GET  /info     ← model metadata        │
             └──────────────────┬──────────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────────┐
             │       MONITORING & LOGGING LAYER        │
             │    logger.py │ performance.py           │
             │                                         │
             │  Rotating file logs  (logs/)            │
             │  Latency tracking                       │
             │  Confidence distribution                │
             │  Drift detection placeholder            │
             │  [AWS] CloudWatch Logs                  │
             └──────────────────┬──────────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────────┐
             │         CLIENT / CONSUMER               │
             │  Web App │ Mobile │ Internal Dashboard  │
             └─────────────────────────────────────────┘
```

---

## Layer Descriptions

### 1. Data Layer (`src/data/`)

| File | Responsibility |
|---|---|
| `ingestion.py` | Loads data from CSV, sklearn, or generates synthetic demo data. Splits into train/test. |
| `validation.py` | Checks schema, dtypes, value ranges, nulls, and class balance. Blocks the pipeline on errors. |
| `preprocessing.py` | ColumnTransformer pipeline: median imputation → StandardScaler (numeric), constant imputation → OneHotEncoder (categorical). Saves the fitted transformer. |

### 2. Feature Engineering Layer (`src/features/`)

Creates six domain-derived features before scaling:

| Feature | Formula / Logic |
|---|---|
| `loan_to_income` | `loan_amount / income` |
| `income_per_account` | `income / (num_accounts + 1)` |
| `credit_tier` | Bucketed FICO: very_poor / fair / good / very_good / exceptional |
| `age_bracket` | 18-25 / 26-35 / 36-45 / 46-55 / 56+ |
| `high_risk_flag` | `debt_ratio > 0.5 AND credit_score < 640` |
| `employment_stability` | Ordinal buckets 0-4 from employment_years |

### 3. Model Training Layer (`src/models/train.py`)

- Two candidate models: **RandomForestClassifier** and **LogisticRegression**.
- **GridSearchCV** searches hyper-parameter grids.
- **StratifiedKFold** (k=5) cross-validation preserves class balance.
- Selection criterion: highest mean cross-validated **F1-score**.

### 4. Evaluation Layer (`src/models/evaluate.py`)

Computes on the held-out test set:
- Accuracy, Weighted Precision, Weighted Recall, Weighted F1
- ROC-AUC (via `predict_proba`)
- Confusion matrix

### 5. Model Registry (`src/models/registry.py`)

- Serializes model with `joblib.dump` → `models/model_v1.pkl`
- Saves preprocessor → `models/preprocessor_v1.pkl`
- Writes `models/metadata.json` with version, metrics, params, timestamp
- **AWS equivalent**: uploads to S3, registers in MLflow Model Registry

### 6. Inference API Layer (`src/api/app.py`)

Flask application served by **Gunicorn** (multi-worker WSGI):
- **Pydantic v2** validates every incoming request.
- Lazy model loading — artifacts loaded once per worker process.
- JSON responses follow a consistent `{status, ...}` envelope.

### 7. Monitoring & Logging Layer (`src/monitoring/`)

| File | Function |
|---|---|
| `logger.py` | Rotating file logger (10 MB / file, 5 backups). Console + file output. |
| `performance.py` | In-memory rolling-window tracker for latency, confidence, and prediction distribution. Drift detection placeholder. |

---

## Model Lifecycle

```
1. DATA INGESTION
   Raw data → data/raw/raw_data.csv
   Stratified train/test split

2. TRAINING
   Feature engineering → Preprocessing → GridSearchCV → Best model

3. VALIDATION
   Hold-out test set evaluation
   If F1 < threshold → reject and retrain

4. MODEL SELECTION
   Compare all candidates by test F1
   Best model promoted to registry

5. SERIALIZATION
   joblib.dump(model, "models/model_v1.pkl")
   joblib.dump(preprocessor, "models/preprocessor_v1.pkl")
   metadata.json written

6. DEPLOYMENT
   Docker image built → pushed to ECR
   ECS Task or EC2 instance pulls image
   Gunicorn serves Flask app on port 5000

7. MONITORING
   Per-request latency, confidence, and prediction logged
   Rolling drift detection
   CloudWatch dashboards (production)

8. RETRAINING TRIGGER
   Triggered by any of:
     * Scheduled cron (e.g. weekly)
     * Drift alert (positive rate delta > 10 pp)
     * Manual trigger from CI/CD pipeline
     * New labeled data exceeds threshold
```

---

## Model Versioning Strategy

| Strategy | This Project |
|---|---|
| **Version naming** | `model_v{N}.pkl` — monotonically increasing integer |
| **Metadata** | `metadata.json` stores version, timestamp, metrics, hyper-params |
| **Rollback** | Keep N-1 version in S3; swap the `MODEL_VERSION` env var and restart containers |
| **A/B testing** | Route X% of traffic to model_v2 via ALB weighted target groups |
| **Registry** | MLflow Model Registry tracks transitions (Staging → Production → Archived) |
| **Promotion gate** | New model must beat current production F1 by ≥ 1% on identical test set |

---

## Retraining Strategy

```
Retraining is triggered when:
  ┌──────────────────────────────────────────────────────┐
  │  Trigger 1: Scheduled (weekly cron via EventBridge) │
  │  Trigger 2: Drift alert from PerformanceMonitor     │
  │  Trigger 3: Manual via CI/CD pipeline (GitHub Actions)│
  │  Trigger 4: New labeled data batch exceeds 500 rows  │
  └──────────────────────────────────────────────────────┘
                          │
                          ▼
  1. Fetch latest data from RDS (AWS) or CSV store
  2. Run data validation — reject if quality check fails
  3. Run full training pipeline (run_pipeline.py)
  4. Compare new model vs current production model on fixed test set
  5. If new F1 ≥ current F1 + 1%:
       → Register as model_v{N+1}
       → Upload to S3
       → Rolling deploy to ECS (blue/green)
  6. Else:
       → Log rejection reason
       → Trigger alert
       → Keep current model in production
```
