# ML Pipeline Deployment System — Loan Default Prediction

> An end-to-end machine learning platform that ingests raw financial data, engineers domain-aware features, trains and selects the best classifier via cross-validated hyperparameter search, and serves real-time predictions through a production-grade Flask API — containerized, monitored, and AWS-ready.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/language-Python%203.11-3B82F6?style=for-the-badge" />
  <img alt="Flask" src="https://img.shields.io/badge/API-Flask%20%2B%20Gunicorn-111827?style=for-the-badge" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/ML-scikit--learn%20%2B%20GridSearchCV-F97316?style=for-the-badge" />
  <img alt="Docker" src="https://img.shields.io/badge/infra-Docker%20%2B%20AWS-0EA5E9?style=for-the-badge" />
</p>

---

## Problem

Financial institutions make high-stakes lending decisions under tight time pressure. The core bottleneck is not a shortage of data — it's the inability to turn raw borrower attributes into a reliable, explainable risk signal fast enough to matter at the point of decision.

**Who is affected:**
- Lenders who rely on slow, manual underwriting processes that do not scale
- Risk teams that need a reproducible, auditable scoring system rather than ad hoc spreadsheets
- Data and ML engineers who inherit untested training scripts with no deployment path

**Why it matters:**
- Unvalidated models trained once and never monitored silently drift in production, generating unreliable predictions without any alert
- Without a clean separation between training and inference, a model change means touching code in multiple places — each a potential regression
- Containerization and API design decisions made at the start determine how easy or impossible it is to deploy, version, and roll back a model later

---

## Solution

Built a **full-stack ML deployment system** that takes raw financial records from ingestion through feature engineering, hyperparameter-tuned training, evaluation, registry, and real-time inference — all in a single orchestrated pipeline with a production-ready API layer on top.

- **Data ingestion** (`src/data/ingestion.py`) supports three sources — a realistic synthetic loan-default generator, arbitrary CSV files, and sklearn built-ins — and always produces a stratified train/test split
- **Data validation** (`src/data/validation.py`) enforces schema, dtype constraints, value ranges, null thresholds, and class-balance checks before any model sees the data; the pipeline hard-exits on validation failure
- **Feature engineering** (`src/features/feature_engineering.py`) derives six domain-aware signals — loan-to-income ratio, income per account, credit tier, age bracket, high-risk flag, and employment stability — before the preprocessor touches anything
- **Preprocessing** (`src/data/preprocessing.py`) applies a `ColumnTransformer` that auto-detects numeric vs. categorical dtypes, applies median imputation and standard scaling to numeric columns, and OneHot-encodes categoricals; the fitted transformer is saved alongside the model for identical inference transforms
- **Model training** (`src/models/train.py`) runs `GridSearchCV` with `StratifiedKFold` across two candidate classifiers (RandomForest and LogisticRegression), selects by mean cross-validated F1, and returns the best estimator
- **Model registry** (`src/models/registry.py`) serializes the champion model and preprocessor with `joblib`, writes a versioned `metadata.json`, and simulates an S3 upload for AWS deployment
- **Flask inference API** (`src/api/app.py`) serves predictions through Pydantic-validated endpoints, lazy-loads the model once per worker, and logs every request/response pair with latency
- **Performance monitor** (`src/monitoring/performance.py`) tracks a rolling 1 000-request window of latency percentiles, confidence distribution, and prediction-class rate — with a drift-detection alert when the positive rate shifts more than 10 percentage points from the training baseline

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **ML Framework** | scikit-learn 1.3+ (GridSearchCV, StratifiedKFold, ColumnTransformer) |
| **Data** | pandas 2.1+, NumPy 1.24+ |
| **API** | Flask 3.0, Gunicorn 21.2 |
| **Request Validation** | Pydantic v2 |
| **Serialization** | joblib |
| **Config** | python-dotenv, dataclasses |
| **Containerization** | Docker (multi-stage), Docker Compose |
| **Testing** | pytest, pytest-flask |
| **Monitoring** | Rotating file logs, in-memory rolling-window stats, drift detection |
| **AWS (production)** | S3 (model storage), EC2 / ECS (serving), CloudWatch (logs), RDS PostgreSQL (dataset store) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                          │
│       Synthetic Generator  ·  CSV File  ·  Sklearn  ·  RDS  │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                    DATA LAYER  (src/data/)                    │
│  ingestion.py — load, save, stratified split                 │
│  validation.py — schema / range / null / balance checks      │
│  preprocessing.py — ColumnTransformer (scale + encode)       │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│             FEATURE ENGINEERING  (src/features/)             │
│  loan_to_income  ·  income_per_account  ·  credit_tier       │
│  age_bracket  ·  high_risk_flag  ·  employment_stability     │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│              MODEL TRAINING LAYER  (src/models/)             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         GridSearchCV + StratifiedKFold (k=5)           │  │
│  │                                                        │  │
│  │   ┌──────────────────┐    ┌──────────────────────┐    │  │
│  │   │ RandomForest     │    │  LogisticRegression  │    │  │
│  │   │ n_estimators,    │    │  C in [0.01 … 10]    │    │  │
│  │   │ max_depth, …     │    │  solver=lbfgs        │    │  │
│  │   └──────────────────┘    └──────────────────────┘    │  │
│  └────────────────────────────────────────────────────────┘  │
│  evaluate.py — accuracy, precision, recall, F1, ROC-AUC      │
└───────────────────────────────┬──────────────────────────────┘
                                │  best model by F1
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                 MODEL REGISTRY  (src/models/)                │
│  models/model_v1.pkl  ·  models/preprocessor_v1.pkl          │
│  models/metadata.json  ·  [AWS] S3 upload simulation         │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│           INFERENCE API  (src/api/app.py)                    │
│  Flask + Gunicorn  ·  Pydantic v2 validation                 │
│  POST /predict  ·  GET /health  ·  GET /metrics  ·  GET /info│
│  Containerized via Docker  ·  ALB-ready health probes        │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│          MONITORING & LOGGING  (src/monitoring/)             │
│  logger.py — rotating file logs (10 MB × 5 backups)         │
│  performance.py — rolling latency / confidence / drift       │
│  [AWS] CloudWatch Logs  ·  PagerDuty drift alerts (hookable) │
└──────────────────────────────────────────────────────────────┘
```

**Data flow:**
- Raw inputs → validation → feature engineering → ColumnTransformer → model artifact saved to `models/`
- At inference: raw JSON → Pydantic validation → feature engineering → same fitted ColumnTransformer → predict/predict_proba → JSON response with prediction, probability, version, and latency
- Every request is logged; the `PerformanceMonitor` singleton maintains a rolling window and surfaces latency p95/p99 and drift deltas through `GET /metrics`

---

## How It Works

1. **Ingestion** — `DataIngestion.run()` generates a 2 000-sample synthetic loan-default dataset (or reads your CSV), saves it to `data/raw/raw_data.csv`, and returns a stratified 80/20 train/test split; the synthetic generator uses a logistic probability model so higher income and credit score genuinely reduce default risk
2. **Validation** — `DataValidator.validate()` checks all eight expected columns are present, dtypes are numeric where required, credit scores are in [300, 850], debt ratios are in [0, 1], null rates are below threshold, and class imbalance is within acceptable bounds; a failed check halts the pipeline with an explicit error log
3. **Feature Engineering** — `FeatureEngineer.fit_transform()` appends six derived columns to the DataFrame before any scaling: `loan_to_income` (leverage signal), `income_per_account` (wealth proxy), `credit_tier` (ordinal FICO bucket), `age_bracket` (life-stage proxy), `high_risk_flag` (rule-based binary), and `employment_stability` (ordinal tenure bucket)
4. **Preprocessing** — `DataPreprocessor.fit_transform()` builds a `ColumnTransformer` that auto-detects string columns as categorical and applies median imputation + `StandardScaler` to numerics, constant imputation + `OneHotEncoder` to categoricals; the fitted preprocessor is saved to `models/preprocessor_v1.pkl` for identical transforms at inference time
5. **Training** — `ModelTrainer.train()` runs `GridSearchCV` with `StratifiedKFold` over two candidate models; both use `class_weight="balanced"` to handle class imbalance; the model with the higher mean CV F1 is selected as the champion
6. **Evaluation** — `ModelEvaluator.evaluate_all()` computes accuracy, weighted precision, recall, F1, ROC-AUC, and a confusion matrix on the held-out test set for all candidates; the comparison table is logged before the registry step
7. **Registry** — `ModelRegistry.save()` serializes the champion estimator with `joblib`, writes a `metadata.json` with version, timestamp, metrics, hyperparameters, feature count, and training sample count, then simulates an S3 upload
8. **API** — `create_app()` builds a Flask application factory; `POST /predict` validates the incoming JSON body against a Pydantic schema (income, age, loan\_amount, credit\_score, employment\_years, debt\_ratio, num\_accounts, employment\_type), runs feature engineering and the fitted ColumnTransformer on the fly, calls `model.predict_proba`, and returns prediction + probability + model version + latency in milliseconds
9. **Monitoring** — `PerformanceMonitor` (singleton, thread-safe deque) records every inference; `GET /metrics` returns latency mean/median/p95/p99, positive prediction rate, and average confidence; `check_drift()` compares the rolling positive rate against the training-time baseline and emits a warning log if the delta exceeds 10 percentage points

---

## Key Techniques

- **Separate fit and transform** — the `ColumnTransformer` is fit only on training data and saved; at inference the exact same fitted object is loaded and applied, preventing any data leakage from the request into the transform
- **dtype auto-detection in the preprocessor** — the pipeline inspects column dtypes at fit time rather than relying on a hardcoded list, so feature-engineered categorical columns (`credit_tier`, `age_bracket`) are automatically routed to the OneHot encoder without manual configuration
- **F1 as the selection criterion** — on a class-imbalanced dataset, accuracy is misleading; cross-validated F1 with `class_weight="balanced"` gives a fairer picture of how well the model handles the minority (default) class
- **Application factory pattern** — `create_app()` returns a fully configured Flask app without side effects, making it trivially testable with `app.test_client()` without starting a real server
- **Lazy model loading** — the predictor loads artifacts on the first request rather than at import time, keeping startup fast and avoiding failures when the API container starts before the model files exist
- **Thread-safe rolling monitor** — `PerformanceMonitor` uses a `threading.Lock` around all deque mutations so it is safe across Gunicorn threads without a separate process or external store
- **Multi-stage Dockerfile** — a `base → dependencies → app` build chain means dependency installation is cached independently of source code changes, keeping rebuild times short in CI
- **AWS deployment simulation** — the registry step logs exactly what an S3 `upload_file`, CloudWatch log group configuration, and ECS task tag would look like in production, making the gap between local and cloud deployment explicit and bridgeable

---

## Results / Impact

- Completes the full training pipeline — ingestion through registry — in **under 60 seconds** on a standard laptop using the synthetic 2 000-sample dataset
- Trains and compares **two classifiers** with full hyperparameter search; the GridSearchCV grid covers 12 RandomForest combinations and 4 LogisticRegression regularization strengths
- Produces **six engineered features** from eight raw inputs before any model sees the data, encoding domain knowledge about credit risk directly into the feature matrix
- API validates incoming requests against **eight typed fields** with bounds checking via Pydantic v2; malformed requests receive structured `422` error responses with per-field detail rather than a generic 500
- Monitoring tracks **four latency percentiles** (mean, median, p95, p99) plus confidence and prediction distribution over a rolling 1 000-request window — without any external dependency
- The full system runs **locally in one command** and deploys to Docker with another; no external services are required for the training or inference path

---

## Business Impact

- **Consistent, reproducible decisions** — every inference runs the same validated feature engineering and the same fitted preprocessor; a new engineer or a new deployment gets identical predictions from identical inputs
- **Auditable model lineage** — `metadata.json` records who trained the model, on how many samples, with which hyperparameters, and what it scored; rollback is a one-env-var change
- **Deployment without a data scientist on call** — the Docker image encapsulates the full runtime; ops can deploy, restart, or roll back the API without touching Python or ML code
- **Drift visibility before it becomes a business problem** — the positive-rate drift check gives risk teams an early signal that the incoming borrower population has shifted, so retraining can happen proactively rather than reactively
- **Scalable by design** — stateless Flask workers behind Gunicorn mean throughput scales horizontally; adding more ECS tasks or EC2 instances increases capacity without changing any application code

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)

### 1. Clone and install

```bash
git clone <repo-url>
cd ml-pipeline-deployment-system
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # adjust values as needed
```

### 2. Train the model

```bash
# Synthetic dataset (default — no data file needed)
PYTHONPATH=. python run_pipeline.py

# From a real CSV
PYTHONPATH=. python run_pipeline.py --source csv --filepath data/raw/my_data.csv

# Verbose output
PYTHONPATH=. python run_pipeline.py --log-level DEBUG
```

This writes `models/model_v1.pkl`, `models/preprocessor_v1.pkl`, and `models/metadata.json`.

### 3. Run the API

```bash
# Development server (macOS: use port 5001 — port 5000 is taken by AirPlay)
PYTHONPATH=. API_PORT=5001 python -m src.api.app
```

### 4. Run with Docker

```bash
# API only
docker compose up --build

# Train inside Docker first, then serve
docker compose --profile training run ml-trainer
docker compose up ml-api
```

### 5. Run tests

```bash
PYTHONPATH=. pytest tests/ -v
```

### Example prediction request

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income": 65000,
    "age": 35,
    "loan_amount": 15000,
    "credit_score": 720,
    "employment_years": 5.0,
    "debt_ratio": 0.25,
    "num_accounts": 3,
    "employment_type": "full_time"
  }'
```

```json
{
  "status": "success",
  "prediction": 0,
  "probability": 0.08,
  "model_version": "v1",
  "latency_ms": 12.4
}
```

---

## Environment Variables

```env
# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=5000
FLASK_DEBUG=false
MODEL_VERSION=v1

# ─── Training ────────────────────────────────────────────────────────────────
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5
N_JOBS=-1
SCORING_METRIC=f1

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO

# ─── AWS (production) ────────────────────────────────────────────────────────
AWS_REGION=us-east-1
S3_BUCKET=ml-pipeline-models
S3_MODEL_KEY=models/model_v1.pkl
RDS_HOST=your-rds-endpoint.us-east-1.rds.amazonaws.com
RDS_PORT=5432
RDS_DATABASE=mldb
RDS_USER=mluser
CLOUDWATCH_LOG_GROUP=/ml-pipeline/predictions
```

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run inference on a single borrower feature payload |
| `/health` | GET | Liveness probe — returns 200 when model is loaded, 503 otherwise |
| `/metrics` | GET | Rolling performance stats (latency percentiles, prediction distribution, drift delta) |
| `/info` | GET | Model version and registry metadata from `metadata.json` |

**Request schema for `POST /predict`:**

| Field | Type | Constraints |
|---|---|---|
| `income` | float | > 0 |
| `age` | int | 18 – 120 |
| `loan_amount` | float | > 0 |
| `credit_score` | int | 300 – 850 |
| `employment_years` | float | ≥ 0 |
| `debt_ratio` | float | 0 – 1 |
| `num_accounts` | int | ≥ 0 |
| `employment_type` | string | `full_time` \| `part_time` \| `self_employed` |

---

## Project Structure

```
ml-pipeline-deployment-system/
├── src/
│   ├── api/
│   │   └── app.py                    # Flask app factory, /predict /health /metrics /info
│   ├── config/
│   │   └── config.py                 # Dataclass config — env vars override defaults
│   ├── data/
│   │   ├── ingestion.py              # Synthetic generator, CSV loader, train/test split
│   │   ├── preprocessing.py          # ColumnTransformer — scale numeric, encode categorical
│   │   └── validation.py             # Schema, dtype, range, null, and balance checks
│   ├── features/
│   │   └── feature_engineering.py    # 6 domain-derived features before preprocessing
│   ├── models/
│   │   ├── train.py                  # GridSearchCV over RandomForest + LogisticRegression
│   │   ├── evaluate.py               # Accuracy, F1, ROC-AUC, confusion matrix
│   │   ├── predict.py                # ModelPredictor — loads artifacts, runs inference
│   │   └── registry.py              # Save/load model, write metadata.json, S3 simulation
│   ├── monitoring/
│   │   ├── logger.py                 # Rotating file + console logger
│   │   └── performance.py            # Thread-safe rolling window — latency, drift
│   └── utils/
│
├── data/
│   └── raw/raw_data.csv              # Written by ingestion step
│
├── models/
│   ├── model_v1.pkl                  # Serialized champion estimator
│   ├── preprocessor_v1.pkl           # Fitted ColumnTransformer
│   └── metadata.json                 # Version, metrics, hyperparameters, timestamp
│
├── tests/
│   ├── test_api.py                   # Flask test client — endpoint response shapes
│   ├── test_ingestion.py             # DataIngestion — split shapes, column presence
│   ├── test_preprocessing.py         # ColumnTransformer fit/transform correctness
│   └── test_training.py              # ModelTrainer — returns fitted estimator
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API_DOCUMENTATION.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── ML_LIFECYCLE.md
│   ├── MONITORING_GUIDE.md
│   └── FOLDER_STRUCTURE.md
│
├── logs/
│   ├── pipeline.log                  # Training pipeline logs (rotating)
│   └── api.log                       # API request/response logs (rotating)
│
├── Dockerfile                        # Multi-stage: base → dependencies → app
├── docker-compose.yml                # ml-api service + ml-trainer (--profile training)
├── run_pipeline.py                   # End-to-end pipeline orchestrator
├── requirements.txt
└── .env.example
```

---

## Key Takeaways

- Demonstrates a **production ML deployment pattern** — the pipeline is not a notebook; it is a versioned, testable, containerized system where training and inference are explicitly separated by a model registry boundary
- Built a **schema-validated inference path** where every incoming request is checked by Pydantic before reaching the model, and the exact same feature engineering and fitted preprocessing transform applied during training is replicated at inference — no silent mismatch
- Applied **real-world engineering practices**: application factory for testability, lazy model loading for resilience, thread-safe monitoring, rotating log files, multi-stage Docker builds, and env-var-driven configuration that works identically locally and in AWS
- The monitoring layer provides **actionable runtime signals** — latency percentiles and prediction-distribution drift — without any external dependency, so the system is observable from day one without requiring a metrics stack to be operational first

---

## License

MIT
