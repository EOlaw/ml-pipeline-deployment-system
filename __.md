# Production ML Pipeline & Model Deployment System

> A fully production-ready, end-to-end Machine Learning pipeline featuring
> data ingestion, validation, feature engineering, model training,
> evaluation, a REST inference API, monitoring, Docker containerization,
> and AWS deployment architecture.

---

## Mission

Build a **modular, clean-architecture MLOps system** that a team can take
from a raw dataset all the way to a monitored, versioned, Dockerized
inference API — following production best practices at every stage.

This project demonstrates how ML systems should be designed for real
engineering teams: testable, observable, reproducible, and deployable
without code changes across local, Docker, and AWS environments.

---

## Architecture at a Glance

```
Data Source ──► Ingestion ──► Validation ──► Feature Engineering
                                                     │
                                                     ▼
                                              Preprocessing
                                                     │
                                                     ▼
                                           Model Training (CV + HPO)
                                          Random Forest │ Logistic Regression
                                                     │
                                                     ▼
                                             Evaluation
                                    Accuracy │ F1 │ ROC-AUC │ Confusion Matrix
                                                     │ (best model)
                                                     ▼
                                           Model Registry
                                   models/model_v1.pkl │ metadata.json
                                                     │
                                                     ▼
                                     Flask API (Docker / AWS EC2)
                                  POST /predict │ GET /health │ GET /metrics
                                                     │
                                             Monitoring
                              Rotating Logs │ Latency │ Drift Detection
```

---

## What's Inside

| Layer | Files | Description |
|---|---|---|
| Config | `src/config/config.py` | Centralized env-var-driven configuration |
| Ingestion | `src/data/ingestion.py` | CSV / synthetic / sklearn data loading |
| Validation | `src/data/validation.py` | Schema, range, null, balance checks |
| Preprocessing | `src/data/preprocessing.py` | Imputation + scaling + encoding |
| Features | `src/features/feature_engineering.py` | 6 domain-derived features |
| Training | `src/models/train.py` | GridSearchCV, StratifiedKFold, 2 models |
| Evaluation | `src/models/evaluate.py` | Accuracy, Precision, Recall, F1, ROC-AUC |
| Registry | `src/models/registry.py` | Joblib persist + metadata.json |
| Prediction | `src/models/predict.py` | Inference wrapper for the API |
| API | `src/api/app.py` | Flask + Pydantic + Gunicorn |
| Logging | `src/monitoring/logger.py` | Rotating-file logger factory |
| Monitoring | `src/monitoring/performance.py` | Latency, confidence, drift |
| Tests | `tests/` | 4 test files covering all layers |

---

## What to Install

### Prerequisites

- **Python 3.11+**
- **pip** (or virtualenv/conda)
- **Docker** + **Docker Compose** (for containerized deployment)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | 1.3.2 | ML models, pipelines, GridSearchCV |
| `pandas` | 2.1.4 | DataFrame processing |
| `numpy` | 1.26.2 | Numerical computing |
| `flask` | 3.0.0 | REST API framework |
| `gunicorn` | 21.2.0 | Production WSGI server |
| `pydantic` | 2.5.2 | Request validation |
| `joblib` | 1.3.2 | Model serialization |
| `python-dotenv` | 1.0.0 | .env file loading |
| `mlflow` | 2.9.2 | Experiment tracking (optional) |
| `pytest` | 7.4.3 | Testing framework |

---

## How to Run

### Option 1 — Local Python (Quickest)

```bash
# 1. Clone / enter the project
cd ml-pipeline-deployment-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env

# 5. Run the training pipeline
#    Generates data ► validates ► engineers features ► trains ► evaluates ► saves
python run_pipeline.py

# 6. Start the inference API
python -m src.api.app
#    API is live at http://localhost:5000

# 7. Make a prediction
curl -X POST http://localhost:5000/predict \
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

Expected response:

```json
{
  "status": "success",
  "prediction": 0,
  "probability": 0.08,
  "model_version": "v1",
  "latency_ms": 14.2
}
```

---

### Option 2 — Docker (Recommended for Production)

```bash
# Build the image
docker build -t ml-pipeline:latest .

# Step 1: Run training to produce model artifacts
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  ml-pipeline:latest \
  python run_pipeline.py

# Step 2: Start the API
docker run -d \
  --name ml-api \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  ml-pipeline:latest

# Check health
curl http://localhost:5000/health
```

---

### Option 3 — Docker Compose

```bash
# Train model (one-shot)
docker compose --profile training up ml-trainer

# Start API
docker compose up -d ml-api

# Tail logs
docker compose logs -f ml-api

# Stop
docker compose down
```

---

## Running Tests

```bash
# All tests with verbose output
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Single module
pytest tests/test_api.py -v
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Run inference — returns prediction + probability |
| `GET` | `/health` | Liveness probe — returns `{"status": "healthy"}` |
| `GET` | `/metrics` | Rolling latency, confidence, drift stats |
| `GET` | `/info` | Model version and registry metadata |

Full API documentation: [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)

---

## ML Details

### Dataset

Synthetic **loan default** dataset (2,000 samples):

| Feature | Type | Description |
|---|---|---|
| `income` | float | Annual income (USD) |
| `age` | int | Borrower age |
| `loan_amount` | float | Requested loan (USD) |
| `credit_score` | int | FICO score (300–850) |
| `employment_years` | float | Years at current employer |
| `debt_ratio` | float | Monthly debt / income [0, 1] |
| `num_accounts` | int | Open bank accounts |
| `employment_type` | str | full_time / part_time / self_employed |
| `target` | int | 0 = no default, 1 = default |

### Models Trained

| Model | Hyperparameters Tuned |
|---|---|
| RandomForestClassifier | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| LogisticRegression | C, penalty |

### Evaluation Metrics

- Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)
- ROC-AUC
- Confusion Matrix

### Selection Criterion

Best model = highest **mean cross-validated F1-score** (5-fold StratifiedKFold).

---

## Project Documentation

| Document | Description |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, text diagram, lifecycle, versioning |
| [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | Full endpoint specs, cURL examples, Python client |
| [docs/ML_LIFECYCLE.md](docs/ML_LIFECYCLE.md) | Stage-by-stage ML lifecycle walkthrough |
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | Local, Docker, AWS deployment instructions |
| [docs/MONITORING_GUIDE.md](docs/MONITORING_GUIDE.md) | Logging, metrics, drift, CloudWatch guide |
| [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md) | Every file and what its code does |

---

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_VERSION` | `v1` | Active model version |
| `API_PORT` | `5000` | Flask port |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `SCORING_METRIC` | `f1` | Hyperparameter search metric |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## AWS Deployment (Simulated)

The system simulates how it would deploy on AWS:

| AWS Service | Role |
|---|---|
| **S3** | Model artifact storage (versioned bucket) |
| **EC2 / ECS Fargate** | Runs the Dockerized Flask API |
| **RDS (PostgreSQL)** | Stores training dataset |
| **CloudWatch** | Log aggregation + alarms |
| **Secrets Manager** | DB passwords, API keys (never in code) |
| **IAM Roles** | Least-privilege access for ECS tasks |
| **ECR** | Docker image registry |
| **ALB** | Load balancer with `/health` target group check |

See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for full step-by-step.

---

## License

MIT — use freely for learning, portfolio projects, and production systems.
