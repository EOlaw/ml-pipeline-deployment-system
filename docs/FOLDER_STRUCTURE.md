# Folder Structure & Code Walkthrough

## Project Tree

```
ml-pipeline-deployment-system/
│
├── data/                          ← All data artifacts (gitignored in production)
│   ├── raw/                       ← Immutable raw snapshots from ingestion
│   │   └── raw_data.csv
│   ├── processed/                 ← Reserved for intermediate processed files
│   └── external/                  ← Third-party data (reference datasets, etc.)
│
├── models/                        ← Trained model artifacts
│   ├── model_v1.pkl               ← Best sklearn estimator (joblib-serialized)
│   ├── preprocessor_v1.pkl        ← Fitted ColumnTransformer (joblib-serialized)
│   └── metadata.json              ← Version, metrics, hyper-params, timestamp
│
├── logs/                          ← Rotating log files
│   ├── pipeline.log               ← Training pipeline log
│   ├── api.log                    ← Flask API request/response log
│   ├── gunicorn_access.log        ← Gunicorn access log (Docker)
│   └── gunicorn_error.log         ← Gunicorn error log (Docker)
│
├── src/                           ← All source code
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py              ← Centralized configuration (env-var overridable)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py           ← Load data from CSV / synthetic / sklearn
│   │   ├── validation.py          ← Schema & quality checks
│   │   └── preprocessing.py       ← ColumnTransformer (scale + encode)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py ← Domain-derived features (6 new columns)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py               ← GridSearchCV on RF + LogReg
│   │   ├── evaluate.py            ← Test-set metrics (accuracy, F1, ROC-AUC, ...)
│   │   ├── registry.py            ← Save / load / list model artifacts
│   │   └── predict.py             ← Inference wrapper (used by Flask API)
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── logger.py              ← Rotating-file logger factory
│   │   └── performance.py         ← In-memory latency/confidence/drift monitor
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                 ← Flask app factory + /predict /health /metrics /info
│   │
│   └── utils/
│       └── __init__.py            ← Shared helpers: timer, save_json, sanitize_log
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py          ← Tests for data ingestion & splitting
│   ├── test_preprocessing.py      ← Tests for validation, feature eng., preprocessing
│   ├── test_training.py           ← Tests for training, CV, evaluation
│   └── test_api.py                ← Integration tests for Flask API endpoints
│
├── docs/
│   ├── ARCHITECTURE.md            ← System design, layers, lifecycle, versioning
│   ├── API_DOCUMENTATION.md       ← Endpoint specs, cURL examples, schemas
│   ├── ML_LIFECYCLE.md            ← Stage-by-stage lifecycle walkthrough
│   ├── DEPLOYMENT_GUIDE.md        ← Local, Docker, and AWS deployment steps
│   ├── MONITORING_GUIDE.md        ← Logging, metrics, drift, CloudWatch
│   └── FOLDER_STRUCTURE.md        ← This file
│
├── Dockerfile                     ← Multi-stage Docker build (slim Python image)
├── docker-compose.yml             ← API service + training runner profile
├── requirements.txt               ← Pinned Python dependencies
├── .env.example                   ← Template for environment variables (never commit .env)
├── run_pipeline.py                ← CLI entry point: runs full training pipeline
└── README.md                      ← Mission, quickstart, architecture overview
```

---

## File-by-File Code Explanation

### `src/config/config.py`

**Purpose:** Single source of truth for every configurable value.

**Key design:**
- Uses Python `@dataclass` for IDE auto-complete and type safety.
- Reads environment variables via `os.getenv()` so the same codebase
  works locally, in Docker, and on AWS without code changes.
- `create_directories()` ensures runtime folders exist on startup.
- Exports a singleton `config` object — import it everywhere with
  `from src.config.config import config`.

**What it configures:**
| Section | Contains |
|---|---|
| `DataConfig` | File paths, column names, train/test split ratio |
| `ModelConfig` | Hyper-parameter defaults, CV folds, feature column lists |
| `APIConfig` | Flask host/port/debug, model version label |
| `AWSConfig` | S3 bucket, RDS host, CloudWatch log group |
| `LoggingConfig` | Log file paths, level, rotation settings |

---

### `src/monitoring/logger.py`

**Purpose:** Provides `get_logger(name)` — a factory that returns a
named Python logger with two handlers already attached:
1. `StreamHandler` — prints to console during development.
2. `RotatingFileHandler` — writes to `logs/pipeline.log` (10 MB, 5 backups).

**Key design:**
- Logger is initialized **once per name** (checked via `not logger.handlers`).
- `propagate = False` prevents duplicate output to the root logger.
- `get_api_logger()` is a convenience wrapper that logs to `api.log`.
- No dependency on `config.py` — uses a hard-coded `LOGS_DIR` path so it
  can be imported before config is fully initialized.

---

### `src/data/ingestion.py`

**Purpose:** Acquire raw data from multiple sources and split it.

**Key classes/methods:**
- `generate_synthetic_dataset(n_samples)` — creates a realistic loan-default
  dataset using `numpy.random` with fixed seed for reproducibility.
  Uses logistic function to generate realistic default probabilities.
- `load_from_csv(filepath)` — reads any CSV; raises `FileNotFoundError` on bad path.
- `load_sklearn_dataset()` — wraps sklearn's breast_cancer dataset.
- `split_data(df)` — stratified train/test split (preserves class ratio).
- `save_raw_data(df)` — archives the raw file to `data/raw/`.

**Synthetic dataset features (9 columns):**
`income`, `age`, `loan_amount`, `credit_score`, `employment_years`,
`debt_ratio`, `num_accounts`, `employment_type`, `target`

---

### `src/data/validation.py`

**Purpose:** Gate the pipeline — reject bad data before it corrupts a model.

**Key class:** `DataValidator`
**Key method:** `validate(df, is_inference=False)` → returns `{is_valid, errors, warnings}`

**Checks performed:**
1. Required columns present.
2. No column > 30% null.
3. Numeric columns actually numeric (or castable).
4. Value ranges (e.g. credit_score ∈ [300, 850]).
5. Categorical values from known set.
6. Duplicate row detection.
7. Class balance warning (if positive rate < 5% or < 15%).

`is_inference=True` skips the `target` column requirement (prediction
requests don't have labels).

---

### `src/data/preprocessing.py`

**Purpose:** Transform raw DataFrames into model-ready numpy arrays.

**Key class:** `DataPreprocessor`

**Pipeline architecture (sklearn ColumnTransformer):**
```
Numeric columns  → SimpleImputer(median) → StandardScaler
Categorical cols → SimpleImputer(constant="unknown") → OneHotEncoder(drop="first")
```

**Key methods:**
- `fit_transform(X_train, X_test)` — fits on train, transforms both.
- `transform(X)` — applies fitted pipeline to new data.
- `save()` / `load()` — joblib serialization to `models/preprocessor_v1.pkl`.

**Why `drop="first"` in OneHotEncoder?**
Avoids perfect multicollinearity (dummy variable trap) which can destabilize
logistic regression training.

---

### `src/features/feature_engineering.py`

**Purpose:** Add domain-knowledge features before scaling.

**Key class:** `FeatureEngineer`

After `fit_transform()`, these columns are added:

| Feature | Why It Matters |
|---|---|
| `loan_to_income` | Classic credit-risk leverage ratio |
| `income_per_account` | Wealth diversification indicator |
| `credit_tier` | Non-linear bucketing of FICO score |
| `age_bracket` | Life-stage proxy (financial stability) |
| `high_risk_flag` | Rule-based composite risk signal |
| `employment_stability` | Ordinal encoding of tenure |

**Stateful design:** After `fit_transform()` the `_fitted` flag is set so
`transform()` can be called safely on inference data.

---

### `src/models/train.py`

**Purpose:** Train and compare multiple classifiers.

**Key class:** `ModelTrainer`

**Candidate models (defined in `_get_candidate_models()`):**
1. `RandomForestClassifier` — ensemble, non-linear, handles feature interactions.
2. `LogisticRegression` — linear baseline, fast, highly interpretable.

**Training flow:**
1. `GridSearchCV` sweeps hyperparameter grid for each model.
2. `StratifiedKFold(n_splits=5)` provides CV folds.
3. Best hyperparameters per model are retained.
4. `cross_val_score` computes fold-by-fold scores for std reporting.
5. Results sorted descending by mean CV F1.

**Output:** `(best_estimator, cv_results_list)`

---

### `src/models/evaluate.py`

**Purpose:** Compute and report all classification metrics on the test set.

**Key class:** `ModelEvaluator`

**Key methods:**
- `evaluate(model, X_test, y_test)` — single model evaluation.
- `evaluate_all(cv_results, X_test, y_test)` — evaluate all candidates.
- `print_report(report)` — formatted console output.
- `comparison_dataframe(reports)` — tidy DataFrame for all models.

**Metrics returned:** accuracy, precision, recall, f1, roc_auc,
confusion_matrix, positive_rate_actual, positive_rate_predicted.

---

### `src/models/registry.py`

**Purpose:** Version, persist, and retrieve model artifacts.

**Key class:** `ModelRegistry`

**Key methods:**
- `save(model, model_name, metrics, ...)` — joblib dump + metadata JSON.
- `load()` — joblib load from `models/model_v1.pkl`.
- `load_metadata()` — reads `models/metadata.json`.
- `model_exists()` — checks if trained model artifact is present.
- `simulate_s3_upload()` — logs the AWS command that production would run.

**Metadata JSON** stores everything needed to reproduce or audit the model:
version, name, training timestamp, all metrics, hyperparameters,
feature count, sample count, and file references.

---

### `src/models/predict.py`

**Purpose:** Single inference object for the Flask API.

**Key class:** `ModelPredictor`

**Inference flow for each request:**
```
raw dict
  → pd.DataFrame
  → FeatureEngineer.transform()    (add derived columns)
  → DataPreprocessor.transform()   (scale + encode)
  → model.predict() + predict_proba()
  → {prediction, probability, model_version, latency_ms}
```

**Key design:**
- Lazy loading — artifacts loaded on first request, cached in memory.
- `_ensure_loaded()` called at the start of every `predict()` call.
- `PerformanceMonitor.record()` called after every successful inference.
- Thread-safe for concurrent Gunicorn workers (read-only after load).

---

### `src/monitoring/performance.py`

**Purpose:** Track inference quality signals in real time.

**Key class:** `PerformanceMonitor` (singleton)

**Rolling window:** Last 1,000 requests stored in `collections.deque(maxlen=1000)`.

**Key methods:**
- `record(prediction, probability, latency_ms)` — called after each inference.
- `get_stats()` — latency percentiles + prediction distribution.
- `check_drift(reference_positive_rate)` — compares rolling vs baseline.
- `reset()` — clears all buffers (used in tests).

**Warnings triggered automatically:**
- Latency > 500ms
- Probability < 0.05 or > 0.95 (out-of-distribution input signal)

---

### `src/api/app.py`

**Purpose:** Flask application factory with four endpoints.

**Key design:**
- **App factory pattern** (`create_app()`) enables pytest to spin up a
  fresh app instance per test session.
- **Pydantic v2** `BaseModel` validates every `/predict` request —
  returns descriptive 422 errors on bad input.
- Request timing via `before_request` / `after_request` hooks.
- `ModelPredictor` is instantiated once per worker process — not per request.

**Endpoints:**

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Run inference |
| GET | `/health` | Liveness probe |
| GET | `/metrics` | Rolling performance stats |
| GET | `/info` | Model metadata |

---

### `src/utils/__init__.py`

**Purpose:** Shared helper utilities.

| Function | Purpose |
|---|---|
| `timer(label)` | Context manager that logs elapsed time |
| `save_json(data, path)` | Write dict to JSON file |
| `load_json(path)` | Read JSON file to dict |
| `sanitize_log(record)` | Remove PII fields before logging |
| `flatten_dict(d)` | Flatten nested dict to dot-notation |

---

### `run_pipeline.py`

**Purpose:** CLI entry point that runs the full training pipeline end-to-end.

**Steps orchestrated:**
1. Data ingestion
2. Data validation
3. Feature engineering
4. Preprocessing
5. Model training
6. Model evaluation
7. Model registry (save artifacts)
8. AWS deployment simulation

**CLI flags:**
- `--source` — `synthetic` (default) | `csv` | `sklearn`
- `--filepath` — path to CSV (when `--source=csv`)
- `--log-level` — `DEBUG` | `INFO` | `WARNING` | `ERROR`

---

### `tests/`

| File | What it tests |
|---|---|
| `test_ingestion.py` | Synthetic dataset shape/dtypes, CSV round-trip, split stratification |
| `test_preprocessing.py` | Validator checks, feature engineer columns, preprocessor fit/transform |
| `test_training.py` | Model training completion, CV score ranges, evaluation metrics |
| `test_api.py` | All 4 endpoints, happy paths, validation errors, error handling |

**Session-scoped fixture in `test_api.py`:**
`trained_model_artifacts` trains a real model once for all API tests,
avoiding repeated training overhead while ensuring tests run against a
real model.

---

### `Dockerfile`

**Multi-stage build:**
1. `base` — python:3.11-slim + system packages.
2. `dependencies` — pip install from requirements.txt.
3. `app` — copy project, create dirs, add non-root user, expose port 5000.

**Production server:** Gunicorn with 2 workers, 4 threads, 120s timeout.

**Security:** Runs as non-root `mluser` inside the container.

---

### `docker-compose.yml`

Two services:
- `ml-api` — always-running Flask API with volume mounts for models/logs/data.
- `ml-trainer` — one-shot training runner (activated with `--profile training`).

Shared `ml-network` bridge network for future service-to-service communication.
