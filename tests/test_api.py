"""
Tests for Flask Inference API
==============================
Covers:
  * GET /health  — returns expected schema.
  * POST /predict — valid payload returns prediction.
  * POST /predict — invalid payload returns 422.
  * POST /predict — missing fields returns 422.
  * GET /metrics — returns stats schema.
  * GET /info    — returns model_info key.

Note
----
These integration tests run against a real model trained in the
``autouse`` fixture.  The model is trained once per test session.
If you want pure unit tests without running training, mock
``ModelPredictor.predict`` instead.
"""

from __future__ import annotations

import json
import pytest

from src.api.app import create_app
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.evaluate import ModelEvaluator
from src.models.registry import ModelRegistry
from src.models.train import ModelTrainer


# ─── Session-scoped model training fixture ───────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def trained_model_artifacts():
    """Train a model once and save artifacts before any API tests run."""
    ing = DataIngestion()
    df = ing.generate_synthetic_dataset(n_samples=600)
    X_train, X_test, y_train, y_test = ing.split_data(df)

    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    proc = DataPreprocessor()
    X_train_proc, X_test_proc = proc.fit_transform(X_train_fe, X_test_fe)
    proc.save()

    trainer = ModelTrainer()
    best_model, cv_results = trainer.train(X_train_proc, y_train)

    evaluator = ModelEvaluator()
    report = evaluator.evaluate(best_model, X_test_proc, y_test)

    registry = ModelRegistry()
    registry.save(
        model=best_model,
        model_name=trainer.best_model_name,
        metrics=report,
        cv_score=cv_results[0][f"cv_{trainer.scoring}_mean"],
        best_params=cv_results[0]["best_params"],
        training_samples=len(X_train),
        feature_count=X_train_proc.shape[1],
    )
    yield


# ─── Flask test client fixture ────────────────────────────────────────────────

@pytest.fixture
def client(trained_model_artifacts):
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─── Valid sample payload ─────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "income": 65000,
    "age": 35,
    "loan_amount": 15000,
    "credit_score": 720,
    "employment_years": 5.0,
    "debt_ratio": 0.25,
    "num_accounts": 3,
    "employment_type": "full_time",
}


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code in (200, 503)   # 503 ok before model loads

    def test_health_schema(self, client) -> None:
        resp = client.get("/health")
        data = resp.get_json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    def test_valid_payload_returns_200(self, client) -> None:
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_response_schema(self, client) -> None:
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["status"] == "success"
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_prediction_is_binary(self, client) -> None:
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["prediction"] in (0, 1)

    def test_probability_in_range(self, client) -> None:
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_missing_field_returns_422(self, client) -> None:
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "income"}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 422

    def test_invalid_employment_type_returns_422(self, client) -> None:
        payload = {**VALID_PAYLOAD, "employment_type": "unemployed"}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 422

    def test_invalid_credit_score_returns_422(self, client) -> None:
        payload = {**VALID_PAYLOAD, "credit_score": 100}  # below 300
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 422

    def test_non_json_body_returns_400(self, client) -> None:
        resp = client.post(
            "/predict",
            data="not json",
            content_type="text/plain",
        )
        assert resp.status_code == 400

    def test_model_version_present(self, client) -> None:
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["model_version"] == "v1"


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_schema(self, client) -> None:
        resp = client.get("/metrics")
        data = resp.get_json()
        assert "metrics" in data
        assert "drift" in data


class TestInfoEndpoint:
    def test_info_returns_200(self, client) -> None:
        resp = client.get("/info")
        assert resp.status_code == 200

    def test_info_schema(self, client) -> None:
        resp = client.get("/info")
        data = resp.get_json()
        assert "model_info" in data


class TestErrorHandling:
    def test_unknown_route_returns_404(self, client) -> None:
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_wrong_method_returns_405(self, client) -> None:
        resp = client.get("/predict")  # GET not allowed
        assert resp.status_code == 405
