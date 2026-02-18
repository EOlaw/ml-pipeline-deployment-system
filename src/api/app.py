"""
Flask Inference API
===================
Provides two endpoints:

  POST /predict   — run model inference on a feature payload.
  GET  /health    — liveness / readiness check for load balancers.
  GET  /metrics   — rolling performance stats (internal use).
  GET  /info      — model version and metadata.

Design choices
--------------
* Pydantic v2 for request validation  — clean error messages.
* Lazy model loading                   — model is loaded once on first request
  (or at startup via ``_init_predictor``).
* No PII stored                        — inputs are logged as feature count
  and value ranges only.
* JSON responses use consistent schema — always include ``status`` key.

Usage
-----
    # Development
    python -m src.api.app

    # Production (via Gunicorn — see Dockerfile)
    gunicorn --bind 0.0.0.0:5000 src.api.app:app
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, Response
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config.config import config
from src.models.predict import ModelPredictor
from src.monitoring.logger import get_api_logger
from src.monitoring.performance import PerformanceMonitor

logger = get_api_logger(__name__)


# ─── Pydantic request schema ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Validates the /predict request body."""

    income: float = Field(..., gt=0, description="Annual income in USD")
    age: int = Field(..., ge=18, le=120, description="Borrower age")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    credit_score: int = Field(..., ge=300, le=850, description="FICO credit score")
    employment_years: float = Field(..., ge=0, description="Years at current employer")
    debt_ratio: float = Field(..., ge=0, le=1, description="Debt-to-income ratio [0, 1]")
    num_accounts: int = Field(..., ge=0, description="Number of open bank accounts")
    employment_type: str = Field(
        ..., description="One of: full_time | part_time | self_employed"
    )

    @field_validator("employment_type")
    @classmethod
    def validate_employment_type(cls, v: str) -> str:
        allowed = {"full_time", "part_time", "self_employed"}
        if v.lower() not in allowed:
            raise ValueError(f"employment_type must be one of {allowed}")
        return v.lower()

    model_config = {"extra": "ignore"}   # silently drop unknown fields


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """Application factory — enables testing with different configs."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # ── Shared predictor (loaded once per worker) ─────────────────────────
    predictor = ModelPredictor()
    monitor = PerformanceMonitor()

    def _get_predictor() -> ModelPredictor:
        """Lazy-load model on first request."""
        if not predictor.is_loaded:
            predictor.load()
        return predictor

    # ── Request / response logging middleware ─────────────────────────────
    @app.before_request
    def _log_request() -> None:
        request._start_time = time.perf_counter()   # type: ignore[attr-defined]
        logger.info(
            f"REQUEST  {request.method} {request.path}  "
            f"content_type={request.content_type}"
        )

    @app.after_request
    def _log_response(response: Response) -> Response:
        elapsed = round(
            (time.perf_counter() - getattr(request, "_start_time", time.perf_counter())) * 1000, 2
        )
        logger.info(
            f"RESPONSE {request.method} {request.path}  "
            f"status={response.status_code}  latency={elapsed}ms"
        )
        return response

    # ── Endpoints ─────────────────────────────────────────────────────────

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        """
        Liveness probe for Docker HEALTHCHECK and AWS ALB target groups.

        Response
        --------
        200  {"status": "healthy",   "model_loaded": true}
        503  {"status": "unhealthy", "model_loaded": false}
        """
        model_loaded = predictor.is_loaded and predictor._model is not None
        status_code = 200 if model_loaded else 503
        return (
            jsonify({"status": "healthy" if model_loaded else "unhealthy",
                     "model_loaded": model_loaded}),
            status_code,
        )

    @app.route("/predict", methods=["POST"])
    def predict() -> tuple[Response, int]:
        """
        Run inference on a single feature payload.

        Request body (JSON)
        -------------------
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

        Success response (200)
        ----------------------
        {
          "status": "success",
          "prediction": 0,
          "probability": 0.08,
          "model_version": "v1",
          "latency_ms": 12.4
        }
        """
        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"status": "error", "message": "Request body must be JSON"}), 400

        # Validate with Pydantic
        try:
            validated = PredictRequest(**body)
        except ValidationError as exc:
            # Pydantic ctx may contain Exception objects — stringify for JSON safety
            errors = [
                {k: str(v) if k == "ctx" else v for k, v in e.items()}
                for e in exc.errors(include_url=False)
            ]
            logger.warning(f"Validation error: {errors}")
            return jsonify({"status": "error", "errors": errors}), 422

        # Run inference
        try:
            pred_input = validated.model_dump()
            result = _get_predictor().predict(pred_input)
        except Exception as exc:
            monitor.record_error(str(exc))
            logger.exception("Inference error")
            return jsonify({"status": "error", "message": "Inference failed"}), 500

        return (
            jsonify(
                {
                    "status": "success",
                    "prediction": result["prediction"],
                    "probability": result["probability"],
                    "model_version": result["model_version"],
                    "latency_ms": result["latency_ms"],
                }
            ),
            200,
        )

    @app.route("/metrics", methods=["GET"])
    def metrics() -> tuple[Response, int]:
        """
        Rolling performance statistics (last 1000 requests).

        Useful for internal dashboards; in production gate behind auth
        or expose only on an internal VPC endpoint.
        """
        stats = monitor.get_stats()
        drift = monitor.check_drift()
        return jsonify({"status": "success", "metrics": stats, "drift": drift}), 200

    @app.route("/info", methods=["GET"])
    def info() -> tuple[Response, int]:
        """Return model version and registry metadata."""
        try:
            meta = _get_predictor().metadata
        except Exception:
            meta = {}
        return jsonify({"status": "success", "model_info": meta}), 200

    @app.errorhandler(404)
    def not_found(e: Any) -> tuple[Response, int]:
        return jsonify({"status": "error", "message": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e: Any) -> tuple[Response, int]:
        return jsonify({"status": "error", "message": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(e: Any) -> tuple[Response, int]:
        logger.exception("Unhandled server error")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

    return app


# ─── WSGI entry point ─────────────────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    logger.info(
        f"Starting Flask dev server on "
        f"{config.api.host}:{config.api.port}  debug={config.api.debug}"
    )
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
    )
