"""
Model Registry Module
=====================
Handles serialization, versioning, and retrieval of trained models.

Responsibilities
----------------
* Saving the best model to disk (joblib) as ``models/model_v1.pkl``.
* Writing a JSON metadata file alongside the model.
* Loading a model artifact for inference.
* Listing available registered models.

Metadata schema (models/metadata.json)
----------------------------------------
{
  "model_version"      : "v1",
  "model_name"         : "RandomForest",
  "trained_at"         : "2024-01-15T10:30:00",
  "metrics"            : { "accuracy": 0.93, "f1": 0.91, ... },
  "cv_score"           : 0.90,
  "best_params"        : { ... },
  "feature_count"      : 14,
  "training_samples"   : 1600,
  "model_file"         : "model_v1.pkl",
  "preprocessor_file"  : "preprocessor_v1.pkl"
}

In a production AWS setup this module would also:
  * Upload model artifact to S3 (boto3 s3.upload_file)
  * Register the model in MLflow Model Registry
  * Tag the EC2 AMI or ECS task with the new version
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.config.config import config
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Saves, loads, and lists ML model artifacts.
    """

    def __init__(self) -> None:
        self.models_path = config.model.models_path
        self.model_filename = config.model.model_filename
        self.metadata_filename = config.model.metadata_filename
        self.model_version = config.api.model_version

    # ── Public API ────────────────────────────────────────────────────────────

    def save(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, Any],
        cv_score: float,
        best_params: Dict[str, Any],
        training_samples: int,
        feature_count: int,
    ) -> Path:
        """
        Persist the best model and write metadata JSON.

        Parameters
        ----------
        model            : Fitted sklearn estimator.
        model_name       : Human-readable name ("RandomForest").
        metrics          : Evaluation metrics dict from ModelEvaluator.
        cv_score         : Mean cross-validated score from training.
        best_params      : Best hyperparameters from GridSearchCV.
        training_samples : Number of rows used for training.
        feature_count    : Number of input features after preprocessing.

        Returns
        -------
        Path to saved model file.
        """
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Save binary artifact
        model_path = self.models_path / self.model_filename
        joblib.dump(model, model_path)
        logger.info(f"Model artifact saved ➜ {model_path}")

        # Write metadata
        metadata = {
            "model_version": self.model_version,
            "model_name": model_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "cv_score": round(float(cv_score), 4),
            "best_params": best_params,
            "feature_count": feature_count,
            "training_samples": training_samples,
            "model_file": self.model_filename,
            "preprocessor_file": config.model.preprocessor_filename,
        }
        self._write_metadata(metadata)

        logger.info(
            f"Model registered — version={self.model_version}, "
            f"name={model_name}, "
            f"test_f1={metrics.get('f1', 'n/a')}"
        )
        return model_path

    def load(self) -> Any:
        """Load the persisted model artifact from disk."""
        model_path = self.models_path / self.model_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model found at {model_path}. Run the training pipeline first."
            )
        model = joblib.load(model_path)
        logger.info(f"Model loaded ← {model_path}")
        return model

    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata JSON. Returns empty dict if file doesn't exist."""
        meta_path = self.models_path / self.metadata_filename
        if not meta_path.exists():
            logger.warning(f"Metadata file not found: {meta_path}")
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded ← {meta_path}")
        return metadata

    def model_exists(self) -> bool:
        """Return True if a trained model artifact is available."""
        return (self.models_path / self.model_filename).exists()

    def list_models(self) -> list[str]:
        """List all .pkl model files in the models directory."""
        return [p.name for p in self.models_path.glob("*.pkl")]

    # ── AWS simulation ────────────────────────────────────────────────────────

    def simulate_s3_upload(self) -> None:
        """
        Log what an S3 upload would look like in production.

        In production replace with:
            import boto3
            s3 = boto3.client("s3")
            s3.upload_file(str(model_path), config.aws.s3_bucket, config.aws.s3_model_key)
        """
        logger.info(
            f"[AWS SIMULATION] Would upload model to "
            f"s3://{config.aws.s3_bucket}/{config.aws.s3_model_key}"
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _write_metadata(self, metadata: Dict[str, Any]) -> None:
        meta_path = self.models_path / self.metadata_filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata written ➜ {meta_path}")
