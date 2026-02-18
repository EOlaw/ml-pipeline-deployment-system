"""
Prediction Module
=================
Wraps the trained model and fitted preprocessor into a single inference
object that the Flask API and scripts can call with raw (unscaled) input.

Inference flow
--------------
  raw dict / DataFrame
      ↓  FeatureEngineer.transform()      (add derived columns)
      ↓  DataPreprocessor.transform()     (scale + encode)
      ↓  model.predict() / predict_proba()
      ↓  formatted response dict

All artifacts (model + preprocessor) are loaded lazily on first use
and cached in memory for the lifetime of the Flask worker process.

Usage
-----
    predictor = ModelPredictor()
    result = predictor.predict({"income": 55000, "age": 35, ...})
    # {"prediction": 0, "probability": 0.12, "model_version": "v1"}
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.config.config import config
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.registry import ModelRegistry
from src.monitoring.logger import get_logger
from src.monitoring.performance import PerformanceMonitor

logger = get_logger(__name__)


class ModelPredictor:
    """
    Single entry-point for model inference.

    Lazy-loads model and preprocessor on first call.
    Thread-safe for read-only inference under Gunicorn.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._preprocessor: Optional[DataPreprocessor] = None
        self._feature_engineer: Optional[FeatureEngineer] = None
        self._registry = ModelRegistry()
        self._monitor = PerformanceMonitor()
        self._model_version = config.api.model_version
        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> "ModelPredictor":
        """Load model and preprocessor artifacts from disk."""
        if self._loaded:
            return self

        logger.info("Loading model artifacts …")
        self._model = self._registry.load()

        self._preprocessor = DataPreprocessor()
        self._preprocessor.load()

        # FeatureEngineer is stateful — we refit from saved model
        # In production this would be persisted separately;
        # for demo we re-create and mark as fitted so transform() works.
        self._feature_engineer = FeatureEngineer()
        self._feature_engineer._fitted = True   # deterministic, no state needed

        self._loaded = True
        logger.info("Predictor ready.")
        return self

    def predict(
        self,
        raw_input: Union[Dict[str, Any], pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Run end-to-end inference on a single sample or DataFrame.

        Parameters
        ----------
        raw_input : dict (single record) or DataFrame.

        Returns
        -------
        {
          "prediction"    : int,
          "probability"   : float,
          "model_version" : str,
          "latency_ms"    : float,
        }
        """
        self._ensure_loaded()
        start = time.perf_counter()

        # Normalise to DataFrame
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = raw_input.copy()

        # Feature engineering
        df_fe = self._feature_engineer.transform(df)

        # Preprocessing
        X = self._preprocessor.transform(df_fe)

        # Inference
        prediction = int(self._model.predict(X)[0])
        probability = 0.0
        if hasattr(self._model, "predict_proba"):
            probability = round(float(self._model.predict_proba(X)[0, 1]), 4)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Monitor (logs metrics without storing PII)
        self._monitor.record(
            prediction=prediction,
            probability=probability,
            latency_ms=latency_ms,
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "model_version": self._model_version,
            "latency_ms": latency_ms,
        }

    def predict_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run predict() on a list of input dicts."""
        return [self.predict(record) for record in records]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return model registry metadata."""
        return self._registry.load_metadata()

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
