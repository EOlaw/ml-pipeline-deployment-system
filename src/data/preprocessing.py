"""
Data Preprocessing Module
=========================
Transforms raw DataFrames into model-ready feature matrices.

Pipeline steps
--------------
Numeric columns  → median imputation → StandardScaler
Categorical cols → constant imputation → OneHotEncoder

The fitted ColumnTransformer (preprocessor) is returned along with the
transformed arrays so it can be:
  * Saved to disk and reused at inference time.
  * Applied identically to train AND test data.

Responsibilities of this module
--------------------------------
* Missing-value imputation.
* Feature scaling.
* Categorical encoding.
* Saving/loading the fitted preprocessor.

What this module does NOT do
------------------------------
* Feature creation (that is src/features/feature_engineering.py).
* Model training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.config import config
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Fits and applies a sklearn ColumnTransformer preprocessing pipeline.

    Attributes
    ----------
    preprocessor : fitted ColumnTransformer (available after ``fit_transform``)
    """

    def __init__(self) -> None:
        self.numeric_features = list(config.model.numeric_features)
        self.categorical_features = list(config.model.categorical_features)
        self.preprocessor: Optional[ColumnTransformer] = None
        self._save_path = (
            config.model.models_path / config.model.preprocessor_filename
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the preprocessor from ``X_train``, apply to both splits.

        Parameters
        ----------
        X_train : training features
        X_test  : test features

        Returns
        -------
        (X_train_processed, X_test_processed) as numpy arrays
        """
        logger.info("Fitting preprocessing pipeline …")
        self.preprocessor = self._build_pipeline(X_train)
        self.preprocessor.fit(X_train)

        X_train_proc = self.preprocessor.transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        logger.info(
            f"Preprocessing complete — "
            f"train shape: {X_train_proc.shape}, "
            f"test shape:  {X_test_proc.shape}"
        )
        return X_train_proc, X_test_proc

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply a *fitted* preprocessor to new data (e.g. inference request).

        Raises
        ------
        RuntimeError if the preprocessor has not been fitted/loaded yet.
        """
        if self.preprocessor is None:
            raise RuntimeError(
                "Preprocessor not fitted. Call fit_transform() or load()."
            )
        return self.preprocessor.transform(X)

    def save(self) -> Path:
        """Persist the fitted preprocessor to disk with joblib."""
        if self.preprocessor is None:
            raise RuntimeError("Cannot save — preprocessor not fitted.")
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, self._save_path)
        logger.info(f"Preprocessor saved ➜ {self._save_path}")
        return self._save_path

    def load(self) -> "DataPreprocessor":
        """Load a previously saved preprocessor from disk."""
        if not self._save_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {self._save_path}. "
                "Run the training pipeline first."
            )
        self.preprocessor = joblib.load(self._save_path)
        logger.info(f"Preprocessor loaded ← {self._save_path}")
        return self

    def get_feature_names(self) -> list[str]:
        """Return output feature names after transformation (for inspection)."""
        if self.preprocessor is None:
            return []
        try:
            return list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            return []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build the ColumnTransformer pipeline, auto-detecting which
        configured columns actually exist in ``X``.

        Remaining columns (from feature engineering) are classified by dtype:
          * numeric dtype  → added to numeric pipeline
          * object/string  → added to categorical pipeline (OneHotEncoder)
        """
        existing_num = [c for c in self.numeric_features if c in X.columns]
        existing_cat = [c for c in self.categorical_features if c in X.columns]
        already_handled = set(existing_num + existing_cat)

        # Auto-classify extra columns produced by feature engineering
        extra_num = []
        extra_cat = []
        for col in X.columns:
            if col in already_handled:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                extra_num.append(col)
            else:
                extra_cat.append(col)

        all_num = existing_num + extra_num
        all_cat = existing_cat + extra_cat

        logger.info(
            f"Pipeline — numeric={all_num}, "
            f"categorical={all_cat}"
        )

        # Reassign for use below
        existing_num, existing_cat = all_num, all_cat
        remaining = []  # nothing left unhandled

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="first",          # avoid multicollinearity
                    ),
                ),
            ]
        )

        transformers = [
            ("numeric", numeric_pipeline, existing_num),
        ]
        if existing_cat:
            transformers.append(("categorical", categorical_pipeline, existing_cat))
        if remaining:
            transformers.append(("passthrough", "passthrough", remaining))

        return ColumnTransformer(transformers=transformers, remainder="drop")
