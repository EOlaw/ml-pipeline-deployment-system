"""
Feature Engineering Module
===========================
Creates domain-aware derived features from the raw DataFrame **before**
the preprocessing pipeline scales / encodes anything.

Why feature engineering matters
---------------------------------
Raw features are often measured in different units and lack
the ratios or interaction terms that directly capture business risk:

  * loan-to-income ratio   — classic credit-risk signal
  * income per account     — wealth diversification proxy
  * credit tier (ordinal)  — bucketed credit score for non-linear splits
  * age bracket            — life-stage proxy

All features returned are still in their raw scale; the DataPreprocessor
handles scaling / encoding in the next step.

Usage
-----
    engineer = FeatureEngineer()
    X_train_fe = engineer.fit_transform(X_train)
    X_test_fe  = engineer.transform(X_test)        # same column set
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# Credit-score tier boundaries (inclusive lower bound)
_CREDIT_TIER_BINS = [300, 580, 670, 740, 800, 850]
_CREDIT_TIER_LABELS = ["very_poor", "fair", "good", "very_good", "exceptional"]

# Age bracket boundaries
_AGE_BINS = [0, 25, 35, 45, 55, 120]
_AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56+"]


class FeatureEngineer:
    """
    Applies deterministic domain-knowledge feature transformations.

    After ``fit_transform``, the list of added columns is recorded so
    ``transform`` can reproduce the identical column order for inference.
    """

    def __init__(self) -> None:
        self._new_columns: List[str] = []
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive features from ``X_train`` and record new column names."""
        logger.info("Running feature engineering on training data …")
        X_out = self._engineer(X)
        self._new_columns = [c for c in X_out.columns if c not in X.columns]
        self._fitted = True
        logger.info(
            f"Feature engineering complete — "
            f"{len(self._new_columns)} new columns: {self._new_columns}"
        )
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same transformations to new data.
        Ensures output column set matches training (important for the
        fitted ColumnTransformer expecting the same schema).
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureEngineer has not been fitted. Call fit_transform() first."
            )
        return self._engineer(X)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Core transformation logic applied to both train and inference data.
        Creates a copy to avoid mutating the caller's DataFrame.
        """
        df = X.copy()

        # 1. Loan-to-income ratio — how leveraged is the borrower?
        if "loan_amount" in df.columns and "income" in df.columns:
            df["loan_to_income"] = np.where(
                df["income"] > 0,
                df["loan_amount"] / df["income"],
                0.0,
            ).round(4)

        # 2. Income per bank account — wealth distribution indicator
        if "income" in df.columns and "num_accounts" in df.columns:
            df["income_per_account"] = np.where(
                df["num_accounts"] > 0,
                df["income"] / (df["num_accounts"] + 1),   # +1 avoids /0
                df["income"],
            ).round(2)

        # 3. Credit risk tier — non-linear credit signal
        if "credit_score" in df.columns:
            df["credit_tier"] = pd.cut(
                df["credit_score"],
                bins=_CREDIT_TIER_BINS,
                labels=_CREDIT_TIER_LABELS,
                right=True,
                include_lowest=True,
            ).astype(str)

        # 4. Age bracket — proxy for life stage and financial stability
        if "age" in df.columns:
            df["age_bracket"] = pd.cut(
                df["age"],
                bins=_AGE_BINS,
                labels=_AGE_LABELS,
                right=True,
                include_lowest=True,
            ).astype(str)

        # 5. High-risk flag — simple rule-based risk signal
        if "debt_ratio" in df.columns and "credit_score" in df.columns:
            df["high_risk_flag"] = (
                (df["debt_ratio"] > 0.5) & (df["credit_score"] < 640)
            ).astype(int)

        # 6. Employment stability score — composite ordinal
        if "employment_years" in df.columns:
            df["employment_stability"] = pd.cut(
                df["employment_years"],
                bins=[-1, 1, 3, 7, 15, 60],
                labels=[0, 1, 2, 3, 4],   # ordinal
            ).astype(float)

        return df

    @property
    def new_feature_names(self) -> List[str]:
        """Names of features created by this engineer (after fit)."""
        return self._new_columns.copy()
