"""
Data Validation Module
======================
Validates incoming DataFrames against a known schema before any
transformation happens.  Catches data quality issues early, preventing
silent model degradation.

Checks performed
----------------
1. Required columns present.
2. No fully empty columns.
3. Data-type compatibility (numeric / categorical).
4. Value-range constraints (e.g. age ≥ 0, credit_score in 300-850).
5. Missing-value threshold — raises if > ``max_missing_pct`` of any column is null.
6. Duplicate rows — warns but does not fail.
7. Class balance — warns if minority class is very rare.

Usage
-----
    validator = DataValidator()
    report = validator.validate(df)
    if not report["is_valid"]:
        raise ValueError(report["errors"])
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# ─── Expected schema ──────────────────────────────────────────────────────────

REQUIRED_COLUMNS: List[str] = [
    "income",
    "age",
    "loan_amount",
    "credit_score",
    "employment_years",
    "debt_ratio",
    "num_accounts",
    "employment_type",
    "target",
]

NUMERIC_COLUMNS: List[str] = [
    "income",
    "age",
    "loan_amount",
    "credit_score",
    "employment_years",
    "debt_ratio",
    "num_accounts",
]

CATEGORICAL_COLUMNS: List[str] = ["employment_type"]

VALID_EMPLOYMENT_TYPES = {"full_time", "part_time", "self_employed"}

COLUMN_RANGES: Dict[str, Dict[str, float]] = {
    "income": {"min": 0, "max": 10_000_000},
    "age": {"min": 18, "max": 120},
    "loan_amount": {"min": 1, "max": 10_000_000},
    "credit_score": {"min": 300, "max": 850},
    "employment_years": {"min": 0, "max": 60},
    "debt_ratio": {"min": 0.0, "max": 1.0},
    "num_accounts": {"min": 0, "max": 100},
}

MAX_MISSING_PCT: float = 0.30   # fail if > 30 % of any column is null


class DataValidator:
    """
    Validates a raw DataFrame for schema compliance and data quality.
    """

    def validate(
        self,
        df: pd.DataFrame,
        is_inference: bool = False,
    ) -> Dict[str, Any]:
        """
        Run all validation checks.

        Parameters
        ----------
        df           : The DataFrame to validate.
        is_inference : When True the 'target' column is not required
                       (prediction requests won't have it).

        Returns
        -------
        dict with keys:
            is_valid  bool
            errors    List[str]   — fatal issues
            warnings  List[str]   — non-fatal issues
        """
        logger.info(f"Validating data  [{df.shape[0]:,} rows × {df.shape[1]} cols]")
        errors: List[str] = []
        warnings: List[str] = []

        required = (
            [c for c in REQUIRED_COLUMNS if c != "target"]
            if is_inference
            else REQUIRED_COLUMNS
        )

        self._check_required_columns(df, required, errors)
        self._check_missing_values(df, errors, warnings)
        self._check_numeric_types(df, errors)
        self._check_value_ranges(df, errors, warnings)
        self._check_categorical_values(df, warnings)
        self._check_duplicates(df, warnings)

        if not is_inference and "target" in df.columns:
            self._check_class_balance(df, warnings)

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Validation passed.")
        else:
            logger.error(f"Validation failed — {len(errors)} error(s)")
            for e in errors:
                logger.error(f"  ERROR   | {e}")
        for w in warnings:
            logger.warning(f"  WARNING | {w}")

        return {"is_valid": is_valid, "errors": errors, "warnings": warnings}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_required_columns(
        self, df: pd.DataFrame, required: List[str], errors: List[str]
    ) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

    def _check_missing_values(
        self,
        df: pd.DataFrame,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        null_pct = df.isnull().mean()
        for col, pct in null_pct.items():
            if pct > MAX_MISSING_PCT:
                errors.append(
                    f"Column '{col}' has {pct:.1%} missing values (limit: {MAX_MISSING_PCT:.0%})"
                )
            elif pct > 0:
                warnings.append(f"Column '{col}' has {pct:.1%} missing values")

    def _check_numeric_types(
        self, df: pd.DataFrame, errors: List[str]
    ) -> None:
        for col in NUMERIC_COLUMNS:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Attempt soft-cast check
                try:
                    pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    errors.append(
                        f"Column '{col}' cannot be cast to numeric (dtype={df[col].dtype})"
                    )

    def _check_value_ranges(
        self,
        df: pd.DataFrame,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        for col, bounds in COLUMN_RANGES.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            n_out = ((series < bounds["min"]) | (series > bounds["max"])).sum()
            if n_out > 0:
                pct = n_out / len(df)
                msg = (
                    f"Column '{col}' has {n_out:,} ({pct:.1%}) values outside "
                    f"[{bounds['min']}, {bounds['max']}]"
                )
                if pct > 0.10:
                    errors.append(msg)
                else:
                    warnings.append(msg)

    def _check_categorical_values(
        self, df: pd.DataFrame, warnings: List[str]
    ) -> None:
        if "employment_type" in df.columns:
            unique_vals = set(df["employment_type"].dropna().unique())
            unknown = unique_vals - VALID_EMPLOYMENT_TYPES
            if unknown:
                warnings.append(
                    f"Unknown 'employment_type' values found: {unknown}"
                )

    def _check_duplicates(
        self, df: pd.DataFrame, warnings: List[str]
    ) -> None:
        n_dup = df.duplicated().sum()
        if n_dup > 0:
            warnings.append(f"{n_dup:,} duplicate rows detected")

    def _check_class_balance(
        self, df: pd.DataFrame, warnings: List[str]
    ) -> None:
        rate = df["target"].mean()
        if rate < 0.05:
            warnings.append(
                f"Severe class imbalance: positive_rate={rate:.2%}. "
                "Consider SMOTE, class_weight='balanced', or resampling."
            )
        elif rate < 0.15:
            warnings.append(
                f"Moderate class imbalance: positive_rate={rate:.2%}. "
                "Monitor precision-recall carefully."
            )
