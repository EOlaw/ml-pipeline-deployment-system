"""
Data Ingestion Module
=====================
Responsible for acquiring raw data from multiple sources:
  * Synthetic generator  — creates a realistic loan-default dataset for demos.
  * CSV file loader      — reads any CSV from disk.
  * Sklearn datasets     — loads built-in sklearn datasets (e.g. breast_cancer).

After loading, raw data is:
  1. Saved to  data/raw/raw_data.csv
  2. Split into train / test sets (stratified)
  3. Returned as (X_train, X_test, y_train, y_test)

The synthetic loan-default dataset contains:
  income, age, loan_amount, credit_score, employment_years,
  debt_ratio, num_accounts  (numeric)
  employment_type            (categorical)
  target                     (0 = no default, 1 = default)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.config.config import config
from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Loads, saves, and splits raw data for the ML pipeline."""

    def __init__(self) -> None:
        self.raw_path = config.data.raw_data_path
        self.random_state = config.model.random_state
        self.test_size = config.data.test_size
        self.target_col = config.data.target_column

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        source: str = "synthetic",
        filepath: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Execute the full ingestion pipeline.

        Parameters
        ----------
        source   : "synthetic" | "csv" | "sklearn"
        filepath : Required when source="csv"

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        logger.info(f"Starting data ingestion  [source={source}]")

        if source == "csv" and filepath:
            df = self.load_from_csv(filepath)
        elif source == "sklearn":
            df = self.load_sklearn_dataset()
        else:
            df = self.generate_synthetic_dataset()

        self.save_raw_data(df)
        splits = self.split_data(df)
        logger.info("Data ingestion completed successfully.")
        return splits

    def load_from_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load raw data from a CSV file on disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading CSV  [{filepath}]")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df

    def load_sklearn_dataset(self, dataset_name: str = "breast_cancer") -> pd.DataFrame:
        """
        Load a sklearn built-in dataset and return it as a DataFrame.
        Supported: 'breast_cancer'
        """
        logger.info(f"Loading sklearn dataset: {dataset_name}")
        if dataset_name == "breast_cancer":
            raw = load_breast_cancer()
            cols = [c.replace(" ", "_").replace("(", "").replace(")", "") for c in raw.feature_names]
            df = pd.DataFrame(raw.data, columns=cols)
            df["target"] = raw.target
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        logger.info(f"Dataset loaded — {df.shape[0]:,} samples, {df.shape[1]} columns")
        return df

    def generate_synthetic_dataset(self, n_samples: int = 2_000) -> pd.DataFrame:
        """
        Generate a realistic synthetic loan-default dataset.

        Features
        --------
        * income           — annual income (normal distribution, clipped)
        * age              — borrower age (uniform 22-65)
        * loan_amount      — requested loan (normal, clipped)
        * credit_score     — FICO-like score (normal, clipped 300-850)
        * employment_years — years at current employer (exponential)
        * debt_ratio       — monthly debt / monthly income (beta)
        * num_accounts     — number of open bank accounts (Poisson)
        * employment_type  — full_time / part_time / self_employed

        Target
        ------
        * target = 1 (default) probability driven by logistic combination
                   of the above features — higher income / credit score ➡ lower risk.
        """
        logger.info(f"Generating synthetic loan dataset  [n={n_samples:,}]")
        rng = np.random.default_rng(self.random_state)

        income = rng.normal(60_000, 20_000, n_samples).clip(15_000, 200_000)
        age = rng.integers(22, 66, n_samples).astype(float)
        loan_amount = rng.normal(15_000, 8_000, n_samples).clip(1_000, 50_000)
        credit_score = rng.normal(680, 80, n_samples).clip(300, 850)
        employment_years = rng.exponential(5, n_samples).clip(0, 40)
        debt_ratio = rng.beta(2, 5, n_samples)
        num_accounts = rng.poisson(3, n_samples).clip(0, 15).astype(float)
        employment_type = rng.choice(
            ["full_time", "part_time", "self_employed"],
            size=n_samples,
            p=[0.65, 0.20, 0.15],
        )

        # Logistic default probability: higher income/score ➡ lower default risk
        logit = (
            2.0
            - 1.5e-5 * income
            - 2.5e-3 * credit_score
            + 5.0e-6 * loan_amount
            + 1.5 * debt_ratio
            - 3.0e-2 * employment_years
        )
        default_prob = 1.0 / (1.0 + np.exp(-logit))
        target = (rng.random(n_samples) < default_prob).astype(int)

        df = pd.DataFrame(
            {
                "income": income.round(2),
                "age": age.astype(int),
                "loan_amount": loan_amount.round(2),
                "credit_score": credit_score.astype(int),
                "employment_years": employment_years.round(1),
                "debt_ratio": debt_ratio.round(4),
                "num_accounts": num_accounts.astype(int),
                "employment_type": employment_type,
                "target": target,
            }
        )

        default_rate = df["target"].mean()
        logger.info(
            f"Synthetic dataset ready — {df.shape}, default_rate={default_rate:.2%}"
        )
        return df

    def save_raw_data(self, df: pd.DataFrame, filename: Optional[str] = None) -> Path:
        """Persist raw DataFrame to data/raw/."""
        self.raw_path.mkdir(parents=True, exist_ok=True)
        filename = filename or config.data.raw_filename
        out = self.raw_path / filename
        df.to_csv(out, index=False)
        logger.info(f"Raw data saved ➜ {out}")
        return out

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified train/test split."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(
            f"Split — train={X_train.shape[0]:,}  test={X_test.shape[0]:,}  "
            f"pos_rate={y_train.mean():.2%}"
        )
        return X_train, X_test, y_train, y_test
