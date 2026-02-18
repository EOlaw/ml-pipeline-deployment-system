"""
Tests for Data Ingestion Module
================================
Covers:
  * Synthetic dataset generation (shape, columns, dtypes, target distribution).
  * CSV round-trip (save raw ➜ reload ➜ compare).
  * Train/test split (stratification, correct proportions).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingestion import DataIngestion


@pytest.fixture
def ingestion() -> DataIngestion:
    return DataIngestion()


@pytest.fixture
def synthetic_df(ingestion: DataIngestion) -> pd.DataFrame:
    return ingestion.generate_synthetic_dataset(n_samples=200)


class TestSyntheticDataset:
    def test_shape(self, synthetic_df: pd.DataFrame) -> None:
        assert synthetic_df.shape[0] == 200
        assert synthetic_df.shape[1] == 9   # 8 features + target

    def test_required_columns_present(self, synthetic_df: pd.DataFrame) -> None:
        expected = {
            "income", "age", "loan_amount", "credit_score",
            "employment_years", "debt_ratio", "num_accounts",
            "employment_type", "target",
        }
        assert expected.issubset(set(synthetic_df.columns))

    def test_target_is_binary(self, synthetic_df: pd.DataFrame) -> None:
        assert set(synthetic_df["target"].unique()).issubset({0, 1})

    def test_no_nulls(self, synthetic_df: pd.DataFrame) -> None:
        assert synthetic_df.isnull().sum().sum() == 0

    def test_employment_type_valid_values(self, synthetic_df: pd.DataFrame) -> None:
        valid = {"full_time", "part_time", "self_employed"}
        assert set(synthetic_df["employment_type"].unique()).issubset(valid)

    def test_credit_score_range(self, synthetic_df: pd.DataFrame) -> None:
        assert synthetic_df["credit_score"].between(300, 850).all()

    def test_age_range(self, synthetic_df: pd.DataFrame) -> None:
        assert synthetic_df["age"].between(22, 65).all()

    def test_debt_ratio_range(self, synthetic_df: pd.DataFrame) -> None:
        assert synthetic_df["debt_ratio"].between(0.0, 1.0).all()


class TestTrainTestSplit:
    def test_split_size(self, ingestion: DataIngestion, synthetic_df: pd.DataFrame) -> None:
        X_train, X_test, y_train, y_test = ingestion.split_data(synthetic_df)
        total = len(X_train) + len(X_test)
        assert total == len(synthetic_df)
        assert abs(len(X_test) / total - ingestion.test_size) < 0.05

    def test_stratification(self, ingestion: DataIngestion, synthetic_df: pd.DataFrame) -> None:
        X_train, X_test, y_train, y_test = ingestion.split_data(synthetic_df)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.10   # within 10 pp

    def test_no_target_in_X(self, ingestion: DataIngestion, synthetic_df: pd.DataFrame) -> None:
        X_train, X_test, y_train, y_test = ingestion.split_data(synthetic_df)
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

    def test_y_series_name(self, ingestion: DataIngestion, synthetic_df: pd.DataFrame) -> None:
        _, _, y_train, y_test = ingestion.split_data(synthetic_df)
        assert y_train.name == "target"


class TestCSVRoundTrip:
    def test_save_and_reload(
        self,
        ingestion: DataIngestion,
        synthetic_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        # Override raw path for this test
        ingestion.raw_path = tmp_path
        path = ingestion.save_raw_data(synthetic_df, "test_raw.csv")

        reloaded = pd.read_csv(path)
        assert reloaded.shape == synthetic_df.shape
        assert list(reloaded.columns) == list(synthetic_df.columns)

    def test_load_from_csv(
        self,
        ingestion: DataIngestion,
        synthetic_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        csv_path = tmp_path / "sample.csv"
        synthetic_df.to_csv(csv_path, index=False)

        loaded = ingestion.load_from_csv(csv_path)
        assert loaded.shape == synthetic_df.shape

    def test_load_missing_csv_raises(self, ingestion: DataIngestion) -> None:
        with pytest.raises(FileNotFoundError):
            ingestion.load_from_csv("/non/existent/path.csv")
