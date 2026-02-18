"""
Tests for Data Preprocessing & Feature Engineering
====================================================
Covers:
  * ColumnTransformer fits on numeric + categorical columns.
  * Transform produces correct output shape.
  * Preprocessor can be saved and reloaded.
  * Feature engineer adds expected derived columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture(scope="module")
def sample_splits():
    ing = DataIngestion()
    df = ing.generate_synthetic_dataset(n_samples=300)
    return ing.split_data(df)


@pytest.fixture(scope="module")
def X_train(sample_splits):
    return sample_splits[0]


@pytest.fixture(scope="module")
def X_test(sample_splits):
    return sample_splits[1]


class TestDataValidator:
    def test_valid_data_passes(self, X_train) -> None:
        ing = DataIngestion()
        df = ing.generate_synthetic_dataset(n_samples=100)
        validator = DataValidator()
        report = validator.validate(df)
        assert report["is_valid"] is True

    def test_missing_column_fails(self) -> None:
        df = pd.DataFrame({"income": [50000], "target": [0]})
        validator = DataValidator()
        report = validator.validate(df)
        assert report["is_valid"] is False
        assert any("Missing required columns" in e for e in report["errors"])

    def test_inference_mode_no_target_required(self, X_train) -> None:
        validator = DataValidator()
        report = validator.validate(X_train, is_inference=True)
        assert report["is_valid"] is True


class TestFeatureEngineer:
    def test_adds_expected_columns(self, X_train) -> None:
        fe = FeatureEngineer()
        out = fe.fit_transform(X_train)
        new_cols = set(out.columns) - set(X_train.columns)
        expected = {"loan_to_income", "income_per_account", "credit_tier",
                    "age_bracket", "high_risk_flag", "employment_stability"}
        assert expected.issubset(new_cols)

    def test_transform_matches_fit_transform_columns(self, X_train, X_test) -> None:
        fe = FeatureEngineer()
        out_train = fe.fit_transform(X_train)
        out_test = fe.transform(X_test)
        # Same set of columns (not necessarily same order)
        assert set(out_train.columns) == set(out_test.columns)

    def test_unfitted_transform_raises(self, X_train) -> None:
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(X_train)

    def test_loan_to_income_non_negative(self, X_train) -> None:
        fe = FeatureEngineer()
        out = fe.fit_transform(X_train)
        assert (out["loan_to_income"] >= 0).all()

    def test_high_risk_flag_is_binary(self, X_train) -> None:
        fe = FeatureEngineer()
        out = fe.fit_transform(X_train)
        assert set(out["high_risk_flag"].unique()).issubset({0, 1})


class TestDataPreprocessor:
    def test_fit_transform_output_shape(self, X_train, X_test) -> None:
        fe = FeatureEngineer()
        Xt = fe.fit_transform(X_train)
        Xs = fe.transform(X_test)

        proc = DataPreprocessor()
        out_train, out_test = proc.fit_transform(Xt, Xs)

        assert out_train.ndim == 2
        assert out_test.ndim == 2
        assert out_train.shape[0] == len(X_train)
        assert out_test.shape[0] == len(X_test)
        assert out_train.shape[1] == out_test.shape[1]

    def test_no_nans_after_transform(self, X_train, X_test) -> None:
        fe = FeatureEngineer()
        Xt = fe.fit_transform(X_train)
        Xs = fe.transform(X_test)

        proc = DataPreprocessor()
        out_train, out_test = proc.fit_transform(Xt, Xs)

        assert not np.isnan(out_train).any()
        assert not np.isnan(out_test).any()

    def test_unfitted_transform_raises(self, X_train) -> None:
        proc = DataPreprocessor()
        with pytest.raises(RuntimeError):
            proc.transform(X_train)

    def test_save_and_load(self, X_train, X_test, tmp_path) -> None:
        from src.config.config import config
        fe = FeatureEngineer()
        Xt = fe.fit_transform(X_train)
        Xs = fe.transform(X_test)

        proc = DataPreprocessor()
        proc._save_path = tmp_path / "preprocessor_test.pkl"
        out1, _ = proc.fit_transform(Xt, Xs)
        proc.save()

        proc2 = DataPreprocessor()
        proc2._save_path = tmp_path / "preprocessor_test.pkl"
        proc2.load()
        out2 = proc2.transform(Xt)

        assert np.allclose(out1, out2)
