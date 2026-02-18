"""
Tests for Model Training & Evaluation
=======================================
Covers:
  * Training completes without error.
  * Best model selected is one of the expected candidates.
  * CV results are sorted by descending score.
  * Evaluation metrics are in valid ranges.
  * Model can predict on processed test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer


@pytest.fixture(scope="module")
def processed_splits():
    """Run the full ingestion ➜ feature engineering ➜ preprocessing pipeline."""
    ing = DataIngestion()
    df = ing.generate_synthetic_dataset(n_samples=500)
    X_train, X_test, y_train, y_test = ing.split_data(df)

    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    proc = DataPreprocessor()
    X_train_proc, X_test_proc = proc.fit_transform(X_train_fe, X_test_fe)

    return X_train_proc, X_test_proc, y_train, y_test


class TestModelTrainer:
    def test_training_completes(self, processed_splits) -> None:
        X_train, _, y_train, _ = processed_splits
        trainer = ModelTrainer()
        best_model, cv_results = trainer.train(X_train, y_train)
        assert best_model is not None
        assert len(cv_results) >= 2

    def test_best_model_name(self, processed_splits) -> None:
        X_train, _, y_train, _ = processed_splits
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        assert trainer.best_model_name in ("RandomForest", "LogisticRegression")

    def test_cv_results_sorted_descending(self, processed_splits) -> None:
        X_train, _, y_train, _ = processed_splits
        trainer = ModelTrainer()
        _, cv_results = trainer.train(X_train, y_train)
        scores = [r[f"cv_{trainer.scoring}_mean"] for r in cv_results]
        assert scores == sorted(scores, reverse=True)

    def test_cv_score_in_valid_range(self, processed_splits) -> None:
        X_train, _, y_train, _ = processed_splits
        trainer = ModelTrainer()
        _, cv_results = trainer.train(X_train, y_train)
        for r in cv_results:
            assert 0.0 <= r[f"cv_{trainer.scoring}_mean"] <= 1.0

    def test_comparison_table_shape(self, processed_splits) -> None:
        X_train, _, y_train, _ = processed_splits
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        table = trainer.get_comparison_table()
        assert len(table) >= 2
        assert "model_name" in table.columns


class TestModelEvaluator:
    def test_evaluate_returns_required_keys(self, processed_splits) -> None:
        X_train, X_test, y_train, y_test = processed_splits
        trainer = ModelTrainer()
        best_model, _ = trainer.train(X_train, y_train)

        evaluator = ModelEvaluator()
        report = evaluator.evaluate(best_model, X_test, y_test)

        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert key in report

    def test_metrics_in_valid_range(self, processed_splits) -> None:
        X_train, X_test, y_train, y_test = processed_splits
        trainer = ModelTrainer()
        best_model, _ = trainer.train(X_train, y_train)

        evaluator = ModelEvaluator()
        report = evaluator.evaluate(best_model, X_test, y_test)

        for metric in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= report[metric] <= 1.0

    def test_roc_auc_present(self, processed_splits) -> None:
        X_train, X_test, y_train, y_test = processed_splits
        trainer = ModelTrainer()
        best_model, _ = trainer.train(X_train, y_train)

        evaluator = ModelEvaluator()
        report = evaluator.evaluate(best_model, X_test, y_test)
        # Should be present since both models have predict_proba
        assert report.get("roc_auc") is not None
        assert 0.0 <= report["roc_auc"] <= 1.0

    def test_comparison_dataframe(self, processed_splits) -> None:
        X_train, X_test, y_train, y_test = processed_splits
        trainer = ModelTrainer()
        _, cv_results = trainer.train(X_train, y_train)

        evaluator = ModelEvaluator()
        reports = evaluator.evaluate_all(cv_results, X_test, y_test)
        df = evaluator.comparison_dataframe(reports)

        assert len(df) >= 2
        assert "model" in df.columns
        assert "f1" in df.columns
