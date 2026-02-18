"""
Model Evaluation Module
=======================
Computes a comprehensive set of classification metrics against the
held-out test set and produces a human-readable report.

Metrics computed
----------------
* Accuracy
* Precision  (weighted)
* Recall     (weighted)
* F1-score   (weighted)
* ROC-AUC    (requires predict_proba)
* Confusion matrix

Usage
-----
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, X_test, y_test)
    evaluator.print_report(report)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.monitoring.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates a fitted classifier on a test split."""

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: pd.Series,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Compute full evaluation metrics.

        Parameters
        ----------
        model      : Fitted sklearn estimator with predict / predict_proba.
        X_test     : Preprocessed test features.
        y_test     : True labels.
        model_name : Label used in log messages and the returned dict.

        Returns
        -------
        dict with keys: model_name, accuracy, precision, recall, f1,
                        roc_auc, confusion_matrix, n_samples,
                        positive_rate_actual, positive_rate_predicted.
        """
        logger.info(f"Evaluating model: {model_name}  [{len(y_test):,} samples]")

        y_pred = model.predict(X_test)
        y_true = np.asarray(y_test)

        # Probability scores for ROC-AUC
        roc_auc = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            try:
                roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
            except ValueError:
                roc_auc = None   # edge case: only one class in y_test

        accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
        precision = round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        recall = round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        f1 = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
        cm = confusion_matrix(y_true, y_pred).tolist()

        report = {
            "model_name": model_name,
            "n_samples": int(len(y_true)),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "positive_rate_actual": round(float(y_true.mean()), 4),
            "positive_rate_predicted": round(float(y_pred.mean()), 4),
        }

        logger.info(
            f"{model_name} results — "
            f"accuracy={accuracy:.4f}  precision={precision:.4f}  "
            f"recall={recall:.4f}  f1={f1:.4f}  roc_auc={roc_auc}"
        )
        return report

    def evaluate_all(
        self,
        cv_results: list,
        X_test: np.ndarray,
        y_test: pd.Series,
    ) -> list[Dict[str, Any]]:
        """
        Evaluate every candidate model from the trainer's cv_results list.

        Returns list of evaluation dicts, sorted by F1 descending.
        """
        reports = []
        for result in cv_results:
            report = self.evaluate(
                model=result["estimator"],
                X_test=X_test,
                y_test=y_test,
                model_name=result["model_name"],
            )
            reports.append(report)

        reports.sort(key=lambda r: r["f1"], reverse=True)
        return reports

    def print_report(self, report: Dict[str, Any]) -> None:
        """Pretty-print a single evaluation report."""
        separator = "─" * 55
        print(f"\n{separator}")
        print(f"  Model Evaluation Report — {report['model_name']}")
        print(separator)
        print(f"  Samples      : {report['n_samples']:,}")
        print(f"  Accuracy     : {report['accuracy']:.4f}")
        print(f"  Precision    : {report['precision']:.4f}")
        print(f"  Recall       : {report['recall']:.4f}")
        print(f"  F1-score     : {report['f1']:.4f}")
        if report.get("roc_auc") is not None:
            print(f"  ROC-AUC      : {report['roc_auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        for row in report["confusion_matrix"]:
            print(f"    {row}")
        print(separator + "\n")

    def comparison_dataframe(
        self, reports: list[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Return a tidy DataFrame comparing all evaluated models."""
        rows = []
        for r in reports:
            rows.append(
                {
                    "model": r["model_name"],
                    "accuracy": r["accuracy"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "roc_auc": r.get("roc_auc"),
                }
            )
        return pd.DataFrame(rows).sort_values("f1", ascending=False)
