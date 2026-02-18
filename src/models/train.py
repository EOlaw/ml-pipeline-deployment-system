"""
Model Training Module
=====================
Trains multiple scikit-learn classifiers with cross-validation and
hyperparameter tuning via GridSearchCV, then returns the best model.

Models trained
--------------
1. Random Forest Classifier      — ensemble, handles non-linearity & interaction.
2. Logistic Regression           — linear baseline, highly interpretable.

Selection criterion
-------------------
The model with the highest mean cross-validated F1 score on the training
set is selected.  F1 is preferred over accuracy for class-imbalanced
datasets (relevant for loan default prediction).

Hyperparameter search
---------------------
GridSearchCV with stratified k-fold CV is used to find the best
hyper-parameter combination for each model.

Usage
-----
    trainer = ModelTrainer()
    best_model, cv_results = trainer.train(X_train_proc, y_train)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config.config import config
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# ─── Candidate model definitions ─────────────────────────────────────────────

def _get_candidate_models() -> List[Dict[str, Any]]:
    """Return list of (name, estimator, param_grid) dicts."""
    rs = config.model.random_state

    return [
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(
                random_state=rs,
                class_weight="balanced",
                n_jobs=config.model.n_jobs,
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        },
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(
                random_state=rs,
                class_weight="balanced",
                max_iter=1_000,
                solver="lbfgs",
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
            },
        },
    ]


class ModelTrainer:
    """
    Trains and compares multiple classifiers using GridSearchCV.

    Results are stored in ``self.cv_results`` as a list of dicts,
    sorted by descending mean CV score.
    """

    def __init__(self) -> None:
        self.cv_folds = config.model.cv_folds
        self.scoring = config.model.scoring_metric
        self.random_state = config.model.random_state
        self.cv_results: List[Dict[str, Any]] = []
        self.best_model: Any = None
        self.best_model_name: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Train all candidate models, run hyperparameter search, pick best.

        Parameters
        ----------
        X_train : preprocessed training features (numpy array)
        y_train : training labels

        Returns
        -------
        (best_estimator, cv_results_sorted_desc)
        """
        logger.info(
            f"Starting model training — "
            f"{len(_get_candidate_models())} candidates, "
            f"CV={self.cv_folds}, scoring='{self.scoring}'"
        )

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        results: List[Dict[str, Any]] = []

        for candidate in _get_candidate_models():
            name = candidate["name"]
            logger.info(f"  Training: {name} …")

            grid_search = GridSearchCV(
                estimator=candidate["estimator"],
                param_grid=candidate["param_grid"],
                cv=cv,
                scoring=self.scoring,
                n_jobs=config.model.n_jobs,
                refit=True,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)

            best_cv_score = grid_search.best_score_
            best_params = grid_search.best_params_

            # Also get all individual CV fold scores for reporting
            cv_scores = cross_val_score(
                grid_search.best_estimator_,
                X_train,
                y_train,
                cv=cv,
                scoring=self.scoring,
                n_jobs=config.model.n_jobs,
            )

            logger.info(
                f"  {name} — best_cv_{self.scoring}={best_cv_score:.4f} "
                f"(std={cv_scores.std():.4f}) | best_params={best_params}"
            )

            results.append(
                {
                    "model_name": name,
                    "estimator": grid_search.best_estimator_,
                    "best_params": best_params,
                    f"cv_{self.scoring}_mean": best_cv_score,
                    f"cv_{self.scoring}_std": cv_scores.std(),
                    "cv_fold_scores": cv_scores.tolist(),
                    "grid_search": grid_search,
                }
            )

        # Sort by score descending
        results.sort(key=lambda r: r[f"cv_{self.scoring}_mean"], reverse=True)
        self.cv_results = results

        best = results[0]
        self.best_model = best["estimator"]
        self.best_model_name = best["model_name"]

        logger.info(
            f"Best model selected: {self.best_model_name}  "
            f"(cv_{self.scoring}={best[f'cv_{self.scoring}_mean']:.4f})"
        )
        return self.best_model, self.cv_results

    def get_comparison_table(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising all trained models."""
        if not self.cv_results:
            return pd.DataFrame()

        rows = []
        for r in self.cv_results:
            rows.append(
                {
                    "model_name": r["model_name"],
                    f"cv_{self.scoring}_mean": r[f"cv_{self.scoring}_mean"],
                    f"cv_{self.scoring}_std": r[f"cv_{self.scoring}_std"],
                    "best_params": str(r["best_params"]),
                }
            )
        return pd.DataFrame(rows)
