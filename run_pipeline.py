"""
End-to-End ML Pipeline Runner
==============================
Orchestrates the complete training pipeline in one command:

  1. Data Ingestion    — generate / load data, train/test split.
  2. Data Validation   — schema and quality checks.
  3. Feature Eng.      — add domain-derived features.
  4. Preprocessing     — scale + encode, save fitted transformer.
  5. Model Training    — GridSearchCV on two classifiers.
  6. Model Evaluation  — metrics on held-out test set.
  7. Model Registry    — save best model + metadata.json.
  8. AWS Simulation    — log what an S3 upload would look like.

Usage
-----
    # With default settings (synthetic dataset)
    python run_pipeline.py

    # With a real CSV
    python run_pipeline.py --source csv --filepath data/raw/my_data.csv

    # Verbose output
    python run_pipeline.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.config import config
from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.validation import DataValidator
from src.features.feature_engineering import FeatureEngineer
from src.models.evaluate import ModelEvaluator
from src.models.registry import ModelRegistry
from src.models.train import ModelTrainer
from src.monitoring.logger import get_logger
from src.utils import timer

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end ML training pipeline."
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "csv", "sklearn"],
        default="synthetic",
        help="Data source (default: synthetic)",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="Path to CSV file (required when --source=csv)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def run_pipeline(source: str = "synthetic", filepath: str | None = None) -> None:
    """
    Execute the full ML training pipeline and save artifacts to disk.

    Parameters
    ----------
    source   : "synthetic" | "csv" | "sklearn"
    filepath : path to CSV (only used when source="csv")
    """
    separator = "=" * 60
    logger.info(separator)
    logger.info("  ML PIPELINE — START")
    logger.info(separator)

    # ── Step 1: Data Ingestion ────────────────────────────────────────────────
    with timer("data_ingestion"):
        logger.info("STEP 1 — Data Ingestion")
        ingestion = DataIngestion()
        X_train, X_test, y_train, y_test = ingestion.run(
            source=source, filepath=filepath
        )

    # ── Step 2: Data Validation ───────────────────────────────────────────────
    with timer("data_validation"):
        logger.info("STEP 2 — Data Validation")
        validator = DataValidator()

        # Validate combined dataset (add target back for validation)
        import pandas as pd
        train_df = X_train.copy()
        train_df["target"] = y_train.values
        report = validator.validate(train_df)

        if not report["is_valid"]:
            logger.error("Data validation FAILED. Aborting pipeline.")
            for err in report["errors"]:
                logger.error(f"  {err}")
            sys.exit(1)

        logger.info(
            f"Validation passed — "
            f"{len(report['warnings'])} warning(s), "
            f"0 error(s)"
        )

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    with timer("feature_engineering"):
        logger.info("STEP 3 — Feature Engineering")
        feature_engineer = FeatureEngineer()
        X_train_fe = feature_engineer.fit_transform(X_train)
        X_test_fe = feature_engineer.transform(X_test)
        logger.info(
            f"Features: {X_train.shape[1]} raw ➜ {X_train_fe.shape[1]} engineered"
        )

    # ── Step 4: Preprocessing ─────────────────────────────────────────────────
    with timer("preprocessing"):
        logger.info("STEP 4 — Data Preprocessing")
        preprocessor = DataPreprocessor()
        X_train_proc, X_test_proc = preprocessor.fit_transform(X_train_fe, X_test_fe)
        preprocessor.save()
        logger.info(f"Final feature matrix: {X_train_proc.shape}")

    # ── Step 5: Model Training ────────────────────────────────────────────────
    with timer("model_training"):
        logger.info("STEP 5 — Model Training (GridSearchCV + Cross-Validation)")
        trainer = ModelTrainer()
        best_model, cv_results = trainer.train(X_train_proc, y_train)

        logger.info("\n  Cross-Validation Comparison:")
        comparison = trainer.get_comparison_table()
        logger.info(f"\n{comparison.to_string(index=False)}")

    # ── Step 6: Model Evaluation ──────────────────────────────────────────────
    with timer("model_evaluation"):
        logger.info("STEP 6 — Model Evaluation on Test Set")
        evaluator = ModelEvaluator()
        all_reports = evaluator.evaluate_all(cv_results, X_test_proc, y_test)

        for report in all_reports:
            evaluator.print_report(report)

        best_report = all_reports[0]   # sorted by F1 desc
        logger.info(
            f"Best test-set model: {best_report['model_name']}  "
            f"F1={best_report['f1']:.4f}  "
            f"ROC-AUC={best_report.get('roc_auc', 'N/A')}"
        )

    # ── Step 7: Model Registry ────────────────────────────────────────────────
    with timer("model_registry"):
        logger.info("STEP 7 — Saving Best Model to Registry")
        registry = ModelRegistry()
        model_path = registry.save(
            model=best_model,
            model_name=trainer.best_model_name,
            metrics=best_report,
            cv_score=cv_results[0][f"cv_{trainer.scoring}_mean"],
            best_params=cv_results[0]["best_params"],
            training_samples=len(X_train),
            feature_count=X_train_proc.shape[1],
        )
        logger.info(f"Model artifact ➜ {model_path}")

    # ── Step 8: AWS Simulation ────────────────────────────────────────────────
    logger.info("STEP 8 — AWS Deployment Simulation")
    registry.simulate_s3_upload()
    logger.info(
        "[AWS SIMULATION] Would deploy Flask API on EC2 t3.medium "
        f"behind an ALB at port {config.api.port}"
    )
    logger.info(
        "[AWS SIMULATION] Would ship logs to CloudWatch group: "
        f"{config.aws.cloudwatch_log_group}"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(separator)
    logger.info("  ML PIPELINE — COMPLETE")
    logger.info(separator)
    logger.info(f"  Best Model     : {trainer.best_model_name}")
    logger.info(f"  Accuracy       : {best_report['accuracy']:.4f}")
    logger.info(f"  Precision      : {best_report['precision']:.4f}")
    logger.info(f"  Recall         : {best_report['recall']:.4f}")
    logger.info(f"  F1-score       : {best_report['f1']:.4f}")
    logger.info(f"  ROC-AUC        : {best_report.get('roc_auc', 'N/A')}")
    logger.info(f"  Model saved at : {model_path}")
    logger.info(separator)
    logger.info("  Next step: python -m src.api.app   (or: docker compose up)")
    logger.info(separator)


if __name__ == "__main__":
    args = parse_args()

    # Apply log level from CLI
    import logging
    logging.getLogger().setLevel(args.log_level)

    run_pipeline(source=args.source, filepath=args.filepath)
