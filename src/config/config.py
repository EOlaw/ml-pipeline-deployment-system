"""
Configuration Management
========================
Centralizes every configurable value in one place.
Environment variables always override defaults — enabling
zero-code changes between local, Docker and AWS deployments.

Usage
-----
    from src.config.config import config

    print(config.model.cv_folds)   # 5
    print(config.api.port)         # 5000
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env file when present

# ─── Resolved project root (works from any working directory) ────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
LOGS_DIR: Path = BASE_DIR / "logs"


# ─── Sub-config dataclasses ──────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Paths and split settings for the data layer."""
    raw_data_path: Path = field(default_factory=lambda: DATA_DIR / "raw")
    processed_data_path: Path = field(default_factory=lambda: DATA_DIR / "processed")
    external_data_path: Path = field(default_factory=lambda: DATA_DIR / "external")

    raw_filename: str = "raw_data.csv"
    train_filename: str = "train.csv"
    test_filename: str = "test.csv"
    target_column: str = "target"
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))


@dataclass
class ModelConfig:
    """Model training hyper-parameters and persistence paths."""
    models_path: Path = field(default_factory=lambda: MODELS_DIR)
    model_filename: str = "model_v1.pkl"
    preprocessor_filename: str = "preprocessor_v1.pkl"
    metadata_filename: str = "metadata.json"

    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))
    n_jobs: int = int(os.getenv("N_JOBS", "-1"))
    scoring_metric: str = os.getenv("SCORING_METRIC", "f1")

    # Numeric columns (post-ingestion, pre-encoding)
    numeric_features: tuple = (
        "income",
        "age",
        "loan_amount",
        "credit_score",
        "employment_years",
        "debt_ratio",
        "num_accounts",
    )
    categorical_features: tuple = ("employment_type",)


@dataclass
class APIConfig:
    """Flask / Gunicorn server settings."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    model_version: str = os.getenv("MODEL_VERSION", "v1")


@dataclass
class AWSConfig:
    """
    AWS integration settings (simulation).
    In production these come from IAM instance roles or AWS Secrets Manager.
    """
    s3_bucket: str = os.getenv("S3_BUCKET", "ml-pipeline-models")
    s3_model_key: str = os.getenv("S3_MODEL_KEY", "models/model_v1.pkl")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")

    rds_host: str = os.getenv("RDS_HOST", "localhost")
    rds_port: int = int(os.getenv("RDS_PORT", "5432"))
    rds_database: str = os.getenv("RDS_DATABASE", "mldb")

    cloudwatch_log_group: str = os.getenv(
        "CLOUDWATCH_LOG_GROUP", "/ml-pipeline/predictions"
    )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_path: Path = field(default_factory=lambda: LOGS_DIR)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_filename: str = "pipeline.log"
    api_log_filename: str = "api.log"
    max_bytes: int = 10 * 1024 * 1024   # 10 MB per file
    backup_count: int = 5               # keep last 5 rotated files


# ─── Root config container ───────────────────────────────────────────────────

@dataclass
class Config:
    """Aggregate root — import this everywhere."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def create_directories(self) -> None:
        """Create all required runtime directories if they don't exist."""
        dirs = [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.data.external_data_path,
            self.model.models_path,
            self.logging.log_path,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


# ─── Singleton ───────────────────────────────────────────────────────────────
config = Config()
config.create_directories()
