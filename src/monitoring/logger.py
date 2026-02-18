"""
Centralized Logging
===================
Provides a single `get_logger(name)` factory used throughout the project.

Design decisions
----------------
* Rotating file handler — avoids unbounded disk growth.
* Structured format  — timestamp | level | module | message.
* No PII in messages — callers must sanitize inputs before logging.
* CloudWatch-ready   — format matches CloudWatch Logs Insights queries.

Usage
-----
    from src.monitoring.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import logging
import logging.handlers
import os
from pathlib import Path

# Resolve independently so logger can be imported before config is loaded
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _build_console_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(_LOG_LEVEL)
    handler.setFormatter(_build_formatter())
    return handler


def _build_file_handler(filename: str) -> logging.handlers.RotatingFileHandler:
    log_path = _LOGS_DIR / filename
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(_LOG_LEVEL)
    handler.setFormatter(_build_formatter())
    return handler


def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Return a named logger with console + rotating-file handlers.

    Parameters
    ----------
    name     : Typically ``__name__`` of the calling module.
    log_file : Filename inside logs/ (default: pipeline.log).
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(_LOG_LEVEL)
        logger.addHandler(_build_console_handler())
        logger.addHandler(_build_file_handler(log_file))
        logger.propagate = False  # prevent duplicate root-logger output

    return logger


def get_api_logger(name: str) -> logging.Logger:
    """Convenience wrapper that logs to api.log for Flask request logging."""
    return get_logger(name, log_file="api.log")
