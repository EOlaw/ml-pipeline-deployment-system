"""
Utility helpers shared across the pipeline.

Currently provides:
  * timer()         — context manager for timing code blocks.
  * save_json()     — write a dict to a JSON file.
  * load_json()     — read a JSON file into a dict.
  * sanitize_log()  — remove PII fields before logging.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

# Fields that must never appear in logs (redact / drop)
_PII_FIELDS: frozenset[str] = frozenset(
    {"name", "email", "phone", "ssn", "dob", "address", "ip_address"}
)


@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """
    Context manager that logs elapsed time for a code block.

    Example
    -------
        with timer("training"):
            model.fit(X, y)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"[timer] {label} completed in {elapsed:.2f}ms")


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Serialize ``data`` to a JSON file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"JSON saved ➜ {path}")


def load_json(path: Path) -> Dict[str, Any]:
    """Deserialize a JSON file into a dict."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_log(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of ``record`` with PII fields removed.

    Always call this before passing user-supplied data to the logger.
    """
    return {k: v for k, v in record.items() if k.lower() not in _PII_FIELDS}


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Example
    -------
        {"a": {"b": 1}} → {"a.b": 1}
    """
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
