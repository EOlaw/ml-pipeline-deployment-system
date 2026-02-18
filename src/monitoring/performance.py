"""
Performance Monitoring Module
==============================
Tracks and logs real-time inference statistics for the deployed model.

Metrics tracked
---------------
* Prediction latency (ms)          — detect degradation / resource issues.
* Model confidence distribution    — flag distribution shift early.
* Prediction class distribution    — detect output drift over time.
* Request counters                 — throughput visibility.

Drift detection placeholder
---------------------------
Real drift detection (e.g. KS-test, PSI) would compare the current
prediction / feature distribution against a reference baseline computed
during training.  The ``check_drift()`` method here logs a placeholder
alert that can be wired to PagerDuty, CloudWatch Alarms, or Slack.

Usage
-----
    monitor = PerformanceMonitor()
    monitor.record(prediction=1, probability=0.87, latency_ms=12.5)
    stats = monitor.get_stats()
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, Optional

from src.monitoring.logger import get_logger

logger = get_logger(__name__)

_WINDOW_SIZE = 1_000   # rolling window for in-memory stats


class PerformanceMonitor:
    """
    Thread-safe, in-memory rolling-window monitor for inference metrics.

    In production replace/augment with Prometheus counters, CloudWatch
    PutMetricData, or a time-series database (InfluxDB, Timestream).
    """

    _instance: Optional["PerformanceMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "PerformanceMonitor":
        """Singleton — one shared monitor across all Flask workers in-process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_state()
        return cls._instance

    def _init_state(self) -> None:
        self._latencies: Deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._probabilities: Deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._predictions: Deque[int] = deque(maxlen=_WINDOW_SIZE)
        self._total_requests: int = 0
        self._errors: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        prediction: int,
        probability: float,
        latency_ms: float,
    ) -> None:
        """
        Record metrics for a single completed inference request.

        Parameters
        ----------
        prediction  : 0 or 1 (model output class).
        probability : Probability of the positive class [0, 1].
        latency_ms  : End-to-end inference time in milliseconds.
        """
        with self._lock:
            self._latencies.append(latency_ms)
            self._probabilities.append(probability)
            self._predictions.append(prediction)
            self._total_requests += 1

        logger.info(
            f"Inference — prediction={prediction}  "
            f"probability={probability:.4f}  "
            f"latency={latency_ms:.2f}ms  "
            f"total_requests={self._total_requests}"
        )

        # Warn on high latency
        if latency_ms > 500:
            logger.warning(
                f"High inference latency detected: {latency_ms:.2f}ms  "
                "(threshold: 500ms)"
            )

        # Warn on very low/high confidence (potential OOD inputs)
        if probability < 0.05 or probability > 0.95:
            logger.warning(
                f"Extreme model confidence: probability={probability:.4f}  "
                "— consider checking for out-of-distribution input."
            )

    def record_error(self, error: str) -> None:
        """Increment error counter and log."""
        with self._lock:
            self._errors += 1
        logger.error(f"Inference error recorded: {error}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics over the rolling window.

        Returns
        -------
        dict with request counts, latency percentiles, and
        prediction distribution metrics.
        """
        with self._lock:
            lats = list(self._latencies)
            probs = list(self._probabilities)
            preds = list(self._predictions)

        n = len(lats)
        if n == 0:
            return {
                "total_requests": self._total_requests,
                "errors": self._errors,
                "window_size": 0,
                "message": "No requests recorded yet.",
            }

        import statistics

        pos_rate = sum(preds) / len(preds) if preds else 0.0
        avg_confidence = sum(probs) / len(probs) if probs else 0.0

        return {
            "total_requests": self._total_requests,
            "errors": self._errors,
            "window_size": n,
            "latency_ms": {
                "mean": round(statistics.mean(lats), 2),
                "median": round(statistics.median(lats), 2),
                "p95": round(self._percentile(lats, 95), 2),
                "p99": round(self._percentile(lats, 99), 2),
                "max": round(max(lats), 2),
            },
            "prediction_distribution": {
                "positive_rate": round(pos_rate, 4),
                "avg_confidence": round(avg_confidence, 4),
            },
        }

    def check_drift(self, reference_positive_rate: float = 0.30) -> Dict[str, Any]:
        """
        Placeholder drift detection.

        Compares current rolling positive-prediction rate against a
        reference baseline.  Returns an alert if the deviation exceeds
        a threshold.

        Parameters
        ----------
        reference_positive_rate : Positive rate observed during training.

        Returns
        -------
        {drift_detected: bool, delta: float, alert: str}
        """
        with self._lock:
            preds = list(self._predictions)

        if not preds:
            return {"drift_detected": False, "delta": 0.0, "alert": "No data"}

        current_rate = sum(preds) / len(preds)
        delta = abs(current_rate - reference_positive_rate)
        drift_detected = delta > 0.10   # 10 pp threshold

        result: Dict[str, Any] = {
            "drift_detected": drift_detected,
            "current_positive_rate": round(current_rate, 4),
            "reference_positive_rate": reference_positive_rate,
            "delta": round(delta, 4),
            "alert": (
                f"DRIFT ALERT: positive_rate shifted by {delta:.2%} "
                f"(current={current_rate:.2%}, reference={reference_positive_rate:.2%})"
                if drift_detected
                else "No significant drift detected."
            ),
        }

        if drift_detected:
            logger.warning(result["alert"])
        else:
            logger.info(result["alert"])

        return result

    def reset(self) -> None:
        """Reset all rolling-window counters (useful for testing)."""
        with self._lock:
            self._init_state()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _percentile(data: list, p: float) -> float:
        """Compute p-th percentile without numpy dependency."""
        sorted_data = sorted(data)
        idx = (len(sorted_data) - 1) * p / 100
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
        return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)
