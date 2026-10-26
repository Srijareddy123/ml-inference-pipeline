"""Data drift detection with the two-sample Kolmogorov-Smirnov test.

Incoming feature rows are buffered. Every `window_size` new rows, each feature in
that window is compared against the training sample saved with the model. A
feature counts as drifted when its p-value is below `threshold / n_features`
(Bonferroni), so testing nine features does not multiply the false alarm rate.
"""

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DriftResult:
    drift_detected: bool
    drifted_features: list[str]
    p_values: dict[str, float]
    threshold: float
    window_size: int


@dataclass
class DriftDetector:
    threshold: float = 0.05
    window_size: int = 100
    _reference: dict[str, np.ndarray] = field(default_factory=dict)
    _buffer: list[dict[str, float]] = field(default_factory=list)
    _last_result: DriftResult | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_reference(self, reference: dict[str, dict[str, Any]]) -> None:
        """Install the training distribution of the serving model and reset the window."""
        with self._lock:
            self._reference = {
                name: np.asarray(stats_["samples"], dtype=float)
                for name, stats_ in reference.items()
            }
            self._buffer = []
            self._last_result = None

    @property
    def has_reference(self) -> bool:
        return bool(self._reference)

    @property
    def last_result(self) -> DriftResult | None:
        return self._last_result

    def observe(self, features: dict[str, float]) -> DriftResult | None:
        """Add one row. Returns a result only when a full window has just been tested."""
        with self._lock:
            if not self._reference:
                return None
            self._buffer.append(features)
            if len(self._buffer) < self.window_size:
                return None
            window, self._buffer = self._buffer, []
            self._last_result = self._test_window(window)
            return self._last_result

    def _test_window(self, window: list[dict[str, float]]) -> DriftResult:
        per_feature = self.threshold / len(self._reference)
        p_values: dict[str, float] = {}
        drifted: list[str] = []
        for name, ref in self._reference.items():
            current = np.asarray([row[name] for row in window if name in row], dtype=float)
            if len(current) < 10:
                continue
            p_value = float(stats.ks_2samp(ref, current).pvalue)
            p_values[name] = p_value
            if p_value < per_feature:
                drifted.append(name)
        return DriftResult(bool(drifted), drifted, p_values, per_feature, len(window))
