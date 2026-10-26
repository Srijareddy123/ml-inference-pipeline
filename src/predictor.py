"""Loads the champion model from the registry and turns requests into predictions."""

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import mlflow.xgboost
import pandas as pd

from src.config import get_settings
from src.features import FEATURES, LABELS, build_features
from src.registry import ModelInfo, Registry
from src.training.train import REFERENCE_FILE


@dataclass(frozen=True)
class Prediction:
    label: str
    failure_probability: float
    model_name: str
    model_version: str
    latency_ms: float
    features: dict[str, float]


class Predictor:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self._model: Any = None
        self._info: ModelInfo | None = None
        self._lock = threading.Lock()

    @property
    def info(self) -> ModelInfo | None:
        return self._info

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> dict[str, dict[str, Any]]:
        """Load the champion and return its drift reference distribution."""
        name = get_settings().model_name
        info = self.registry.champion(name)
        if info is None:
            raise RuntimeError(f"No champion version of {name}. Train a model first.")
        model = mlflow.xgboost.load_model(info.uri)
        with tempfile.TemporaryDirectory() as tmp:
            path = self.registry.client.download_artifacts(
                info.run_id, f"monitoring/{REFERENCE_FILE}", tmp
            )
            with open(path) as fh:
                reference = json.load(fh)
        with self._lock:
            self._model, self._info = model, info
        return reference

    def predict(self, raw: dict[str, Any]) -> Prediction:
        with self._lock:
            model, info = self._model, self._info
        if model is None or info is None:
            raise RuntimeError("model is not loaded")
        start = time.perf_counter()
        features = build_features(pd.DataFrame([raw]))[FEATURES]
        prob = float(model.predict_proba(features)[0, 1])
        latency_ms = (time.perf_counter() - start) * 1000
        return Prediction(
            label=LABELS[int(prob >= 0.5)],
            failure_probability=prob,
            model_name=info.name,
            model_version=info.version,
            latency_ms=latency_ms,
            features={k: float(v) for k, v in features.iloc[0].items()},
        )
