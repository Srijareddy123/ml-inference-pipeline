"""Train XGBoost on AI4I 2020, track it in MLflow, register it, promote it if it passes the gate."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

from src.config import get_settings
from src.features import FEATURES, build_features
from src.registry import Registry
from src.training.pipeline import DEFAULT_PARAMS, fit_and_score, make_split

logger = logging.getLogger(__name__)

EXPERIMENT = "machine-failure-xgboost"
REFERENCE_FILE = "reference_distribution.json"
REFERENCE_SAMPLES = 2000


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    version: str
    metrics: dict[str, Any]
    promoted: bool
    reason: str


def reference_distribution(features: pd.DataFrame, seed: int) -> dict[str, dict[str, Any]]:
    """Per-feature mean, std and a sample of values for the KS test at serving time."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, Any]] = {}
    for col in FEATURES:
        values = features[col].to_numpy(dtype=float)
        if len(values) > REFERENCE_SAMPLES:
            values = rng.choice(values, REFERENCE_SAMPLES, replace=False)
        out[col] = {
            "mean": float(features[col].mean()),
            "std": float(features[col].std()),
            "samples": [float(v) for v in values],
        }
    return out


def passes_gate(metrics: dict[str, Any], min_precision: float, min_recall: float) -> bool:
    return metrics["precision"] >= min_precision and metrics["recall"] >= min_recall


def _ensure_experiment() -> None:
    settings = get_settings()
    if mlflow.get_experiment_by_name(EXPERIMENT) is not None:
        mlflow.set_experiment(EXPERIMENT)
        return
    location = None
    if not settings.mlflow_tracking_uri.startswith("http"):
        location = Path(settings.mlflow_artifact_root).resolve().as_uri()
    mlflow.create_experiment(EXPERIMENT, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT)


def train(
    data_path: str | None = None,
    seed: int | None = None,
    params: dict[str, Any] | None = None,
    promote: bool = True,
) -> TrainResult:
    settings = get_settings()
    seed = settings.random_seed if seed is None else seed
    data_path = data_path or settings.data_path
    registry = Registry(settings.mlflow_tracking_uri)
    _ensure_experiment()

    split = make_split(data_path, settings.test_size, seed)
    model, metrics = fit_and_score(split, seed, params)

    with mlflow.start_run() as run:
        mlflow.log_params({**DEFAULT_PARAMS, **(params or {}), "seed": seed})
        mlflow.log_params(
            {"dataset": "AI4I 2020", "n_train": len(split.y_train), "n_test": len(split.y_test)}
        )
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_metrics({k: metrics[k] for k in ("tn", "fp", "fn", "tp")})
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, REFERENCE_FILE)
            with open(ref_path, "w") as fh:
                json.dump(reference_distribution(build_features(split.raw_train), seed), fh)
            mlflow.log_artifact(ref_path, artifact_path="monitoring")
            cm_path = os.path.join(tmp, "confusion_matrix.json")
            with open(cm_path, "w") as fh:
                json.dump({"labels": ["normal", "failure"], **metrics}, fh, indent=2)
            mlflow.log_artifact(cm_path, artifact_path="evaluation")
        info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=settings.model_name,
            input_example=build_features(split.raw_test)[FEATURES].head(2),
        )
        version = str(info.registered_model_version)
        run_id = run.info.run_id

    promoted, reason = False, "promotion disabled"
    if promote:
        promoted, reason = _maybe_promote(registry, settings.model_name, version, metrics)
    logger.info("version %s: %s", version, reason)
    return TrainResult(run_id, version, metrics, promoted, reason)


def _maybe_promote(
    registry: Registry, name: str, version: str, metrics: dict[str, Any]
) -> tuple[bool, str]:
    settings = get_settings()
    if not passes_gate(metrics, settings.min_precision, settings.min_recall):
        return False, (
            f"not promoted: precision {metrics['precision']:.3f} / recall "
            f"{metrics['recall']:.3f} is below the gate "
            f"({settings.min_precision} / {settings.min_recall})"
        )
    champion = registry.champion(name)
    if champion is not None:
        champion_f1 = registry.run_metric(champion.run_id, "f1")
        if champion_f1 is not None and metrics["f1"] < champion_f1:
            return False, (
                f"not promoted: f1 {metrics['f1']:.3f} is below champion v{champion.version} "
                f"({champion_f1:.3f})"
            )
    registry.promote(name, version)
    return True, "promoted to champion"
