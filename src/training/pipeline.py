"""Split, fit and score. Used by `make eval` and by the MLflow training run."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features import FEATURES, TARGET, build_features, load_ai4i

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 2,
}


@dataclass(frozen=True)
class Split:
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def make_split(path: str, test_size: float, seed: int) -> Split:
    """Stratified hold-out split on the raw frame (features are built per set)."""
    df = load_ai4i(path)
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    return Split(train, test, train[TARGET].to_numpy(), test[TARGET].to_numpy())


def make_model(seed: int, params: dict[str, Any] | None = None) -> XGBClassifier:
    return XGBClassifier(**{**DEFAULT_PARAMS, **(params or {})}, random_state=seed)


def score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Metrics for the failure class plus the confusion matrix [[tn, fp], [fn, tp]]."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_test": int(len(y_true)),
        "n_failures": int(y_true.sum()),
    }


def fit_and_score(split: Split, seed: int, params: dict[str, Any] | None = None):
    """Fit XGBoost on the training part and score the held-out part."""
    model = make_model(seed, params)
    model.fit(build_features(split.raw_train)[FEATURES], split.y_train)
    prob = model.predict_proba(build_features(split.raw_test)[FEATURES])[:, 1]
    return model, score(split.y_test, prob)
