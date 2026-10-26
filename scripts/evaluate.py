"""Held-out evaluation of the failure classifier. Run with `make eval`.

Writes results/metrics.json and results/metrics.md. Every number in the README
comes from this file. No MLflow or database is needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.features import BASE_FEATURES, FEATURES, build_features  # noqa: E402
from src.training.pipeline import make_model, make_split, score  # noqa: E402

OUT = Path("results")


def cv_scores(split, columns: list[str], seed: int, scale_pos_weight: bool) -> dict:
    """5-fold cross-validation on the training part only (the test part stays untouched)."""
    x = build_features(split.raw_train)[columns]
    params = {}
    if scale_pos_weight:
        params["scale_pos_weight"] = float((split.y_train == 0).sum() / split.y_train.sum())
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    prob = cross_val_predict(
        make_model(seed, params), x, split.y_train, cv=folds, method="predict_proba"
    )
    return score(split.y_train, prob[:, 1])


def main() -> None:
    settings = get_settings()
    seed = settings.random_seed
    split = make_split(settings.data_path, settings.test_size, seed)
    x_train = build_features(split.raw_train)[FEATURES]
    x_test = build_features(split.raw_test)[FEATURES]

    # Held-out test set: baselines and the final model.
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=2).fit(
        x_train, split.y_train
    )
    base_cols = BASE_FEATURES
    xgb_raw = make_model(seed).fit(x_train[base_cols], split.y_train)
    xgb = make_model(seed).fit(x_train, split.y_train)
    test = {
        "always_normal": score(split.y_test, np.zeros(len(split.y_test))),
        "random_forest": score(split.y_test, rf.predict_proba(x_test)[:, 1]),
        "xgboost_raw_features": score(split.y_test, xgb_raw.predict_proba(x_test[base_cols])[:, 1]),
        "xgboost_final": score(split.y_test, xgb.predict_proba(x_test)[:, 1]),
    }

    # Design choices, judged on cross-validation of the training part.
    cv = {
        "raw features": cv_scores(split, base_cols, seed, False),
        "raw features + class weight": cv_scores(split, base_cols, seed, True),
        "engineered features": cv_scores(split, FEATURES, seed, False),
        "engineered features + class weight": cv_scores(split, FEATURES, seed, True),
    }

    result = {
        "dataset": "AI4I 2020 (UCI), 10000 rows",
        "split": {
            "train": len(split.y_train),
            "test": len(split.y_test),
            "test_failures": int(split.y_test.sum()),
            "seed": seed,
        },
        "test": test,
        "cross_validation": cv,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    (OUT / "metrics.md").write_text(render(result))
    print(render(result))


def render(result: dict) -> str:
    def row(name: str, m: dict) -> str:
        cells = " | ".join(f"{m[k]:.3f}" for k in ("precision", "recall", "f1", "pr_auc"))
        return f"| {name} | {cells} |"

    head = "| model | precision | recall | F1 | PR-AUC |\n|---|---|---|---|---|"
    final = result["test"]["xgboost_final"]
    s = result["split"]
    lines = [
        f"Held-out test set: {s['test']} rows, {s['test_failures']} failures (seed {s['seed']}).",
        "Precision, recall and F1 are for the failure class at threshold 0.5.",
        "",
        head,
        *(row(k, v) for k, v in result["test"].items()),
        "",
        "Confusion matrix, final XGBoost model (rows are actual, columns are predicted):",
        "",
        "| | predicted normal | predicted failure |",
        "|---|---|---|",
        f"| actual normal | {final['tn']} | {final['fp']} |",
        f"| actual failure | {final['fn']} | {final['tp']} |",
        "",
        f"5-fold cross-validation on the {s['train']} training rows:",
        "",
        head,
        *(row(k, v) for k, v in result["cross_validation"].items()),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
