from src.config import get_settings
from src.registry import Registry
from src.training.train import train


def test_first_version_is_promoted_and_has_held_out_metrics(trained):
    first, _ = trained
    assert first.promoted
    for key in ("precision", "recall", "f1", "tn", "fp", "fn", "tp"):
        assert key in first.metrics
    assert first.metrics["n_test"] == 2000
    # The classifier must clearly beat "always predict normal" (precision and recall of 0).
    assert first.metrics["recall"] > 0.5


def test_unpromoted_version_leaves_champion_alone(trained):
    first, second = trained
    registry = Registry(get_settings().mlflow_tracking_uri)
    assert not second.promoted
    assert registry.champion(get_settings().model_name).version == first.version


def test_weak_model_is_rejected_by_the_gate(trained):
    result = train(params={"max_depth": 1, "n_estimators": 3})
    assert not result.promoted
    assert "below the gate" in result.reason


def test_model_worse_than_champion_is_not_promoted(trained, monkeypatch):
    monkeypatch.setattr(Registry, "run_metric", lambda self, run_id, metric: 0.999)
    result = train(params={"max_depth": 5})
    assert not result.promoted
    assert "below champion" in result.reason


def test_run_logs_confusion_matrix_artifact(trained):
    first, _ = trained
    registry = Registry(get_settings().mlflow_tracking_uri)
    artifacts = [a.path for a in registry.client.list_artifacts(first.run_id, "evaluation")]
    assert "evaluation/confusion_matrix.json" in artifacts
