import numpy as np

from src.features import FEATURES, build_features
from src.training.pipeline import make_split, score
from src.training.train import passes_gate, reference_distribution


def test_score_counts_confusion_matrix_cells():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.7])
    m = score(y_true, y_prob)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3


def test_split_is_stratified_and_disjoint():
    split = make_split("data/raw/ai4i2020.csv", 0.2, 42)
    assert len(split.y_test) == 2000
    assert abs(split.y_test.mean() - split.y_train.mean()) < 0.002
    assert set(split.raw_train.index).isdisjoint(split.raw_test.index)


def test_gate_needs_both_precision_and_recall():
    assert passes_gate({"precision": 0.9, "recall": 0.7}, 0.7, 0.6)
    assert not passes_gate({"precision": 0.9, "recall": 0.1}, 0.7, 0.6)
    assert not passes_gate({"precision": 0.5, "recall": 0.9}, 0.7, 0.6)


def test_reference_distribution_caps_samples():
    split = make_split("data/raw/ai4i2020.csv", 0.2, 42)
    ref = reference_distribution(build_features(split.raw_train), seed=1)
    assert set(ref) == set(FEATURES)
    assert len(ref["torque_nm"]["samples"]) == 2000
