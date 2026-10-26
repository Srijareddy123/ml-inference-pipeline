import numpy as np

from src.drift import DriftDetector


def reference(rng):
    return {
        "a": {"samples": rng.normal(0, 1, 2000).tolist()},
        "b": {"samples": rng.normal(10, 2, 2000).tolist()},
    }


def feed(detector, rng, n, shift_a=0.0):
    last = None
    for _ in range(n):
        out = detector.observe({"a": rng.normal(shift_a, 1), "b": rng.normal(10, 2)})
        last = out or last
    return last


def test_no_result_until_window_is_full():
    rng = np.random.default_rng(0)
    d = DriftDetector(window_size=100)
    d.set_reference(reference(rng))
    assert feed(d, rng, 99) is None


def test_same_distribution_is_not_flagged():
    rng = np.random.default_rng(1)
    d = DriftDetector(window_size=200)
    d.set_reference(reference(rng))
    result = feed(d, rng, 200)
    assert result is not None and not result.drift_detected


def test_shifted_feature_is_flagged_and_others_are_not():
    rng = np.random.default_rng(2)
    d = DriftDetector(window_size=100)
    d.set_reference(reference(rng))
    result = feed(d, rng, 100, shift_a=1.0)
    assert result.drift_detected
    assert result.drifted_features == ["a"]
    assert result.p_values["a"] < result.threshold


def test_threshold_is_bonferroni_corrected():
    rng = np.random.default_rng(3)
    d = DriftDetector(threshold=0.05, window_size=100)
    d.set_reference(reference(rng))
    assert feed(d, rng, 100).threshold == 0.025


def test_windows_do_not_overlap():
    rng = np.random.default_rng(4)
    d = DriftDetector(window_size=50)
    d.set_reference(reference(rng))
    assert feed(d, rng, 50) is not None
    assert feed(d, rng, 49) is None


def test_without_reference_nothing_happens():
    d = DriftDetector(window_size=10)
    assert not d.has_reference
    assert d.observe({"a": 1.0}) is None


def test_new_reference_resets_the_window():
    rng = np.random.default_rng(5)
    d = DriftDetector(window_size=50)
    d.set_reference(reference(rng))
    feed(d, rng, 30)
    d.set_reference(reference(rng))
    assert feed(d, rng, 30) is None
