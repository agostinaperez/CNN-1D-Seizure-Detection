import numpy as np
import pytest

from src.metrics import binary_metrics, detect_event_intervals, rank_metrics


def test_detect_event_intervals_new_event_after_interval():
    starts = np.array([0.0, 5.0, 10.0])
    ends = np.array([5.0, 10.0, 15.0])
    preds = np.array([True, False, True])
    events = detect_event_intervals(starts, ends, preds, min_alarm_interval=3.0)
    assert events == [(0.0, 5.0), (10.0, 15.0)]


def test_detect_event_intervals_merges_close():
    starts = np.array([0.0, 5.0])
    ends = np.array([5.0, 10.0])
    preds = np.array([True, True])
    events = detect_event_intervals(starts, ends, preds, min_alarm_interval=30.0)
    assert events == [(0.0, 10.0)]


def test_binary_metrics_perfect():
    probs = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    m = binary_metrics(probs, labels, threshold=0.5)
    assert m["sensibility"] == 1.0
    assert m["specificity"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["false_positive"] == 0


def test_rank_metrics_perfect_separation():
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    rk = rank_metrics(probs, labels)
    assert rk["auc_roc"] == pytest.approx(1.0)
    assert rk["auprc"] == pytest.approx(1.0)
