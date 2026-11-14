from pathlib import Path

import numpy as np
import pytest

from src.config import POSITIVES_FOR_EVENT, WINDOW_RANGE_FOR_EVENT
from src.metrics import (
    binary_metrics,
    detect_alarm_indices,
    detect_event_intervals,
    event_metrics,
    rank_metrics,
)
from src.timing import decision_times_from_local_ids


def test_detect_event_intervals_uses_two_of_three():
    starts = np.array([0.0, 5.12, 10.24])
    ends = np.array([5.12, 10.24, 15.36])
    preds = np.array([True, False, True])
    events = detect_event_intervals(starts, ends, preds, n_within=2, n_window=3,
                                    min_alarm_interval=240.0)
    assert events == [(0.0, 15.36)]


def test_detect_event_intervals_respects_min_alarm_interval():
    starts = np.array([0.0, 5.12, 10.24, 15.36, 20.48, 25.6])
    ends = np.array([5.12, 10.24, 15.36, 20.48, 25.6, 30.72])
    preds = np.array([True, False, True, False, True, True])

    events = detect_event_intervals(starts, ends, preds, n_within=2, n_window=3,
                                    min_alarm_interval=6.0)

    assert events == [(0.0, 15.36), (10.24, 25.6)]


def test_detect_event_intervals_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="igual longitud"):
        detect_event_intervals([0.0], [5.0, 10.0], [True])


def test_detect_alarm_indices_fires_two_of_three_and_respects_interval():
    times = np.array([5.12, 10.24, 15.36, 20.48, 25.6, 30.72])
    preds = np.array([True, False, True, False, True, True])

    indices = detect_alarm_indices(preds, times, n_within=2, n_window=3,
                                   min_alarm_interval=6.0)

    assert indices == [2, 4]


def test_detect_alarm_indices_rejects_invalid_config():
    times = np.array([5.12, 10.24, 15.36])
    preds = np.array([True, False, True])
    with pytest.raises(ValueError, match="ventanas de evento"):
        detect_alarm_indices(preds, times, n_within=4, n_window=3)


def test_event_metrics_uses_configured_two_of_three_and_window_end_latency(tmp_path):
    valid_file = tmp_path / "chb01" / "record.edf"
    local_ids = np.array([0, 1, 2])
    decision_time = float(decision_times_from_local_ids([2])[0])
    seizure_onset = decision_time - 2.5

    assert (POSITIVES_FOR_EVENT, WINDOW_RANGE_FOR_EVENT) == (2, 3)
    metrics = event_metrics(
        probs=np.array([0.9, 0.1, 0.9]),
        labels=np.array([1, 0, 1]),
        file_ids=np.zeros(3, dtype=int),
        local_ids=local_ids,
        valid_files=[valid_file],
        annotations={valid_file.name: [(seizure_onset, seizure_onset + 10.0)]},
        threshold=0.5,
    )

    assert metrics["n_alarms"] == 1
    assert metrics["n_detected"] == 1
    assert metrics["n_false_alarms"] == 0
    assert metrics["latency_mean"] == pytest.approx(2.5)


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
