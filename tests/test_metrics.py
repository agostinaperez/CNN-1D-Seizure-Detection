from pathlib import Path

import numpy as np
import pytest

import src.metrics as metrics_module
from src.config import (
    FS,
    POSITIVES_FOR_EVENT,
    STRIDE_SAMPLES,
    WIN_SECONDS_EFFECTIVE,
    WINDOW_RANGE_FOR_EVENT,
)
from src.metrics import (
    _span_hours,
    _span_hours_by_file,
    binary_metrics,
    detect_alarm_indices,
    detect_event_intervals,
    event_metrics,
    rank_metrics,
    select_event_operating_point,
    select_lowest_false_alarm_point,
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


# ---------------------------------------------------------------- horas cubiertas
def test_span_hours():
    span_s_1 = WIN_SECONDS_EFFECTIVE
    span_s_3 = (3 - 1) * (STRIDE_SAMPLES / FS) + WIN_SECONDS_EFFECTIVE
    assert _span_hours(0) == 0.0
    assert _span_hours(1) == pytest.approx(span_s_1 / 3600.0)
    assert _span_hours(3) == pytest.approx(span_s_3 / 3600.0)


def test_span_hours_by_file_sums_per_file():
    # archivo 0: 3 ventanas, archivo 1: 2 ventanas
    file_ids = np.array([0, 0, 0, 1, 1])
    local_ids = np.array([0, 1, 2, 0, 1])
    expected = _span_hours(3) + _span_hours(2)
    assert _span_hours_by_file(file_ids, local_ids) == pytest.approx(expected)


def test_binary_metrics_false_positives_per_hour():
    probs = np.array([0.9, 0.1])
    labels = np.array([0, 0])
    file_ids = np.array([0, 0])
    local_ids = np.array([0, 1])
    m = binary_metrics(probs, labels, 0.5, file_ids=file_ids, local_ids=local_ids)
    span_s = (2 - 1) * (STRIDE_SAMPLES / FS) + WIN_SECONDS_EFFECTIVE
    expected_fp_per_hour = 1.0 / (span_s / 3600.0)
    assert m["false_positive"] == 1
    assert m["false_positive_per_hour"] == pytest.approx(expected_fp_per_hour)


# ---------------------------------------------------------------- punto de operación
def test_select_event_operating_point_respects_ceiling(monkeypatch):
    results = {
        0.3: {"sensibility": 0.9, "median_false_alarms_per_hour": 2.0},  # excede techo
        0.5: {"sensibility": 0.6, "median_false_alarms_per_hour": 0.8},
        0.7: {"sensibility": 0.8, "median_false_alarms_per_hour": 0.9},
        0.9: {"sensibility": 0.4, "median_false_alarms_per_hour": 0.1},
    }

    def fake(probs, labels, **kw):
        return results[kw["threshold"]]

    monkeypatch.setattr(metrics_module, "event_metrics", fake)
    best, thr = select_event_operating_point(
        [0.5], [0], file_ids=[0], local_ids=[0], valid_files=[], annotations={},
        max_false_alarms_per_hour=1.0, thresholds=[0.3, 0.5, 0.7, 0.9],
    )
    # entre los que respetan el techo (0.5, 0.7, 0.9) el de mayor sensibilidad es 0.7 (0.8)
    assert thr == 0.7
    assert best["sensibility"] == 0.8


def test_select_event_operating_point_tie_break_lower_fa(monkeypatch):
    results = {
        0.5: {"sensibility": 0.7, "median_false_alarms_per_hour": 0.9},
        0.7: {"sensibility": 0.7, "median_false_alarms_per_hour": 0.2},
    }

    def fake(probs, labels, **kw):
        return results[kw["threshold"]]

    monkeypatch.setattr(metrics_module, "event_metrics", fake)
    best, thr = select_event_operating_point(
        [0.5], [0], file_ids=[0], local_ids=[0], valid_files=[], annotations={},
        max_false_alarms_per_hour=1.0, thresholds=[0.5, 0.7],
    )
    assert thr == 0.7  # misma sensibilidad, menor FA mediana


def test_select_event_operating_point_none_when_none_respect_ceiling(monkeypatch):
    def fake(probs, labels, **kw):
        return {"sensibility": 0.5, "median_false_alarms_per_hour": 2.0}

    monkeypatch.setattr(metrics_module, "event_metrics", fake)
    best, thr = select_event_operating_point(
        [0.5], [0], file_ids=[0], local_ids=[0], valid_files=[], annotations={},
        max_false_alarms_per_hour=1.0, thresholds=[0.3, 0.5],
    )
    assert best is None and thr is None


def test_select_lowest_false_alarm_point_tie_break_sensibility(monkeypatch):
    results = {
        0.5: {"sensibility": 0.6, "median_false_alarms_per_hour": 0.9},
        0.7: {"sensibility": 0.3, "median_false_alarms_per_hour": 0.2},
        0.9: {"sensibility": 0.4, "median_false_alarms_per_hour": 0.2},  # empate FA, mayor sens
    }

    def fake(probs, labels, **kw):
        return results[kw["threshold"]]

    monkeypatch.setattr(metrics_module, "event_metrics", fake)
    best, thr = select_lowest_false_alarm_point(
        [0.5], [0], file_ids=[0], local_ids=[0], valid_files=[], annotations={},
        thresholds=[0.5, 0.7, 0.9],
    )
    assert thr == 0.9  # menor FA mediana (0.2), desempate por mayor sensibilidad
