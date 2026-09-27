"""
Métricas de evaluación para detección de crisis.

Sigue la convención de los papers del marco teórico:

  - Nivel SEGMENTO (ventana): sensibilidad, especificidad, accuracy, AUC y fp/hora.
  - Nivel EVENTO (crisis): sensibilidad + FDR (falsas detecciones / hora) + latencia,
    con postprocesado "n positivos en N ventanas consecutivas" + intervalo mínimo
    entre alarmas (IEEE TNSRE 2025, "EEG-Based Seizure Onset Detection ... 1DCNN").

También expone:
  - `threshold_sweep`: la curva sensibilidad <-> fp/h barriendo umbrales.
  - `select_operating_point`: el umbral con MENOR fp/h que cumpla sensibilidad >= objetivo
    (lo que se usa para elegir el mejor checkpoint en train.py).
"""

from __future__ import annotations

import numpy as np

from src.config import FS, STRIDE_SAMPLES, WIN_SECONDS


def _span_hours(n_windows: int) -> float:
    """Horas cubiertas por `n_windows` ventanas solapadas al 50%."""
    if n_windows <= 0:
        return 0.0
    span_seconds = (n_windows - 1) * (STRIDE_SAMPLES / FS) + WIN_SECONDS
    return span_seconds / 3600.0


def _counts(probs, labels, threshold: float) -> tuple[int, int, int, int]:
    """Matriz de confusión (TP, FP, TN, FN) para un umbral dado."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    pred = probs >= threshold
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return tp, fp, tn, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def binary_metrics(probs, labels, threshold: float) -> dict:
    """Métricas por VENTANA (segment-based) sobre la matriz de confusión.
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    tp, fp, tn, fn = _counts(probs, labels, threshold)
    pos = tp + fn
    neg = fp + tn
    total = pos + neg
    total_hours = _span_hours(int(labels.shape[0]))
    return {
        "sensibility": tp / pos if pos > 0 else 0.0,
        "specificity": tn / neg if neg > 0 else 0.0,
        "false_positive_rate": fp / neg if neg > 0 else 0.0,
        "false_positive_per_hour": fp / total_hours if total_hours > 0 else 0.0,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "f1": _f1(tp, fp, fn),
        "mcc": _mcc(tp, fp, tn, fn),
        "actual_positive": tp,
        "false_positive": fp,
        "actual_negative": tn,
        "false_negative": fn,
    }


def threshold_sweep(probs, labels, thresholds=None) -> list[dict]:
    """Curva sensibilidad <-> fp/h barriendo umbrales. Una fila (dict) por umbral."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    total_hours = _span_hours(int(labels.shape[0]))
    rows: list[dict] = []
    for t in thresholds:
        tp, fp, tn, fn = _counts(probs, labels, t)
        pos = tp + fn
        neg = fp + tn
        total = pos + neg
        rows.append({
            "threshold": float(t),
            "sensibility": tp / pos if pos > 0 else 0.0,
            "specificity": tn / neg if neg > 0 else 0.0,
            "false_positive_rate": fp / neg if neg > 0 else 0.0,
            "false_positive_per_hour": fp / total_hours if total_hours > 0 else 0.0,
            "accuracy": (tp + tn) / total if total > 0 else 0.0,
            "f1": _f1(tp, fp, fn),
            "mcc": _mcc(tp, fp, tn, fn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return rows


def select_operating_point(curve: list[dict], min_sensibility: float) -> dict | None:
    """Entre los umbrales con sensibilidad >= objetivo, el de MENOR fp/h.

    Devuelve None si ningún umbral alcanza el objetivo de sensibilidad.
    """
    feasible = [r for r in curve if r["sensibility"] >= min_sensibility]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["false_positive_per_hour"])


def event_metrics(probs, labels, *, file_ids, local_ids, valid_files, annotations,
                  threshold: float, n_within: int = 3, n_window: int = 4,
                  min_alarm_interval: float = 30.0, max_latency: float = 30.0) -> dict:
    """
    Evaluación a nivel EVENTO (crisis), como en el paper 1DCNN (IEEE TNSRE 2025).

    - Postprocesado: disparo una alarma cuando hay >= `n_within` predicciones
      positivas dentro de las últimas `n_window` ventanas CONSECUTIVAS (por archivo,
      porque el tiempo se reinicia en cada EDF).
    - Supresión: entre dos alarmas debe haber >= `min_alarm_interval` segundos.
    - Una crisis anotada se considera DETECTADA si hay una alarma dentro de
      [onset, onset + max_latency].
    - Latencia = tiempo desde el onset de la crisis hasta su primera alarma.

    Devuelve sensibilidad (evento), FDR (falsas alarmas / hora), latencia media y
    mediana, y conteos crudos.
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    file_ids = np.asarray(file_ids)
    local_ids = np.asarray(local_ids)

    pred = probs >= threshold
    total_hours = _span_hours(int(labels.shape[0]))

    # 1) Alarmas por archivo (postprocesado sobre las predicciones en orden temporal).
    alarms_by_file: dict[int, list[float]] = {}
    for fid in np.unique(file_ids):
        fid = int(fid)
        idx = np.flatnonzero(file_ids == fid)
        order = np.argsort(local_ids[idx])
        idx = idx[order]
        p = pred[idx]
        win_times = local_ids[idx].astype(np.float64) * STRIDE_SAMPLES / FS

        alarm_times: list[float] = []
        last_alarm = -np.inf
        recent: list[bool] = []
        for j, t in enumerate(win_times):
            recent.append(bool(p[j]))
            if len(recent) > n_window:
                recent.pop(0)
            if len(recent) >= n_window and sum(recent) >= n_within:
                if t - last_alarm >= min_alarm_interval:
                    alarm_times.append(t)
                    last_alarm = t
        alarms_by_file[fid] = alarm_times

    # 2) Matcheo de alarmas contra crisis anotadas.
    n_seizures = 0
    n_detected = 0
    n_false_alarms = 0
    n_alarms = 0
    latencies: list[float] = []

    for fid, path in enumerate(valid_files):
        seizures = annotations.get(path.name, [])
        file_alarms = sorted(alarms_by_file.get(fid, []))
        n_alarms += len(file_alarms)
        matched = [False] * len(file_alarms)
        for onset, _end in seizures:
            n_seizures += 1
            det_time = None
            for k, t in enumerate(file_alarms):
                if matched[k]:
                    continue
                if onset <= t <= onset + max_latency:
                    matched[k] = True
                    det_time = t
                    break
            if det_time is not None:
                n_detected += 1
                latencies.append(det_time - onset)
        n_false_alarms += sum(1 for k in range(len(file_alarms)) if not matched[k])

    return {
        "sensibility": n_detected / n_seizures if n_seizures > 0 else 0.0,
        "false_detection_per_hour": n_false_alarms / total_hours if total_hours > 0 else 0.0,
        "latency_mean": float(np.mean(latencies)) if latencies else 0.0,
        "latency_median": float(np.median(latencies)) if latencies else 0.0,
        "n_seizures": n_seizures,
        "n_detected": n_detected,
        "n_alarms": n_alarms,
        "n_false_alarms": n_false_alarms,
        "total_hours": total_hours,
    }
