"""
Métricas de evaluación para detección de crisis.

Sigue la convención de los papers del marco teórico:

  - Nivel SEGMENTO (ventana): sensibilidad, especificidad, accuracy, AUC y fp/hora.
  - Nivel EVENTO (crisis): sensibilidad + tasa de falsas alarmas por hora + latencia,
    con postprocesado "n positivos en N ventanas consecutivas" + intervalo mínimo
    entre alarmas (IEEE TNSRE 2025, "EEG-Based Seizure Onset Detection ... 1DCNN").

También expone:
  - `threshold_sweep`: la curva sensibilidad <-> fp/h barriendo umbrales.
  - `select_operating_point`: el umbral con MENOR tasa de falsas alarmas por hora que cumpla sensibilidad >= objetivo
    (lo que se usa para elegir el mejor checkpoint en train.py).
"""

from __future__ import annotations

import numpy as np

from src.config import FS, STRIDE_SAMPLES, WIN_SECONDS_EFFECTIVE
from src.protocol import THRESHOLD_GRID
from src.timing import decision_times_from_local_ids


def _span_hours(n_windows: int) -> float:
    """Horas cubiertas por `n_windows` ventanas solapadas al 50%."""
    if n_windows <= 0:
        return 0.0
    span_seconds = (n_windows - 1) * (STRIDE_SAMPLES / FS) + WIN_SECONDS_EFFECTIVE
    return span_seconds / 3600.0


def _span_hours_by_file(file_ids, local_ids) -> float:
    """Suma la duración cubierta por las ventanas de cada EDF."""
    file_ids = np.asarray(file_ids).reshape(-1)
    local_ids = np.asarray(local_ids).reshape(-1)
    if file_ids.shape != local_ids.shape:
        raise ValueError("file_ids y local_ids deben tener la misma longitud")

    total_hours = 0.0
    for file_id in np.unique(file_ids):
        file_windows = local_ids[file_ids == file_id]
        if file_windows.size:
            total_hours += _span_hours(int(file_windows.max()) + 1)
    return total_hours


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


def binary_metrics(probs, labels, threshold: float, *, file_ids=None, local_ids=None) -> dict:
    """Métricas por VENTANA (segment-based) sobre la matriz de confusión.
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    tp, fp, tn, fn = _counts(probs, labels, threshold)
    pos = tp + fn
    neg = fp + tn
    total = pos + neg
    total_hours = (
        _span_hours_by_file(file_ids, local_ids)
        if file_ids is not None and local_ids is not None
        else _span_hours(int(labels.shape[0]))
    )
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


def threshold_sweep(probs, labels, thresholds=None, *, file_ids=None, local_ids=None) -> list[dict]:
    """Curva sensibilidad <-> fp/h barriendo umbrales. Una fila (dict) por umbral."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if thresholds is None:
        thresholds = THRESHOLD_GRID
    total_hours = (
        _span_hours_by_file(file_ids, local_ids)
        if file_ids is not None and local_ids is not None
        else _span_hours(int(labels.shape[0]))
    )
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
      porque el tiempo se reinicia en cada EDF). La alarma se ubica al final de
      la ventana, que es cuando la prediccion esta disponible.
    - Supresión: entre dos alarmas debe haber >= `min_alarm_interval` segundos.
    - Una crisis anotada se considera DETECTADA si hay una alarma dentro de
      [onset, onset + max_latency].
    - Latencia = tiempo desde el onset de la crisis hasta su primera alarma.

    Devuelve sensibilidad (evento), falsas alarmas por hora de registro
    cubierto, latencia media y mediana, y conteos crudos.
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    file_ids = np.asarray(file_ids)
    local_ids = np.asarray(local_ids)

    pred = probs >= threshold
    total_hours = _span_hours_by_file(file_ids, local_ids)

    # 1) Alarmas por archivo (postprocesado sobre las predicciones en orden temporal).
    alarms_by_file: dict[int, list[float]] = {}
    for fid in np.unique(file_ids):
        fid = int(fid)
        idx = np.flatnonzero(file_ids == fid)
        order = np.argsort(local_ids[idx])
        idx = idx[order]
        p = pred[idx]
        # La red ve la ventana completa. La convención offline ubica la decisión
        # cuando termina la ventana, no en su instante de inicio.
        win_times = decision_times_from_local_ids(local_ids[idx])

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
    by_patient: dict[str, dict] = {}

    for fid, path in enumerate(valid_files):
        seizures = annotations.get(path.name, [])
        file_alarms = sorted(alarms_by_file.get(fid, []))
        n_alarms += len(file_alarms)
        matched = [False] * len(file_alarms)
        file_detected = 0
        file_latencies: list[float] = []
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
                latency = det_time - onset
                latencies.append(latency)
                file_latencies.append(latency)
                file_detected += 1
        file_false_alarms = sum(1 for k in range(len(file_alarms)) if not matched[k])
        n_false_alarms += file_false_alarms

        patient = path.parent.name
        patient_metrics = by_patient.setdefault(patient, {
            "n_files": 0,
            "n_seizures": 0,
            "n_detected": 0,
            "n_alarms": 0,
            "n_false_alarms": 0,
            "hours": 0.0,
            "latencies": [],
        })
        patient_metrics["n_files"] += 1
        patient_metrics["n_seizures"] += len(seizures)
        patient_metrics["n_detected"] += file_detected
        patient_metrics["n_alarms"] += len(file_alarms)
        patient_metrics["n_false_alarms"] += file_false_alarms
        patient_metrics["hours"] += _span_hours(int(local_ids[file_ids == fid].max()) + 1)
        patient_metrics["latencies"].extend(file_latencies)

    for patient_metrics in by_patient.values():
        patient_metrics["sensitivity"] = (
            patient_metrics["n_detected"] / patient_metrics["n_seizures"]
            if patient_metrics["n_seizures"] > 0 else 0.0
        )
        patient_metrics["false_alarms_per_hour"] = (
            patient_metrics["n_false_alarms"] / patient_metrics["hours"]
            if patient_metrics["hours"] > 0 else 0.0
        )
        patient_metrics["latency_mean"] = (
            float(np.mean(patient_metrics["latencies"]))
            if patient_metrics["latencies"] else 0.0
        )
        patient_metrics["latency_median"] = (
            float(np.median(patient_metrics["latencies"]))
            if patient_metrics["latencies"] else 0.0
        )
        del patient_metrics["latencies"]

    event_false_alarms_per_hour = n_false_alarms / total_hours if total_hours > 0 else 0.0
    return {
        "sensibility": n_detected / n_seizures if n_seizures > 0 else 0.0,
        "event_false_alarms_per_hour": event_false_alarms_per_hour,
        # Alias de lectura para historiales generados por versiones anteriores.
        "false_detection_per_hour": event_false_alarms_per_hour,
        "latency_mean": float(np.mean(latencies)) if latencies else 0.0,
        "latency_median": float(np.median(latencies)) if latencies else 0.0,
        "n_seizures": n_seizures,
        "n_detected": n_detected,
        "n_alarms": n_alarms,
        "n_false_alarms": n_false_alarms,
        "total_hours": total_hours,
        "by_patient": by_patient,
    }


def select_event_operating_point(probs, labels, *, file_ids, local_ids, valid_files, annotations,
                                 min_sensibility: float, n_within: int = 3, n_window: int = 4,
                                  min_alarm_interval: float = 30.0, max_latency: float = 30.0,
                                  thresholds=None) -> tuple[dict | None, float | None]:
    """
    Punto de operación a nivel EVENTO: entre los umbrales cuya sensibilidad de CRISIS
    >= `min_sensibility`, el de MENOR tasa de falsas alarmas por hora.

    Recorre la misma malla de umbrales que `threshold_sweep`, evalúa cada uno con
    `event_metrics` y se queda con el mejor (menor tasa de falsas alarmas por hora) que cumpla el objetivo de
    sensibilidad de crisis.

    Devuelve (event_metrics, threshold) del punto elegido, o (None, None) si ningún
    umbral alcanza el objetivo de sensibilidad a nivel evento.
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if thresholds is None:
        thresholds = THRESHOLD_GRID

    best: dict | None = None
    best_threshold: float | None = None
    for t in thresholds:
        ev = event_metrics(probs, labels, file_ids=file_ids, local_ids=local_ids,
                           valid_files=valid_files, annotations=annotations, threshold=t,
                           n_within=n_within, n_window=n_window,
                           min_alarm_interval=min_alarm_interval, max_latency=max_latency)
        if ev["sensibility"] >= min_sensibility:
            if best is None or ev["event_false_alarms_per_hour"] < best["event_false_alarms_per_hour"]:
                best = ev
                best_threshold = float(t)
    return best, best_threshold
