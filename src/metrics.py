"""
Métricas de evaluación para detección de crisis.

  - Nivel SEGMENTO (ventana): sensibilidad, especificidad, accuracy, AUC y fp/hora.
  - Nivel EVENTO (crisis): sensibilidad + tasa de falsas alarmas por hora + latencia, con postprocesado "n positivos en N ventanas consecutivas" + intervalo mínimo entre alarmas

También expone:
  - `select_operating_point`: el umbral con MENOR tasa de falsas alarmas por hora que cumpla sensibilidad >= objetivo
    (lo que uso para elegir el mejor checkpoint en train)
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

#media entre precisión y recall, pondera x igual falsos negativos y falsos positivos. Se calcula a nivel ventana
def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

#coeficiente de correlacion de matthews: correlacion entre lo predicho y lo real considerando las 4 celdas. Es robusto al desbalance de clases
def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def binary_metrics(probs, labels, threshold: float, *, file_ids=None, local_ids=None) -> dict:
    """Métricas por VENTANA sobre la matriz de confusión.
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

def event_metrics(probs, labels, *, file_ids, local_ids, valid_files, annotations,
                  threshold: float, n_within: int = 3, n_window: int = 4,
                  min_alarm_interval: float = 30.0, max_latency: float = 30.0) -> dict:
    """
    Evaluación a nivel evento (como en el paper 1DCNN IEEE TNSRE 2025).
    Evalúa un umbral fijo así q lo llamo barias veces desde select_event_operating_point para que me decida cuál es mejor
    Devuelve sensibilidad (evento), falsas alarmas por hora de registro cubierto, latencia media y mediana, y conteos crudos.
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    file_ids = np.asarray(file_ids)
    local_ids = np.asarray(local_ids)

    pred = probs >= threshold
    total_hours = _span_hours_by_file(file_ids, local_ids)

    # Alarmas por archivo (postprocesado sobre las predicciones en orden temporal).
    alarms_by_file: dict[int, list[float]] = {}
    for fid in np.unique(file_ids):
        fid = int(fid)
        idx = np.flatnonzero(file_ids == fid)
        order = np.argsort(local_ids[idx])
        idx = idx[order]
        predictions = pred[idx]
        # La red ve la ventana completa y ubica la decisión cuando termina la ventana, no en su instante de inicio.
        #o sea este es un array de los tiempos en los q termina cada ventana
        win_times = decision_times_from_local_ids(local_ids[idx])

        alarm_times: list[float] = []
        last_alarm = -np.inf
        recent: list[bool] = []
        #por cada una de las ventanas, si hay n_within predicciones positivas en las últimas n_window ventanas consecutivas, 
        # y no hubo alarma en los últimos min_alarm_interval segundos, dispara una alarma
        for j, t in enumerate(win_times): #(contador, tiempo de decision)
            recent.append(bool(predictions[j])) #miro ventana x ventana y anoto si esa ventana predijo o no crisis
            if len(recent) > n_window: #mantengo solo las últimas n_window predicciones
                recent.pop(0)
            if len(recent) >= n_window and sum(recent) >= n_within: #si en las últimas n_window ventanas consecutivas hubo al menos n_within predicciones positivas, disparo alarma
                if t - last_alarm >= min_alarm_interval:
                    alarm_times.append(t)
                    last_alarm = t
        alarms_by_file[fid] = alarm_times

    # Matcheo de alarmas contra crisis anotadas.
    n_seizures = 0
    n_detected = 0
    n_false_alarms = 0
    n_alarms = 0
    latencies: list[float] = []
    by_patient: dict[str, dict] = {}

    for fid, path in enumerate(valid_files): #recorro los archivos del splot
        seizures = annotations.get(path.name, []) #lista de tuplas (onset, offset) de crisis anotadas para ese archivo
        file_alarms = sorted(alarms_by_file.get(fid, [])) #las alarmas q disparó este archivo, ordenadas por tiempo
        n_alarms += len(file_alarms)
        matched = [False] * len(file_alarms) #Lista de flags para saber si cada alarma fue matcheada con una crisis anotada
        file_detected = 0
        file_latencies: list[float] = []
        for onset, _end in seizures: #recorro cada crisis
            n_seizures += 1
            det_time = None #tiempo de detección de la crisis, si se detecta
            for k, t in enumerate(file_alarms):
                if matched[k]: #si la alarma esta ya la use, la salteo
                    continue
                if onset <= t <= onset + max_latency: #si la alarma cae dentro del rango de latencia permitido (o sea entre q arranca y los 30segs), la matcheo con la crisis
                    matched[k] = True
                    det_time = t
                    break
            if det_time is not None: #la guardo en "detectadas", calculo la latencia
                n_detected += 1
                latency = det_time - onset
                latencies.append(latency)
                file_latencies.append(latency)
                file_detected += 1
        file_false_alarms = sum(1 for k in range(len(file_alarms)) if not matched[k])
        n_false_alarms += file_false_alarms #cuento las alarmas que quedaron sin matchear! q son mis falsas alarmas
#ahora agrupo por paciente
        patient = path.parent.name #creo un diccionario por paciente, donde voy a ir guardando las métricas de cada uno
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
#por cada paciente calculo sensibilidad, falsas alarmas por hora y latencia media y mediana
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
        del patient_metrics["latencies"] #ya no me hace falta guardar esto
#falsas alarmas del split entero
    event_false_alarms_per_hour = n_false_alarms / total_hours if total_hours > 0 else 0.0
   #mediana de falsas alarmas por hora por paciente, para que un paciente ruidoso no domine la selección
    per_patient_fdr = [
        patient_metrics["false_alarms_per_hour"]
        for patient_metrics in by_patient.values()
        if patient_metrics["hours"] > 0
    ]
    median_false_alarms_per_hour = (
        float(np.median(per_patient_fdr)) if per_patient_fdr else 0.0
    )
    return {
        "sensibility": n_detected / n_seizures if n_seizures > 0 else 0.0,
        "event_false_alarms_per_hour": event_false_alarms_per_hour,
        "median_false_alarms_per_hour": median_false_alarms_per_hour,
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
                                 max_false_alarms_per_hour: float, n_within: int = 3, n_window: int = 4,
                                 min_alarm_interval: float = 30.0, max_latency: float = 30.0,
                                 thresholds=None) -> tuple[dict | None, float | None]:
    """
    Punto de operación a nivel EVENTO: entre los umbrales con mediana de falsas alarmas por hora por paciente NO supera `max_false_alarms_per_hour`, se elige el
    de MAYOR sensibilidad de crisis (desempate: menor mediana de falsas alarmas por hora).

    Devuelve (event_metrics, threshold) del punto elegido, o (None, None) si ningún
    umbral respeta el techo de falsas alarmas.
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if thresholds is None:
        thresholds = THRESHOLD_GRID

    best: dict | None = None
    best_threshold: float | None = None
    for t in thresholds: #me hace un barrido de todos los thresholds (de 0.05 a 0.95, va de a 0.05 avanzando)
        ev = event_metrics(probs, labels, file_ids=file_ids, local_ids=local_ids,
                           valid_files=valid_files, annotations=annotations, threshold=t,
                           n_within=n_within, n_window=n_window,
                           min_alarm_interval=min_alarm_interval, max_latency=max_latency)
        if ev["median_false_alarms_per_hour"] <= max_false_alarms_per_hour: #filtro por los thresholds que cumplen el techo de falsas alarmas por hora
           #voy definiendo cuál es mejor: el que tenga mayor sensibilidad, y si hay empate, el que tenga menor mediana de falsas alarmas por hora
            better = (best is None
                or ev["sensibility"] > best["sensibility"]
                or (
                    ev["sensibility"] == best["sensibility"]
                    and ev["median_false_alarms_per_hour"] < best["median_false_alarms_per_hour"]
                ))
            if better:
                best = ev
                best_threshold = float(t)
    return best, best_threshold
