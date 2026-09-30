"""Convenciones temporales compartidas."""
#MODULARIZO ESTO PQ EN EL FUTURO LO VOY A TENER Q REUTILIZAR PARA LAS CNN 2D MUY MUY PROBABLEMENTE

from __future__ import annotations

import numpy as np

from src.config import FS, STRIDE_SAMPLES, WIN_SECONDS_EFFECTIVE


def decision_times_from_starts(starts: np.ndarray) -> np.ndarray:
    """Convierte inicios de ventana en tiempos de decisión.

    La red consume la ventana COMPLETA antes de emitir su logit, así que la decisión solo está disponible cuando la ventana termina.
    """
    # Se fuerza float64 para conservar precisión al sumar segundos fraccionarios.
    starts = np.asarray(starts, dtype=np.float64)
    return starts + WIN_SECONDS_EFFECTIVE


def decision_times_from_local_ids(local_ids: np.ndarray) -> np.ndarray:
    """Convierte índices locales de ventana en tiempos de decisión."""
    # Cada ventana comienza después de `local_id * stride / fs` segundos.
    starts = np.asarray(local_ids, dtype=np.float64) * STRIDE_SAMPLES / FS
    # Reutiliza la misma regla temporal que usa inferencia y métricas.
    return decision_times_from_starts(starts)
