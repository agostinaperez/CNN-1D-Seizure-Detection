"""Convenciones temporales compartidas por evaluación e inferencia offline."""

from __future__ import annotations

import numpy as np  # Permite trabajar con arrays de tiempos de forma vectorizada.

from src.config import DECISION_TIME_MODE, FS, STRIDE_SAMPLES, WIN_SECONDS_EFFECTIVE

# Estos son los únicos nombres aceptados para describir cuándo ocurre una decisión.
VALID_DECISION_TIME_MODES = frozenset({"window_start", "window_end"})


def validate_decision_time_mode(mode: str) -> str:
    """Valida y devuelve el modo temporal usado por un checkpoint o la config."""
    # Evita que una configuración mal escrita produzca métricas temporalmente inválidas.
    if mode not in VALID_DECISION_TIME_MODES:
        # Se informa explícitamente qué valores puede usar la persona.
        valid = ", ".join(sorted(VALID_DECISION_TIME_MODES))
        raise ValueError(f"decision_time_mode inválido: {mode!r}. Opciones: {valid}")
    # Devuelve el mismo valor para poder encadenar validación y uso en una expresión.
    return mode


def decision_times_from_starts(
    starts: np.ndarray,
    mode: str = DECISION_TIME_MODE,
) -> np.ndarray:
    """Convierte inicios de ventana en tiempos de decisión."""
    # Primero se valida el modo para no calcular tiempos con una convención desconocida.
    validate_decision_time_mode(mode)
    # Se fuerza float64 para conservar precisión al sumar segundos fraccionarios.
    starts = np.asarray(starts, dtype=np.float64)
    if mode == "window_end":
        # La predicción se reporta cuando termina la ventana completa.
        return starts + WIN_SECONDS_EFFECTIVE
    # En el modo alternativo, la decisión se reporta en el inicio de la ventana.
    return starts


def decision_times_from_local_ids(
    local_ids: np.ndarray,
    mode: str = DECISION_TIME_MODE,
) -> np.ndarray:
    """Convierte índices locales de ventana en tiempos de decisión."""
    # Cada ventana comienza después de `local_id * stride / fs` segundos.
    starts = np.asarray(local_ids, dtype=np.float64) * STRIDE_SAMPLES / FS
    # Reutiliza la misma regla temporal que usa inferencia y métricas.
    return decision_times_from_starts(starts, mode)
