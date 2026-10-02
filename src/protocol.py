"""Metadatos del protocolo experimental.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from src.config import (
    FILTER_ORDER,
    FS,
    HIGH_FREQ,
    LOW_FREQ,
    OVERLAP,
    STRIDE_SAMPLES,
    WIN_SAMPLES,
    WIN_SECONDS_EFFECTIVE,
)


PROTOCOL_VERSION = 1
THRESHOLD_GRID = [round(i / 20, 2) for i in range(1, 20)]


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def preprocessing_config() -> dict[str, Any]:
    """Parámetros que deben coincidir entre train, evaluación e inferencia."""
    return {
        "fs": FS,
        "win_samples": WIN_SAMPLES,
        "win_seconds_effective": WIN_SECONDS_EFFECTIVE,
        "overlap": OVERLAP,
        "stride_samples": STRIDE_SAMPLES,
        "low_freq": LOW_FREQ,
        "high_freq": HIGH_FREQ,
        "filter_order": FILTER_ORDER,
    }


def split_config(split: dict) -> dict[str, Any]:
    """Extrae sólo la parte reproducible del split."""
    return {
        "protocol_version": split.get("protocol_version", PROTOCOL_VERSION),
        "train": list(split.get("train", [])),
        "val": list(split.get("val", [])),
        "test": list(split.get("test", [])),
        "test_ratio": split.get("test_ratio_solicitado"),
        "n_val_patients": split.get("n_val_patients_config"),
        "split_w_zsec": split.get("split_w_zsec"),
    }


def split_id(split: dict) -> str:
    """Hash estable de pacientes y parámetros del reparto."""
    return _digest(split_config(split))


def scaler_id(stats: dict) -> str:
    """Hash estable de los canales y estadísticas del RobustScaler."""
    payload = {
        "channels": list(stats.get("channels", [])),
        "median": np.asarray(stats["median"], dtype=np.float64).round(15).tolist(),
        "iqr": np.asarray(stats["iqr"], dtype=np.float64).round(15).tolist(),
    }
    return _digest(payload)


def compatibility_config(
    split: dict,
    scaler_stats: dict,
    *,
    model_config: dict[str, Any],
    event_config: dict[str, Any],
    max_false_alarms_per_hour: float,
    threshold_grid: list[float],
    seed: int,
) -> dict[str, Any]:
    """Parte común que debe ser idéntica entre escenarios comparados."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "split_id": split_id(split),
        "scaler_id": scaler_id(scaler_stats),
        "preprocessing": preprocessing_config(),
        "model_config": model_config,
        "event_config": event_config,
        "max_false_alarms_per_hour": float(max_false_alarms_per_hour),
        "threshold_grid": [float(t) for t in threshold_grid],
        "seed": int(seed),
    }
