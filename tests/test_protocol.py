"""Tests de metadatos de protocolo (src/protocol.py)."""

import numpy as np
import pytest

from src.config import CHANNELS_TUEV
from src.protocol import (
    preprocessing_config,
    scaler_id,
    split_id,
    validate_checkpoint_scaler,
)


def _stats():
    return {
        "channels": list(CHANNELS_TUEV),
        "median": np.zeros(16, dtype=np.float64),
        "iqr": np.ones(16, dtype=np.float64),
        "scaler": "robust",
        "split_id": "some-split",
        "preprocessing_config": preprocessing_config(),
    }


def test_validate_checkpoint_scaler_ok():
    stats = _stats()
    stats["scaler_id"] = scaler_id(stats)
    checkpoint = {"scaler_id": stats["scaler_id"], "preprocessing_config": preprocessing_config()}
    validate_checkpoint_scaler(checkpoint, stats)  # no debe levantar


def test_validate_checkpoint_scaler_detects_tampered_median():
    stats = _stats()
    stats["scaler_id"] = scaler_id(stats)
    stats["median"][0] += 1.0  # altero la mediana después de calcular el id
    checkpoint = {"scaler_id": stats["scaler_id"], "preprocessing_config": preprocessing_config()}
    with pytest.raises(RuntimeError, match="corruptas"):
        validate_checkpoint_scaler(checkpoint, stats)


def test_validate_checkpoint_scaler_rejects_missing_scaler_id():
    stats = _stats()  # sin "scaler_id"
    checkpoint = {"scaler_id": "x", "preprocessing_config": preprocessing_config()}
    with pytest.raises(RuntimeError, match="scaler_id"):
        validate_checkpoint_scaler(checkpoint, stats)


def test_scaler_id_deterministic_and_sensitive():
    a = _stats()
    b = _stats()
    assert scaler_id(a) == scaler_id(b)
    b["median"][0] += 1.0
    assert scaler_id(a) != scaler_id(b)


def test_split_id_deterministic_and_distinct():
    s1 = {"train": ["a"], "val": ["b"], "test": ["c"]}
    s2 = {"train": ["a"], "val": ["b"], "test": ["d"]}
    assert split_id(s1) == split_id(s1)
    assert split_id(s1) != split_id(s2)
