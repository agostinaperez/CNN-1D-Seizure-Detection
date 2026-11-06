import numpy as np
import pytest

from src.config import CHANNELS_TUEV, FS
from src.preprocessing import (
    apply_scaler,
    compute_stats_for_patients,
    filter_bandpass,
    label_windows,
    normalize_channel_name,
    segment_windows,
)


def test_normalize_channel_name():
    assert normalize_channel_name("fp1-f7") == "FP1-F7"
    assert normalize_channel_name(" F7-CS2 ") == "F7-CS2"
    # corrige "01" -> "O1" y recorta sufijos duplicados
    assert normalize_channel_name("F7-01") == "F7-O1"


def test_filter_bandpass_keeps_shape_and_is_finite():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=(4, 1024))
    out = filter_bandpass(data, fs=FS)
    assert out.shape == data.shape
    assert np.isfinite(out).all()


def test_filter_bandpass_rejects_invalid_params():
    with pytest.raises(ValueError):
        filter_bandpass(np.zeros((1, 100)), fs=FS, low=60, high=50)


def test_segment_windows():
    data = np.zeros((2, 5000), dtype=np.float32)
    windows, starts = segment_windows(data, win_samples=1310, stride=655)
    n = (5000 - 1310) // 655 + 1
    assert windows.shape == (n, 2, 1310)
    assert starts.shape == (n,)
    assert starts[0] == 0 and starts[1] == 655


def test_label_windows_overlap():
    starts = np.array([0, 655, 1310], dtype=np.int64)
    labels = label_windows(starts, 3, [(1.0, 3.0)], fs=FS, win_seconds=1310 / FS)
    np.testing.assert_array_equal(labels, np.array([1, 1, 0]))


def test_apply_scaler():
    windows = np.full((2, 16, 10), 5.0, dtype=np.float32)
    stats = {
        "channels": list(CHANNELS_TUEV),
        "median": np.zeros(16, dtype=np.float64),
        "iqr": np.ones(16, dtype=np.float64),
    }
    out = apply_scaler(windows, stats)
    assert out.shape == windows.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 5.0)


def test_compute_stats_for_patients_filters_by_patient(tmp_path, monkeypatch):
    """El scaler por fold debe usar SOLO los archivos de los pacientes pedidos.

    Se monkeypatchea la carga de anotaciones, la resolución de archivos y el cómputo
    pesado de stats (que requiere EDFs reales), para verificar el MAPEO
    archivos/anotaciones y la metadata embebida sin tocar el dataset.
    """
    from pathlib import Path

    import src.preprocessing as pp
    from src.protocol import split_id

    data_dir = tmp_path / "chbmit"

    fake_annotations = {
        "chb01": {"chb01_01.edf": [(10, 30)]},
        "chb02": {"chb02_01.edf": [(5, 12)]},
        "chb99": {"chb99_01.edf": [(1, 2)]},  # no debe entrar
    }

    def fake_files_for_patient(data_dir, patient):
        return [Path(data_dir) / patient / f"{patient}_01.edf"]

    captured: dict = {}

    def fake_preprocess_train_data(files, seizures_by_file, verbose=True):
        captured["files"] = sorted(p.name for p in files)
        captured["seizures"] = {k: list(v) for k, v in sorted(seizures_by_file.items())}
        return {
            "scaler": "robust",
            "channels": list(CHANNELS_TUEV),
            "n_files": len(files),
            "n_windows": 100,
            "n_samples": 100 * 1310,
            "n_points": 1000,
            "median": np.zeros(16, dtype=np.float64),
            "iqr": np.ones(16, dtype=np.float64),
        }

    monkeypatch.setattr(pp, "load_annotations", lambda d: fake_annotations)
    monkeypatch.setattr(pp, "files_for_patient", fake_files_for_patient)
    monkeypatch.setattr(pp, "preprocess_train_data", fake_preprocess_train_data)

    split = {
        "protocol_version": 1,
        "train": ["chb01", "chb02"],
        "val": [],
        "test": ["chb07"],
        "test_ratio_solicitado": 0.21,
        "n_val_patients_config": 5,
        "val_required_patients": [],
        "forced_train_patients": [],
        "forced_test_patients": [],
        "split_w_zsec": 2.0,
    }

    stats = compute_stats_for_patients(
        data_dir, ["chb01", "chb02"], split=split, out=None, verbose=False,
    )

    # Solo los archivos de chb01/chb02, nunca chb99.
    assert captured["files"] == ["chb01_01.edf", "chb02_01.edf"]
    assert set(captured["seizures"]) == {"chb01_01.edf", "chb02_01.edf"}

    # Metadata de protocolo embebida y consistente con el split del fold.
    assert stats["split_id"] == split_id(split)
    assert stats["scaler"] == "robust"
    assert stats["scaler_id"]  # no vacío
    assert stats["preprocessing_config"] is not None


def test_compute_stats_for_patients_rejects_empty(monkeypatch):
    """Sin archivos para los pacientes pedidos debe fallar claro."""
    import src.preprocessing as pp

    monkeypatch.setattr(pp, "load_annotations", lambda d: {})
    monkeypatch.setattr(pp, "files_for_patient", lambda d, p: [])

    with pytest.raises(RuntimeError):
        compute_stats_for_patients("dummy", ["chb01"], verbose=False)
