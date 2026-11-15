import numpy as np
import pytest

import src.preprocessing as pp

from src.config import CHANNELS_TUEV, FS
from src.preprocessing import (
    ELECTRODES,
    _solve_selection,
    apply_scaler,
    compute_stats_for_patients,
    filter_bandpass,
    label_windows,
    load_16_channels,
    normalize_channel_name,
    segment_windows,
)


def test_normalize_channel_name():
    assert normalize_channel_name("fp1-f7") == "FP1-F7"
    assert normalize_channel_name(" F7-CS2 ") == "F7-CS2"
    # corrige "01" -> "O1"
    assert normalize_channel_name("F7-01") == "F7-O1"


def test_normalize_channel_name_trims_numeric_suffix():
    # recorta sufijos numéricos de canales duplicados
    assert normalize_channel_name("FP1-F7-1") == "FP1-F7"
    assert normalize_channel_name("F7-CS2-2") == "F7-CS2"


def test_filter_bandpass_keeps_shape_and_is_finite():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=(4, 1024))
    out = filter_bandpass(data, fs=FS)
    assert out.shape == data.shape
    assert np.isfinite(out).all()


def test_filter_bandpass_attenuates_out_of_band():
    fs = FS
    t = np.arange(4096) / fs
    in_band = np.sin(2 * np.pi * 10 * t)
    out_of_band = np.sin(2 * np.pi * 100 * t)
    y_in = filter_bandpass(in_band[None, :], fs=fs)[0]
    y_out = filter_bandpass(out_of_band[None, :], fs=fs)[0]
    core = slice(512, -512)  # descarto el transitorio del filtro de fase cero
    rms_in = float(np.sqrt(np.mean(y_in[core] ** 2)))
    rms_out = float(np.sqrt(np.mean(y_out[core] ** 2)))
    assert rms_in > 0.5             # 10 Hz está dentro de la banda -> pasa
    assert rms_out < 0.05 * rms_in  # 100 Hz queda fuertemente atenuado


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


def test_segment_windows_content_matches_starts():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(2, 5000)).astype(np.float32)
    win, stride = 1310, 655
    windows, starts = segment_windows(data, win_samples=win, stride=stride)
    for i in range(windows.shape[0]):
        np.testing.assert_allclose(windows[i], data[:, starts[i]:starts[i] + win])


# ---------------------------------------------------------------- canales (montaje)
def test_solve_selection_direct_bipolar():
    plan = _solve_selection([normalize_channel_name(c) for c in CHANNELS_TUEV])
    assert plan is not None
    for i, ch in enumerate(CHANNELS_TUEV):
        assert plan[ch] == ("row", i)


def test_solve_selection_monopolar_reconstruction():
    # todos los canales referidos a CS2 -> los bipolares se reconstruyen por resta
    ch_names = [f"{e}-CS2" for e in ELECTRODES]
    plan = _solve_selection(ch_names)
    assert plan is not None
    idx = {e: i for i, e in enumerate(ELECTRODES)}
    assert plan["FP1-F7"] == ("diff", idx["FP1"], idx["F7"])
    assert plan["T8-P8"] == ("diff", idx["T8"], idx["P8"])


def test_solve_selection_incompatible_returns_none():
    assert _solve_selection(["FP1-CS2"]) is None  # falta F7 -> no se arma FP1-F7


class _FakeRaw:
    def __init__(self, ch_names, sfreq, data):
        self.info = {"sfreq": sfreq}
        self.ch_names = list(ch_names)
        self._data = data

    def get_data(self):
        return self._data

    def close(self):
        pass


def _install_fake_raw(monkeypatch, ch_names, sfreq, data):
    monkeypatch.setattr(pp.mne.io, "read_raw_edf", lambda *a, **k: _FakeRaw(ch_names, sfreq, data))


def test_load_16_channels_direct_bipolar(monkeypatch):
    n = 100
    data = np.arange(16 * n, dtype=float).reshape(16, n)
    _install_fake_raw(monkeypatch, CHANNELS_TUEV, FS, data)
    out, reason = load_16_channels("dummy.edf", fs_expected=FS)
    assert reason is None
    np.testing.assert_array_equal(out, data)


def test_load_16_channels_reconstructs_by_diff(monkeypatch):
    n = 50
    ch_names = [f"{e}-CS2" for e in ELECTRODES]
    data = np.random.default_rng(0).normal(size=(len(ch_names), n))
    _install_fake_raw(monkeypatch, ch_names, FS, data)
    out, reason = load_16_channels("dummy.edf", fs_expected=FS)
    assert reason is None
    idx = {e: i for i, e in enumerate(ELECTRODES)}
    expected_fp1f7 = data[idx["FP1"]] - data[idx["F7"]]
    np.testing.assert_allclose(out[0], expected_fp1f7)  # FP1-F7 es el canal 0


def test_load_16_channels_rejects_wrong_sampling_rate(monkeypatch):
    data = np.zeros((16, 100))
    _install_fake_raw(monkeypatch, CHANNELS_TUEV, 128, data)
    out, reason = load_16_channels("dummy.edf", fs_expected=FS)
    assert out is None and "sampling_rate" in reason


def test_label_windows_overlap():
    starts = np.array([0, 655, 1310], dtype=np.int64)
    labels = label_windows(starts, 3, [(1.0, 3.0)], fs=FS, win_seconds=1310 / FS)
    np.testing.assert_array_equal(labels, np.array([1, 1, 0]))


def test_apply_scaler():
    windows = np.arange(2 * 16 * 3, dtype=np.float32).reshape(2, 16, 3)
    median = np.arange(1, 17, dtype=np.float64)
    iqr = np.arange(2, 18, dtype=np.float64)
    stats = {
        "channels": list(CHANNELS_TUEV),
        "median": median,
        "iqr": iqr,
    }

    out = apply_scaler(windows, stats)
    assert out.shape == windows.shape
    assert out.dtype == np.float32
    expected = (windows - median[None, :, None]) / iqr[None, :, None]
    np.testing.assert_allclose(out, expected)


def test_apply_scaler_uses_one_as_denominator_for_zero_iqr():
    windows = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    median = np.zeros(16, dtype=np.float64)
    iqr = np.full(16, 2.0, dtype=np.float64)
    iqr[0] = 0.0
    stats = {"channels": list(CHANNELS_TUEV), "median": median, "iqr": iqr}

    out = apply_scaler(windows, stats)

    expected = windows / np.where(iqr == 0, 1.0, iqr)[None, :, None]
    np.testing.assert_allclose(out, expected)


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
