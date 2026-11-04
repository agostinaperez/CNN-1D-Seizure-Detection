import numpy as np
import pytest

from src.config import CHANNELS_TUEV, FS
from src.preprocessing import (
    apply_scaler,
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
