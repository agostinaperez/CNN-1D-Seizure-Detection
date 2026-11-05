import numpy as np

from src.baseline import N_FEATURES, band_power_features


def test_band_power_features_shape_and_finite():
    rng = np.random.default_rng(0)
    windows = rng.normal(0, 1, size=(4, 16, 1310)).astype(np.float32)
    feats = band_power_features(windows)
    assert feats.shape == (4, N_FEATURES)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_band_power_features_non_negative_powers():
    rng = np.random.default_rng(1)
    windows = rng.normal(0, 1, size=(3, 16, 1310)).astype(np.float32)
    feats = band_power_features(windows)
    # 80 potencias absolutas no negativas
    assert (feats[:, :80] >= 0).all()
    # 80 potencias relativas suman ~1 por ventana
    relative = feats[:, 80:160].reshape(3, -1)
    np.testing.assert_allclose(relative.sum(axis=1), 1.0, atol=1e-5)


def test_band_power_features_empty():
    feats = band_power_features(np.empty((0, 16, 1310), dtype=np.float32))
    assert feats.shape == (0, N_FEATURES)
