import numpy as np

from src.config import FS, STRIDE_SAMPLES, WIN_SECONDS_EFFECTIVE
from src.timing import decision_times_from_local_ids, decision_times_from_starts


def test_decision_times_from_starts():
    starts = np.array([0.0, 5.12])
    out = decision_times_from_starts(starts)
    np.testing.assert_allclose(out, [WIN_SECONDS_EFFECTIVE, 5.12 + WIN_SECONDS_EFFECTIVE])


def test_decision_times_from_local_ids():
    local_ids = np.array([0, 1, 2])
    expected = local_ids * (STRIDE_SAMPLES / FS) + WIN_SECONDS_EFFECTIVE
    np.testing.assert_allclose(decision_times_from_local_ids(local_ids), expected)
