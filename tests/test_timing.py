import numpy as np

from src.config import WIN_SECONDS_EFFECTIVE
from src.timing import decision_times_from_starts


def test_decision_times_from_starts():
    starts = np.array([0.0, 5.12])
    out = decision_times_from_starts(starts)
    np.testing.assert_allclose(out, [WIN_SECONDS_EFFECTIVE, 5.12 + WIN_SECONDS_EFFECTIVE])
