"""Tests del split inter-paciente (src/split.py) con stats sintéticas."""

import pytest

from src.split import pick_validation, split_dataset


def _stats():
    return [
        {"patient": "chb01", "n_files": 10, "seizure_seconds": 100},
        {"patient": "chb02", "n_files": 10, "seizure_seconds": 200},
        {"patient": "chb03", "n_files": 10, "seizure_seconds": 300},
        {"patient": "chb04", "n_files": 10, "seizure_seconds": 400},
        {"patient": "chb05", "n_files": 10, "seizure_seconds": 500},
    ]


def test_split_dataset_respects_forced_and_integrity():
    train, test = split_dataset(
        _stats(), test_ratio=0.3,
        forced_train_patients=["chb01"], forced_test_patients=["chb02"],
    )
    assert "chb01" in train and "chb01" not in test
    assert "chb02" in test and "chb02" not in train
    # partición completa y sin solapamiento
    assert set(train) | set(test) == {"chb01", "chb02", "chb03", "chb04", "chb05"}
    assert not (set(train) & set(test))


def test_split_dataset_rejects_unknown_forced_patient():
    with pytest.raises(ValueError):
        split_dataset(_stats(), forced_train_patients=["chb99"])


def test_pick_validation_required_first_and_size():
    stats = _stats()
    train = [s["patient"] for s in stats]
    val = pick_validation(train, stats, n_val=3, required_patients=["chb03"])
    assert len(val) == 3
    assert val[0] == "chb03"
    assert set(val) <= set(train)


def test_pick_validation_rejects_required_not_in_train():
    with pytest.raises(ValueError):
        pick_validation(["chb01", "chb02"], _stats(), n_val=2, required_patients=["chb05"])
