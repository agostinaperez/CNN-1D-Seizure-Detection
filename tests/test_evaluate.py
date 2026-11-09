import numpy as np

import src.evaluate as evaluate_module


class _FakeDataset:
    n_positive = 1
    file_ids = np.array([0])
    local_ids = np.array([0])
    valid_files = []
    annotations = {}

    def __len__(self):
        return 1

    def _ensure_index(self):
        pass


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def test_validation_can_use_checkpoint_threshold_without_reselecting(monkeypatch, tmp_path):
    checkpoint = {
        "op_threshold": 0.7,
        "threshold_grid": [0.5, 0.7],
        "event_config": {},
        "split_id": "split",
        "scaler_stats": {"split_id": "split"},
    }
    dataset = _FakeDataset()
    monkeypatch.setattr(evaluate_module, "get_device", lambda device: "cpu")
    monkeypatch.setattr(evaluate_module, "load_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(evaluate_module, "rebuild_model", lambda checkpoint: _FakeModel())
    monkeypatch.setattr(evaluate_module, "split_id", lambda split: "split")
    monkeypatch.setattr(evaluate_module, "validate_checkpoint_scaler", lambda checkpoint, stats: None)
    monkeypatch.setattr(
        evaluate_module,
        "build_splits_dataloaders",
        lambda *args, **kwargs: {"val": {"dataset": dataset, "dataloader": object()}},
    )
    monkeypatch.setattr(
        evaluate_module,
        "evaluate",
        lambda *args, **kwargs: (0.1, np.array([0.8]), np.array([1])),
    )
    monkeypatch.setattr(
        evaluate_module,
        "binary_metrics",
        lambda *args, **kwargs: {"sensibility": 1.0},
    )
    monkeypatch.setattr(evaluate_module, "rank_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        evaluate_module,
        "event_metrics",
        lambda *args, **kwargs: {"sensibility": 1.0},
    )

    def unexpected_reselection(*args, **kwargs):
        raise AssertionError("No debería re-seleccionar el umbral en este modo.")

    monkeypatch.setattr(evaluate_module, "select_event_operating_point", unexpected_reselection)
    monkeypatch.setattr(evaluate_module, "select_lowest_false_alarm_point", unexpected_reselection)

    result = evaluate_module.evaluate_checkpoint_on_split(
        tmp_path / "checkpoint.pt",
        data_dir=tmp_path,
        split={},
        split_name="val",
        reselect_operating_point=False,
    )

    assert result["threshold"] == 0.7
    assert result["event_selection"] == "checkpoint_threshold"
