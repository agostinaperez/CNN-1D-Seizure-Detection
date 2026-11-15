"""Tests del análisis de sesgo de la validación cruzada (src/cv_bias.py).

Usan artefactos sintéticos (result.json / history.json / predicciones) para verificar
la lógica de selección leave-one-out y la agregación, sin tocar el dataset ni entrenar.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import src.cv_bias as cb


# ---------------------------------------------------------------- helpers
def _preds(probs, labels, files, annotations=None):
    """Predicciones sintéticas en el formato que devuelve load_predictions."""
    n = len(probs)
    return {
        "probs": np.asarray(probs, dtype=float),
        "labels": np.asarray(labels, dtype=int),
        "file_ids": np.zeros(n, dtype=int),
        "local_ids": np.arange(n, dtype=int),
        "valid_files": list(files),
        "annotations": annotations or {f: [] for f in files},
    }


def _write_fold(out_dir, cid, fold_idx, best_epoch, reported, history_rows):
    """Escribe foldX.result.json y foldX.history.json sintéticos para una config."""
    run_dir = out_dir / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {"fold": fold_idx, "best_epoch": best_epoch, **reported}
    (run_dir / f"fold{fold_idx}.result.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    history = {"history": history_rows, "best_epoch": best_epoch}
    (run_dir / f"fold{fold_idx}.history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )


# ---------------------------------------------------------------- opción 2 (época)
def test_median_epoch_leave_one_out():
    fold_results = [
        {"fold": 0, "best_epoch": 10},
        {"fold": 1, "best_epoch": 12},
        {"fold": 2, "best_epoch": 14},
        {"fold": 3, "best_epoch": 8},
        {"fold": 4, "best_epoch": 20},
    ]
    assert cb.median_epoch_leave_one_out(fold_results) == {
        0: 13,  # mediana de [12, 14, 8, 20]
        1: 12,  # mediana de [10, 14, 8, 20]
        2: 11,  # mediana de [10, 12, 8, 20]
        3: 13,  # mediana de [10, 12, 14, 20]
        4: 11,  # mediana de [10, 12, 8, 14]
    }


def test_history_at_epoch_exact_and_nearest():
    history = {"history": [
        {"epoch": 1, "val_auprc": 0.1},
        {"epoch": 2, "val_auprc": 0.2},
        {"epoch": 5, "val_auprc": 0.5},
    ]}
    row, used = cb._history_at_epoch(history, 2)
    assert used == 2 and row["val_auprc"] == 0.2
    # 4 no existe -> la más cercana es 5 (dist 1) antes que 2 (dist 2).
    row, used = cb._history_at_epoch(history, 4)
    assert used == 5 and row["val_auprc"] == 0.5
    # historial vacío
    row, used = cb._history_at_epoch({"history": []}, 3)
    assert row is None and used == 0


def test_epoch_debiased_summary_computes_bias(tmp_path):
    # 3 folds con best_epoch 10/12/14 -> épocas medianas LOO 13/12/11.
    # La historia tiene val_auprc = epoch/100, así el "debiased" es 0.13/0.12/0.11.
    for fold_idx, best_epoch in enumerate([10, 12, 14]):
        _write_fold(
            tmp_path, "bce", fold_idx, best_epoch,
            reported={
                "auprc": {0: 0.30, 1: 0.20, 2: 0.25}[fold_idx],
                "auc_roc": 0.8,
                "event_sensibility": 0.5,
                "event_median_fa_per_hour": 0.7,
                "threshold": 0.5,
            },
            history_rows=[
                {
                    "epoch": e,
                    "val_auprc": e / 100.0,
                    "val_auc_roc": 0.8,
                    "event_sensibility": 0.5,
                    "event_median_false_alarms_per_hour": 0.7,
                }
                for e in range(1, 21)
            ],
        )

    report = cb.epoch_debiased_summary(tmp_path, ["bce"])
    data = report["bce"]

    # reportado: mediana de [0.30, 0.20, 0.25] = 0.25
    # debiased: mediana de [0.13, 0.12, 0.11] = 0.12
    assert data["reported"]["auprc"]["median"] == pytest.approx(0.25)
    assert data["debiased"]["auprc"]["median"] == pytest.approx(0.12)
    assert data["bias"]["auprc"] == pytest.approx(0.13)
    # las épocas usadas deben ser exactas (11/12/13 están en la historia)
    assert all(f["epoch_exact"] for f in data["folds"])


# ---------------------------------------------------------------- opción 1 (umbral)
def test_pool_predictions_remaps_file_ids():
    p1 = {
        "probs": np.array([0.9, 0.1]), "labels": np.array([1, 0]),
        "file_ids": np.array([0, 1]), "local_ids": np.array([0, 0]),
        "valid_files": ["a.edf", "b.edf"], "annotations": {"a.edf": [(1, 2)]},
    }
    p2 = {
        "probs": np.array([0.8]), "labels": np.array([0]),
        "file_ids": np.array([0]), "local_ids": np.array([0]),
        "valid_files": ["c.edf"], "annotations": {"c.edf": [(3, 4)]},
    }
    pooled = cb._pool_predictions([p1, p2])
    # p2 empieza con offset = len(p1.valid_files) = 2
    assert pooled["file_ids"].tolist() == [0, 1, 2]
    assert pooled["valid_files"] == ["a.edf", "b.edf", "c.edf"]
    assert set(pooled["annotations"]) == {"a.edf", "c.edf"}
    assert pooled["probs"].tolist() == [0.9, 0.1, 0.8]


def test_threshold_leave_one_out_uses_other_folds(monkeypatch):
    preds = {
        0: _preds([0.9, 0.1, 0.9], [1, 0, 1], ["f0.edf"]),
        1: _preds([0.1, 0.9, 0.1], [0, 1, 0], ["f1.edf"]),
        2: _preds([0.9, 0.9, 0.1], [1, 1, 0], ["f2.edf"]),
    }

    selections = []
    events = []

    def fake_select(probs, labels, **kw):
        selections.append({
            "probs": np.asarray(probs).copy(),
            "file_ids": np.asarray(kw["file_ids"]).copy(),
            "valid_files": list(kw["valid_files"]),
        })
        return ({"sensibility": 0.5}, 0.42)

    def fake_event(probs, labels, **kw):
        events.append({
            "threshold": kw["threshold"],
            "probs": np.asarray(probs).copy(),
        })
        return {
            "sensibility": 0.5, "n_detected": 1, "n_seizures": 2,
            "event_false_alarms_per_hour": 0.0, "median_false_alarms_per_hour": 0.0,
            "latency_mean": 1.0, "latency_median": 1.0,
            "n_alarms": 1, "n_false_alarms": 0, "total_hours": 1.0, "by_patient": {},
        }

    monkeypatch.setattr(cb, "select_event_operating_point", fake_select)
    monkeypatch.setattr(cb, "event_metrics", fake_event)

    out = cb.threshold_leave_one_out(
        preds,
        event_config={"n_within": 2, "n_window": 3, "min_alarm_interval": 6.0,
                      "max_latency": 30.0, "max_false_alarms_per_hour": 1.0},
        threshold_grid=[0.1, 0.2, 0.42],
    )

    # Para el fold 0, el umbral se eligió SOLO con los folds 1 y 2 (no el 0).
    # Además, valid_files debe llegar como Path (event_metrics usa path.name).
    assert all(isinstance(f, Path) for f in selections[0]["valid_files"])
    assert [f.name for f in selections[0]["valid_files"]] == ["f1.edf", "f2.edf"]
    assert selections[0]["file_ids"].tolist() == [0, 0, 0, 1, 1, 1]
    # Y el evento del fold 0 se evaluó con su propia data y el umbral elegido.
    assert events[0]["threshold"] == 0.42
    assert events[0]["probs"].tolist() == [0.9, 0.1, 0.9]
    assert out[0]["threshold"] == 0.42


# ---------------------------------------------------------------- guardado/carga
def test_save_and_load_predictions_roundtrip(tmp_path):
    preds = _preds(
        [0.9, 0.1, 0.8], [1, 0, 0], ["a.edf", "b.edf"],
        annotations={"a.edf": [(1, 2)], "b.edf": [(3, 4)]},
    )
    # file_ids con 2 archivos para verificar que se preservan
    preds["file_ids"] = np.array([0, 0, 1], dtype=int)
    preds["local_ids"] = np.array([0, 1, 0], dtype=int)

    cb.save_predictions(tmp_path, "bce", 0, preds)
    loaded = cb.load_predictions(tmp_path, "bce", 0)

    np.testing.assert_allclose(loaded["probs"], [0.9, 0.1, 0.8], rtol=1e-6)
    assert loaded["labels"].tolist() == [1, 0, 0]
    assert loaded["file_ids"].tolist() == [0, 0, 1]
    assert loaded["local_ids"].tolist() == [0, 1, 0]
    assert loaded["valid_files"] == ["a.edf", "b.edf"]
    assert loaded["annotations"] == {"a.edf": [(1, 2)], "b.edf": [(3, 4)]}
