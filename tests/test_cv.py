"""Tests unitarios de la validación cruzada.

Usan stats sintéticas con los mismos códigos de paciente del split real para ser
representativas, pero no tocan el disco ni el dataset: prueban la partición de
folds, las validaciones anti-leakage, la agregación de métricas y la selección de
la configuración ganadora.
"""

import pytest
from src.cv import (
    CONFIG_REGISTRY,
    aggregate,
    build_fold_split,
    build_folds,
    select_best,
    validate_folds,
    validate_pools,
)
from src.protocol import split_id

# Mismos pools que el split real: test congelado (6) y desarrollo (18).
TEST_PATIENTS = ["chb07", "chb09", "chb16", "chb17", "chb18", "chb20"]
DEV_PATIENTS = [
    "chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb08", "chb10",
    "chb11", "chb12", "chb13", "chb14", "chb15", "chb19", "chb21", "chb22",
    "chb23", "chb24",
]

# Segundos de crisis reales del pool de desarrollo (para estratificación realista).
_ZSEC = {
    "chb01": 442, "chb02": 172, "chb03": 402, "chb04": 378, "chb05": 558,
    "chb06": 153, "chb08": 919, "chb10": 447, "chb11": 806, "chb12": 1475,
    "chb13": 535, "chb14": 169, "chb15": 1992, "chb19": 236, "chb21": 199,
    "chb22": 204, "chb23": 424, "chb24": 511,
}

_BASE_SPLIT = {
    "protocol_version": 1,
    "train": [p for p in DEV_PATIENTS if p not in {"chb03", "chb11", "chb14", "chb22", "chb24"}],
    "val": ["chb03", "chb11", "chb14", "chb22", "chb24"],
    "test": TEST_PATIENTS,
    "test_ratio_solicitado": 0.21,
    "n_val_patients_config": 5,
    "val_required_patients": ["chb14"],
    "forced_train_patients": ["chb06"],
    "forced_test_patients": ["chb16"],
    "split_w_zsec": 2.0,
}


def _make_stats() -> list[dict]:
    return [
        {
            "patient": p,
            "n_files": 30,
            "n_seizures": 5,
            "seizure_seconds": _ZSEC[p],
            "mean_seizure_seconds": _ZSEC[p] / 5,
        }
        for p in DEV_PATIENTS
    ]


def _fold_result(auprc, event_sens, detected, seizures):
    return {
        "auprc": auprc,
        "auc_roc": 0.8,
        "sens_window": 0.5,
        "spec_window": 0.9,
        "threshold": 0.7,
        "event_sensibility": event_sens,
        "event_n_detected": detected,
        "event_n_seizures": seizures,
        "event_fa_per_hour": 1.0,
        "event_median_fa_per_hour": 0.5,
        "latency_mean": 10.0,
        "best_epoch": 17,
        "best_val_loss": 0.02,
    }


def test_bce_config_does_not_set_focal_parameters():
    assert CONFIG_REGISTRY["bce"]["loss"] == "bce"
    assert CONFIG_REGISTRY["bce"]["focal_alpha"] is None
    assert CONFIG_REGISTRY["bce"]["focal_gamma"] is None


# ---------------------------------------------------------------- folds
def test_build_folds_integrity():
    folds = build_folds(DEV_PATIENTS, _make_stats(), K=5, seed=42)
    assert len(folds) == 5
    validate_folds(folds, DEV_PATIENTS)  # no debe levantar
    for fold in folds:
        assert set(fold["train"]).isdisjoint(set(fold["val"]))
        assert set(fold["train"]) | set(fold["val"]) == set(DEV_PATIENTS)
        assert fold["val"]


def test_default_build_folds_is_standard_and_covers_each_patient_once():
    folds = build_folds(DEV_PATIENTS, _make_stats(), K=5, seed=42)
    assert len(folds) == 5
    val_counts = {
        patient: sum(patient in fold["val"] for fold in folds)
        for patient in DEV_PATIENTS
    }
    assert set(val_counts.values()) == {1}
    assert all(set(fold["train"]).isdisjoint(fold["val"]) for fold in folds)


def test_build_folds_deterministic():
    a = build_folds(DEV_PATIENTS, _make_stats(), K=5, seed=42)
    b = build_folds(DEV_PATIENTS, _make_stats(), K=5, seed=42)
    assert a == b


def test_build_folds_seed_changes_partition():
    a = build_folds(DEV_PATIENTS, _make_stats(), K=5, seed=42)
    b = build_folds(DEV_PATIENTS, _make_stats(), K=5, seed=7)
    assert a != b


def test_build_folds_rejects_invalid_k():
    with pytest.raises(ValueError):
        build_folds(DEV_PATIENTS, _make_stats(), K=1, seed=42)
    with pytest.raises(ValueError):
        build_folds(DEV_PATIENTS, _make_stats(), K=len(DEV_PATIENTS) + 1, seed=42)


def test_validate_folds_rejects_leakage():
    bad = [{"fold": 0, "train": ["chb01", "chb02"], "val": ["chb02"]}]
    with pytest.raises(ValueError):
        validate_folds(bad, ["chb01", "chb02"])


# ---------------------------------------------------------------- pools
def test_validate_pools_ok():
    validate_pools(DEV_PATIENTS, TEST_PATIENTS)


def test_validate_pools_rejects_test_in_dev():
    with pytest.raises(ValueError):
        validate_pools(DEV_PATIENTS + ["chb07"], TEST_PATIENTS)


# ---------------------------------------------------------------- split dict
def test_build_fold_split_preserves_test_and_metadata():
    s = build_fold_split(_BASE_SPLIT, ["chb01", "chb02"], ["chb06", "chb14", "chb03"])
    assert s["test"] == TEST_PATIENTS
    assert s["val"] == ["chb06", "chb14", "chb03"]
    assert s["test_ratio_solicitado"] == 0.21
    assert s["n_val_patients_config"] == 3
    assert s["val_required_patients"] == ["chb14"]
    assert s["forced_train_patients"] == []
    assert s["forced_test_patients"] == ["chb16"]
    # no muta el split original
    assert _BASE_SPLIT["test"] == TEST_PATIENTS
    assert len(_BASE_SPLIT["val"]) == 5


def test_build_fold_split_ids_consistent():
    s1 = build_fold_split(_BASE_SPLIT, ["chb01", "chb02"], ["chb06", "chb14", "chb03"])
    s2 = build_fold_split(_BASE_SPLIT, ["chb01", "chb03"], ["chb06", "chb14", "chb02"])
    assert split_id(s1) == split_id(s1)  # determinista
    assert split_id(s1) != split_id(s2)  # distinta partición -> distinto id


# ---------------------------------------------------------------- agregación
def test_aggregate_mean_std_min_max():
    results = [
        _fold_result(auprc=0.5, event_sens=0.5, detected=3, seizures=6),
        _fold_result(auprc=0.7, event_sens=0.8, detected=8, seizures=10),
    ]
    agg = aggregate(results, metric_keys=["auprc"])
    assert agg["auprc"]["mean"] == pytest.approx(0.6)
    assert agg["auprc"]["std"] == pytest.approx(0.1)
    assert agg["auprc"]["min"] == pytest.approx(0.5)
    assert agg["auprc"]["max"] == pytest.approx(0.7)
    assert agg["auprc"]["values"] == pytest.approx([0.5, 0.7])


def test_aggregate_pooled_sensitivity():
    results = [
        _fold_result(auprc=0.5, event_sens=0.5, detected=3, seizures=6),
        _fold_result(auprc=0.7, event_sens=0.8, detected=8, seizures=10),
    ]
    agg = aggregate(results)
    # pooled: (3 + 8) / (6 + 10) = 11/16 (pondera por crisis, no por fold)
    assert agg["event_sensibility_pooled"] == pytest.approx(11 / 16)
    assert agg["event_counts"] == {"detected": 11, "seizures": 16}


def test_aggregate_handles_none_values():
    results = [
        _fold_result(auprc=0.5, event_sens=0.5, detected=3, seizures=6),
        {**_fold_result(auprc=None, event_sens=None, detected=0, seizures=0), "auprc": None, "auc_roc": None},
    ]
    agg = aggregate(results, metric_keys=["auprc"])
    assert agg["auprc"]["mean"] == pytest.approx(0.5)


# ---------------------------------------------------------------- selección
def test_select_best_highest_mean():
    summary = {
        "a": {"auprc": {"mean": 0.5, "std": 0.1}},
        "b": {"auprc": {"mean": 0.6, "std": 0.2}},
    }
    assert select_best(summary, metric="auprc") == "b"


def test_select_best_tie_breaker_lowest_std():
    summary = {
        "b": {"auprc": {"mean": 0.6, "std": 0.2}},
        "c": {"auprc": {"mean": 0.6, "std": 0.05}},
    }
    assert select_best(summary, metric="auprc") == "c"


def test_select_best_ignores_missing_metrics():
    summary = {
        "a": {"auprc": {"mean": 0.5, "std": 0.1}},
        "b": {"auprc": None},
    }
    assert select_best(summary, metric="auprc") == "a"
