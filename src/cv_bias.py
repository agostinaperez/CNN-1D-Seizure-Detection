"""
Análisis del sesgo optimista de la validación cruzada (offline, sin reentrenar).

El problema
-----------
En la CV actual, para cada fold k el modelo usa las ETIQUETAS del propio fold k para
tomar DOS decisiones:
  1. qué ÉPOCA guardar (early stopping + criterio clínico sobre el val del fold k);
  2. qué UMBRAL de decisión usar (barrido de la grilla sobre el val del fold k).

Como después reportás las métricas sobre ese MISMO fold k, el número sale optimista
(las decisiones "vieron" las etiquetas que después medís). El conjunto de TEST no
tiene este sesgo: umbral y época se fijaron ANTES de verlo.

Este módulo cuantifica el sesgo SIN volver a entrenar, con dos análisis complementarios:

  * ``--epoch-bias``   (opción 2): para cada fold k, se usa como época la MEDIANA de
    los mejores epochs de los OTROS folds (leave-one-out). Como las curvas por época
    ya están en ``fold{i}.history.json``, alcanza con leerlas. Mide el sesgo de la
    SELECCIÓN DE ÉPOCA. Las métricas de ranking (AUPRC / AUC-ROC) quedan limpiamente
    debiased; las de evento todavía usan el umbral propio del fold en esa época
    (el sesgo de umbral se mide aparte, con ``--threshold-bias``).

  * ``--threshold-bias`` (opción 1): para cada fold k, se elige el umbral con las
    predicciones de validación de los OTROS folds (leave-one-out) y se evalúa el
    fold k con ese umbral. Mide el sesgo de la SELECCIÓN DE UMBRAL. Requiere las
    probabilidades de validación guardadas por ``run_fold`` (``fold{i}.preds.npz``
    + ``fold{i}.preds.json``) o regeneradas con ``evaluate_checkpoint_predictions``.

Uso
---
    python -m src.cv_bias --out-dir cv_run --epoch-bias
    python -m src.cv_bias --out-dir cv_run --threshold-bias
    python -m src.cv_bias --out-dir cv_run --epoch-bias --configs focal_a025

La salida se escribe en ``<out-dir>/cv_bias_report.json`` y se imprime en pantalla.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.config import (
    CV_CONFIGS,
    CV_OUT_DIR,
    EVENT_MAX_LATENCY,
    EVENT_MIN_ALARM_INTERVAL,
    MAX_FALSE_ALARMS_PER_HOUR,
    POSITIVES_FOR_EVENT,
    WINDOW_RANGE_FOR_EVENT,
)
from src.metrics import (
    event_metrics,
    select_event_operating_point,
    select_lowest_false_alarm_point,
)
from src.protocol import THRESHOLD_GRID

# ---------------------------------------------------------------------------
# Helpers de E/S (artefactos que ya existen en <out_dir>/<config>/)
# ---------------------------------------------------------------------------


def _fold_result_paths(out_dir: Path, config_id: str) -> list[tuple[int, Path]]:
    """Devuelve [(fold_idx, ruta foldX.result.json)] ordenados por índice de fold."""
    run_dir = out_dir / config_id
    pairs = []
    for path in sorted(run_dir.glob("fold*.result.json")):
        # "fold3.result.json" -> 3
        idx = int(path.stem.split("fold")[1].split(".")[0])
        pairs.append((idx, path))
    return sorted(pairs, key=lambda p: p[0])


def load_fold_results(out_dir: Path, config_id: str) -> list[dict]:
    """Carga los ``fold*.result.json`` de una configuración (ya ordenados por fold)."""
    results = []
    for _idx, path in _fold_result_paths(out_dir, config_id):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    if not results:
        raise RuntimeError(f"No hay fold*.result.json en {out_dir / config_id}")
    return results


def load_fold_history(out_dir: Path, config_id: str, fold_idx: int) -> dict:
    """Carga el ``fold{fold_idx}.history.json`` de una configuración."""
    path = out_dir / config_id / f"fold{fold_idx}.history.json"
    if not path.is_file():
        raise RuntimeError(f"No existe {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Selección de época por mediana leave-one-out (opción 2)
# ---------------------------------------------------------------------------


def median_epoch_leave_one_out(fold_results: list[dict]) -> dict[int, int]:
    """Para cada fold k, devuelve la MEDIANA (redondeada) de los best_epoch de los demás folds.

    Es la misma idea que usa ``refit_final`` para elegir la cantidad de épocas del
    refit, pero aplicada fold por fold: así la época del fold k nunca depende de las
    etiquetas del fold k.
    """
    epochs_by_fold = {int(r["fold"]): int(r["best_epoch"]) for r in fold_results}
    fold_indices = sorted(epochs_by_fold)
    out: dict[int, int] = {}
    for k in fold_indices:
        others = [epochs_by_fold[j] for j in fold_indices if j != k]
        # round() en lugar de int() para que .5 redondee al entero más cercano.
        out[k] = max(1, int(round(float(np.median(others))))) if others else epochs_by_fold[k]
    return out


def _history_at_epoch(history: dict, epoch: int) -> tuple[dict | None, int]:
    """Devuelve (fila de métricas de la época pedida, época efectivamente usada).

    Si el entrenamiento terminó antes de esa época (early stopping), se usa la
    época más cercana disponible. Devuelve (None, 0) si el historial está vacío.
    """
    rows = history.get("history") or []
    if not rows:
        return None, 0
    by_epoch = {int(row["epoch"]): row for row in rows}
    if epoch in by_epoch:
        return by_epoch[epoch], epoch
    nearest = min(by_epoch, key=lambda e: abs(e - epoch))
    return by_epoch[nearest], nearest


def _aggregate_scalars(values: list[float]) -> dict:
    """Resumen mediana ± IQR (más min/max) de una lista de escalares, como en cv.aggregate."""
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if vals.size == 0:
        return {"median": None, "q25": None, "q75": None, "iqr": None, "min": None, "max": None, "values": []}
    q25, q75 = np.percentile(vals, [25, 75])
    return {
        "median": float(np.median(vals)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "values": [float(v) for v in vals],
    }


# Mapa: nombre de métrica -> (clave en result.json, clave en history.json por época).
# Las de ranking (auprc, auc_roc) son independientes del umbral -> el debias de época las deja limpias.
# Las de evento dependen del umbral -> quedan debiased SOLO en época; el sesgo de umbral se mide aparte.
_EPOCH_METRICS = {
    "auprc": ("auprc", "val_auprc"),
    "auc_roc": ("auc_roc", "val_auc_roc"),
    "event_sensibility": ("event_sensibility", "event_sensibility"),
    # Ojo: el result.json usa "event_median_fa_per_hour" pero el history.json por época usa
    # "event_median_false_alarms_per_hour" (dos nombres distintos para lo mismo).
    "event_median_fa_per_hour": ("event_median_fa_per_hour", "event_median_false_alarms_per_hour"),
}


def epoch_debiased_summary(out_dir: Path, config_ids: list[str]) -> dict:
    """Compara, config por config, las métricas REPORTADAS (en el best_epoch propio de
    cada fold) contra las DEBIASED (en el epoch mediano de los otros folds).

    Devuelve un dict listo para serializar y para ``print_epoch_bias_report``.
    """
    report: dict = {}
    for cid in config_ids:
        fold_results = load_fold_results(out_dir, cid)
        median_epochs = median_epoch_leave_one_out(fold_results)

        per_fold: list[dict] = []
        # Acumulo valores reportados y debiased de cada métrica para agregar después.
        reported_vals: dict[str, list] = {m: [] for m in _EPOCH_METRICS}
        debiased_vals: dict[str, list] = {m: [] for m in _EPOCH_METRICS}

        for r in fold_results:
            fold_idx = int(r["fold"])
            median_epoch = median_epochs[fold_idx]
            history = load_fold_history(out_dir, cid, fold_idx)
            row, used_epoch = _history_at_epoch(history, median_epoch)

            fold_report = {
                "fold": fold_idx,
                "best_epoch": int(r["best_epoch"]),
                "median_epoch": median_epoch,
                "used_epoch": used_epoch,
                "epoch_exact": used_epoch == median_epoch,
                "reported": {},
                "debiased": {},
            }
            for metric, (result_key, history_key) in _EPOCH_METRICS.items():
                reported = r.get(result_key)
                debiased = row.get(history_key) if row else None
                fold_report["reported"][metric] = reported
                fold_report["debiased"][metric] = debiased
                reported_vals[metric].append(reported)
                debiased_vals[metric].append(debiased)
            per_fold.append(fold_report)

        reported_agg = {m: _aggregate_scalars(reported_vals[m]) for m in _EPOCH_METRICS}
        debiased_agg = {m: _aggregate_scalars(debiased_vals[m]) for m in _EPOCH_METRICS}
        # Sesgo = reportado - debiased (positivo => optimismo). Solo si ambas medianas existen.
        bias = {
            m: (
                round(reported_agg[m]["median"] - debiased_agg[m]["median"], 6)
                if reported_agg[m]["median"] is not None and debiased_agg[m]["median"] is not None
                else None
            )
            for m in _EPOCH_METRICS
        }
        report[cid] = {
            "reported": reported_agg,
            "debiased": debiased_agg,
            "bias": bias,
            "folds": per_fold,
        }
    return report


# ---------------------------------------------------------------------------
# Umbral leave-one-out (opción 1) — requiere predicciones guardadas
# ---------------------------------------------------------------------------


def _pool_predictions(preds_list: list[dict]) -> dict:
    """Concatena varias corridas de predicciones remapeando los file_ids para que
    no colisionen entre folds (cada fold empieza su numeración en 0).

    Cada elemento de ``preds_list`` es el dict que devuelve ``load_predictions``:
        {probs, labels, file_ids, local_ids, valid_files, annotations}.
    """
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    file_ids: list[np.ndarray] = []
    local_ids: list[np.ndarray] = []
    valid_files: list[str] = []
    annotations: dict[str, list] = {}

    offset = 0
    for p in preds_list:
        probs.append(np.asarray(p["probs"], dtype=np.float64))
        labels.append(np.asarray(p["labels"], dtype=np.int64))
        file_ids.append(np.asarray(p["file_ids"], dtype=np.int64) + offset)
        local_ids.append(np.asarray(p["local_ids"], dtype=np.int64))
        valid_files.extend(p["valid_files"])
        for fname, ivs in p["annotations"].items():
            annotations[fname] = [list(iv) for iv in ivs]
        offset += len(p["valid_files"])

    return {
        "probs": np.concatenate(probs) if probs else np.empty(0, dtype=np.float64),
        "labels": np.concatenate(labels) if labels else np.empty(0, dtype=np.int64),
        "file_ids": np.concatenate(file_ids) if file_ids else np.empty(0, dtype=np.int64),
        "local_ids": np.concatenate(local_ids) if local_ids else np.empty(0, dtype=np.int64),
        "valid_files": valid_files,
        "annotations": annotations,
    }


def threshold_leave_one_out(preds_by_fold: dict[int, dict], *, event_config: dict | None = None,
                            threshold_grid: list[float] | None = None) -> dict:
    """Para cada fold k, elige el umbral con las predicciones de los OTROS folds y
    reevalúa el evento del fold k con ese umbral (leave-one-out de umbral).

    ``preds_by_fold``: {fold_idx: dict de ``load_predictions``}. Devuelve
    {fold_idx: {"threshold": t_k, "event": ev_k}}.
    """
    event_config = event_config or {}
    n_within = event_config.get("n_within", POSITIVES_FOR_EVENT)
    n_window = event_config.get("n_window", WINDOW_RANGE_FOR_EVENT)
    min_alarm_interval = event_config.get("min_alarm_interval", EVENT_MIN_ALARM_INTERVAL)
    max_latency = event_config.get("max_latency", EVENT_MAX_LATENCY)
    max_fa = event_config.get("max_false_alarms_per_hour", MAX_FALSE_ALARMS_PER_HOUR)
    if threshold_grid is None:
        threshold_grid = THRESHOLD_GRID

    fold_indices = sorted(preds_by_fold)
    out: dict = {}
    for k in fold_indices:
        others = [preds_by_fold[j] for j in fold_indices if j != k]
        pooled = _pool_predictions(others)

        # Elegimos el umbral sobre las predicciones de los OTROS folds (sin ver el k).
        op_metrics, op_threshold = select_event_operating_point(
            pooled["probs"], pooled["labels"],
            file_ids=pooled["file_ids"], local_ids=pooled["local_ids"],
            valid_files=pooled["valid_files"], annotations=pooled["annotations"],
            max_false_alarms_per_hour=max_fa,
            n_within=n_within, n_window=n_window,
            min_alarm_interval=min_alarm_interval, max_latency=max_latency,
            thresholds=threshold_grid,
        )
        if op_metrics is None or op_threshold is None:
            # Ningún umbral de los otros folds cumple el techo -> fallback al de menor mediana.
            op_metrics, op_threshold = select_lowest_false_alarm_point(
                pooled["probs"], pooled["labels"],
                file_ids=pooled["file_ids"], local_ids=pooled["local_ids"],
                valid_files=pooled["valid_files"], annotations=pooled["annotations"],
                n_within=n_within, n_window=n_window,
                min_alarm_interval=min_alarm_interval, max_latency=max_latency,
                thresholds=threshold_grid,
            )

        pk = preds_by_fold[k]
        ev = event_metrics(
            pk["probs"], pk["labels"],
            file_ids=np.asarray(pk["file_ids"]), local_ids=np.asarray(pk["local_ids"]),
            valid_files=[Path(f) for f in pk["valid_files"]], annotations=pk["annotations"],
            threshold=float(op_threshold),
            n_within=n_within, n_window=n_window,
            min_alarm_interval=min_alarm_interval, max_latency=max_latency,
        )
        out[k] = {"threshold": float(op_threshold), "event": ev}
    return out


def threshold_debiased_summary(out_dir: Path, config_ids: list[str]) -> dict:
    """Lee las predicciones guardadas de cada fold y computa el umbral leave-one-out.

    Para cada config compara: umbral REPORTADO (del result.json) vs umbral LOO, y la
    sensibilidad de evento con cada umbral. Devuelve un dict serializable.
    """
    report: dict = {}
    for cid in config_ids:
        fold_results = load_fold_results(out_dir, cid)
        preds_by_fold: dict[int, dict] = {}
        for r in fold_results:
            fold_idx = int(r["fold"])
            try:
                preds_by_fold[fold_idx] = load_predictions(out_dir, cid, fold_idx)
            except FileNotFoundError:
                raise RuntimeError(
                    f"No hay predicciones guardadas para {cid}/fold{fold_idx}. "
                    "Regeneralas con `evaluate_checkpoint_predictions` o volvé a correr la CV "
                    "con el guardado de predicciones habilitado."
                )
        loo = threshold_leave_one_out(preds_by_fold)

        per_fold: list[dict] = []
        reported_thr: list[float] = []
        loo_thr: list[float] = []
        reported_sens: list[float] = []
        loo_sens: list[float] = []
        for r in fold_results:
            fold_idx = int(r["fold"])
            reported_threshold = r.get("threshold")
            loo_threshold = loo[fold_idx]["threshold"]
            loo_ev = loo[fold_idx]["event"]
            reported_thr.append(reported_threshold)
            loo_thr.append(loo_threshold)
            reported_sens.append(r.get("event_sensibility"))
            loo_sens.append(loo_ev["sensibility"])
            per_fold.append({
                "fold": fold_idx,
                "reported_threshold": reported_threshold,
                "loo_threshold": loo_threshold,
                "reported_sensibility": r.get("event_sensibility"),
                "loo_sensibility": loo_ev["sensibility"],
                "loo_n_detected": loo_ev["n_detected"],
                "loo_n_seizures": loo_ev["n_seizures"],
                "loo_fa_per_hour": loo_ev["event_false_alarms_per_hour"],
                "loo_median_fa_per_hour": loo_ev["median_false_alarms_per_hour"],
            })

        report[cid] = {
            "threshold": {
                "reported": _aggregate_scalars(reported_thr),
                "loo": _aggregate_scalars(loo_thr),
            },
            "event_sensibility": {
                "reported": _aggregate_scalars(reported_sens),
                "loo": _aggregate_scalars(loo_sens),
            },
            "folds": per_fold,
        }
    return report


# ---------------------------------------------------------------------------
# Guardado / carga de predicciones de validación (para la opción 1)
# ---------------------------------------------------------------------------


def save_predictions(out_dir: Path, config_id: str, fold_idx: int, predictions: dict) -> Path:
    """Persiste las predicciones de validación de un fold en ``fold{i}.preds.npz``
    + ``fold{i}.preds.json`` (metadatos que no caben cómodo en un .npz).

    ``predictions`` = {probs, labels, file_ids, local_ids, valid_files, annotations}.
    """
    run_dir = out_dir / config_id
    run_dir.mkdir(parents=True, exist_ok=True)
    npz_path = run_dir / f"fold{fold_idx}.preds.npz"
    json_path = run_dir / f"fold{fold_idx}.preds.json"

    np.savez_compressed(
        npz_path,
        probs=np.asarray(predictions["probs"], dtype=np.float32),
        labels=np.asarray(predictions["labels"], dtype=np.int8),
        file_ids=np.asarray(predictions["file_ids"], dtype=np.int32),
        local_ids=np.asarray(predictions["local_ids"], dtype=np.int32),
    )
    meta = {
        "valid_files": [str(p) for p in predictions["valid_files"]],
        "annotations": {k: [list(iv) for iv in v] for k, v in predictions["annotations"].items()},
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return npz_path


def load_predictions(out_dir: Path, config_id: str, fold_idx: int) -> dict:
    """Carga las predicciones de validación guardadas por ``save_predictions``."""
    run_dir = out_dir / config_id
    npz_path = run_dir / f"fold{fold_idx}.preds.npz"
    json_path = run_dir / f"fold{fold_idx}.preds.json"
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Faltan predicciones para {config_id}/fold{fold_idx}")
    with np.load(npz_path, allow_pickle=False) as z:
        probs = z["probs"]
        labels = z["labels"]
        file_ids = z["file_ids"]
        local_ids = z["local_ids"]
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "probs": probs,
        "labels": labels,
        "file_ids": file_ids,
        "local_ids": local_ids,
        "valid_files": meta["valid_files"],
        "annotations": {k: [tuple(iv) for iv in v] for k, v in meta["annotations"].items()},
    }


# ---------------------------------------------------------------------------
# Reportes por pantalla
# ---------------------------------------------------------------------------


def print_epoch_bias_report(report: dict) -> None:
    print("\n" + "=" * 88)
    print("SESGO DE ÉPOCA (opción 2) — reportado vs época mediana de los otros folds")
    print("=" * 88)
    for cid, data in report.items():
        print(f"\n[{cid}]")
        header = (f"{'métrica':<24}{'reportado':>22}{'debiased':>22}{'sesgo':>10}")
        print(header)
        print("-" * len(header))
        for metric in _EPOCH_METRICS:
            rep = data["reported"][metric]
            deb = data["debiased"][metric]
            bias = data["bias"][metric]
            def fmt(agg):
                if not agg or agg["median"] is None:
                    return "—"
                return f"{agg['median']:.4f} ({agg['q25']:.4f}–{agg['q75']:.4f})"
            bias_s = f"{bias:+.4f}" if bias is not None else "—"
            print(f"{metric:<24}{fmt(rep):>22}{fmt(deb):>22}{bias_s:>10}")
    print("=" * 88)
    print("Nota: las métricas de evento (sensibilidad, FA/h) quedan debiased SOLO en época;")
    print("      el sesgo de umbral se mide por separado con --threshold-bias.")


def print_threshold_bias_report(report: dict) -> None:
    print("\n" + "=" * 88)
    print("SESGO DE UMBRAL (opción 1) — umbral propio vs umbral leave-one-out")
    print("=" * 88)
    for cid, data in report.items():
        print(f"\n[{cid}]")
        thr_rep = data["threshold"]["reported"]
        thr_loo = data["threshold"]["loo"]
        sens_rep = data["event_sensibility"]["reported"]
        sens_loo = data["event_sensibility"]["loo"]
        def fmt(agg):
            if not agg or agg["median"] is None:
                return "—"
            return f"{agg['median']:.3f} ({agg['q25']:.3f}–{agg['q75']:.3f})"
        print(f"  umbral:        reportado={fmt(thr_rep)}   loo={fmt(thr_loo)}")
        print(f"  sens. evento:  reportado={fmt(sens_rep)}   loo={fmt(sens_loo)}")
    print("=" * 88)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análisis del sesgo de la validación cruzada (offline).")
    parser.add_argument("--out-dir", type=str, default=str(CV_OUT_DIR),
                        help="Carpeta de resultados de la CV (contiene <config>/fold*.json).")
    parser.add_argument("--configs", type=str, default=",".join(CV_CONFIGS),
                        help="Configuraciones separadas por coma.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--epoch-bias", action="store_true",
                       help="Medir el sesgo de selección de época (opción 2).")
    group.add_argument("--threshold-bias", action="store_true",
                       help="Medir el sesgo de selección de umbral (opción 1).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
    try:
        if args.epoch_bias:
            report = epoch_debiased_summary(out_dir, config_ids)
            print_epoch_bias_report(report)
        else:
            report = threshold_debiased_summary(out_dir, config_ids)
            print_threshold_bias_report(report)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    out_path = out_dir / "cv_bias_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nReporte guardado en: {out_path}")


if __name__ == "__main__":
    main()
