"""
Resume las metricas de validacion de experimentos con folds por paciente.

Estructura esperada:
    models/cv/ratio3_pw1/fold_1.history.json
    models/cv/ratio3_pw1/fold_2.history.json
    ...

El script no lee resultados de test. Solo compara los historiales de validacion
despues de que terminaron todos los folds de cada experimento.

Uso:
    python -m src.compare_cv --root models/cv --expected-folds 4
    python -m src.compare_cv --root models/cv --out results/cv_summary.json
"""

from __future__ import annotations

import argparse  # Permite ejecutar la comparación desde la terminal.
import json  # Lee historiales y guarda el resumen opcional.
import math  # Representa métricas no disponibles como infinito.
from pathlib import Path
import numpy as np  # Calcula medias y desvíos entre folds.

from src.config import MIN_EVENT_SENSITIVITY, MODELS_DIR


def _best_row(payload: dict) -> dict:
    """Obtiene la fila de history correspondiente al checkpoint seleccionado."""
    # Extrae la lista de métricas acumuladas durante el entrenamiento.
    history = payload.get("history", [])
    # Busca la época que train.py marcó como mejor.
    best_epoch = payload.get("best_epoch")
    if best_epoch is not None:
        for row in history:
            if row.get("epoch") == best_epoch:
                return row
    # Para historiales incompletos, usa la última fila disponible como fallback.
    return history[-1] if history else {}


def _fold_metrics(path: Path) -> dict:
    # Carga el resumen producido junto al checkpoint.
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Obtiene la fila exacta que corresponde al checkpoint seleccionado.
    row = _best_row(payload)
    # Recupera el FDR guardado a nivel global o, en historiales viejos, desde la fila.
    fdr = payload.get("best_fdr_per_hour", row.get("event_fdr_per_hour"))
    sensitivity = payload.get(
        "best_op_sensibility",
        row.get("op_sensibility", row.get("event_sensibility")),
    )
    val_loss = payload.get("best_val_loss", row.get("val_loss"))
    # Deriva el nombre esperado del checkpoint a partir del nombre del historial.
    checkpoint_file = path.with_name(path.name.replace(".history.json", ".pt"))
    # Deriva también los dos archivos de predicciones out-of-fold.
    oof_file = path.with_name(path.name.replace(".history.json", ".oof.npz"))
    oof_meta_file = path.with_name(path.name.replace(".history.json", ".oof.json"))
    # Devuelve métricas y metadata necesarias para validar el experimento.
    return {
        "history_file": str(path),
        "checkpoint_file": str(checkpoint_file),
        "oof_file": str(oof_file),
        "oof_meta_file": str(oof_meta_file),
        "checkpoint_saved": bool(payload.get("checkpoint_saved", True)),
        "oof_saved": bool(payload.get("oof_saved", True)),
        "cv_fold": payload.get("cv_fold"),
        "cv_n_folds": payload.get("cv_n_folds"),
        "split_signature": payload.get("split_signature"),
        "neg_pos_ratio": payload.get("neg_pos_ratio"),
        "pos_weight": payload.get("pos_weight"),
        "decision_time_mode": payload.get("decision_time_mode"),
        "learning_rate": payload.get("learning_rate"),
        "weight_decay": payload.get("weight_decay"),
        "batch_size": payload.get("batch_size"),
        "seed": payload.get("seed"),
        "best_epoch": payload.get("best_epoch", row.get("epoch")),
        "fdr_per_hour": float(fdr) if fdr is not None else math.inf,
        "event_sensitivity": float(sensitivity) if sensitivity is not None else 0.0,
        "val_loss": float(val_loss) if val_loss is not None else math.inf,
        "threshold": payload.get("best_op_threshold", row.get("op_threshold")),
    }


def summarize_cv(
    root: Path,
    expected_folds: int = 4,
    min_event_sensitivity: float = MIN_EVENT_SENSITIVITY,
) -> dict:
    """Agrupa historiales por experimento y calcula media/desvio por fold."""
    # Busca historiales dentro de una carpeta por experimento.
    histories = sorted(root.rglob("*.history.json")) if root.exists() else []
    grouped: dict[str, list[dict]] = {}
    for path in histories:
        relative_parent = path.parent.relative_to(root)
        experiment = str(relative_parent) if str(relative_parent) != "." else root.name
        # Agrupa folds que viven dentro de la misma carpeta de experimento.
        grouped.setdefault(experiment, []).append(_fold_metrics(path))

    experiments: list[dict] = []
    for name, folds in sorted(grouped.items()):
        # Convierte las métricas de los folds en arrays para calcular estadísticas.
        fdrs = np.asarray([fold["fdr_per_hour"] for fold in folds], dtype=np.float64)
        sensitivities = np.asarray(
            [fold["event_sensitivity"] for fold in folds], dtype=np.float64
        )
        finite_fdr = np.isfinite(fdrs)
        # Comprueba que los folds sean exactamente 1..N y no haya duplicados.
        fold_ids = [fold["cv_fold"] for fold in folds]
        expected_ids = set(range(1, expected_folds + 1))
        fold_ids_valid = (
            all(isinstance(fold_id, int) for fold_id in fold_ids)
            and len(set(fold_ids)) == len(fold_ids)
            and set(fold_ids) == expected_ids
        )
        # Un historial fallido no cuenta aunque haya quedado un .pt viejo en disco.
        checkpoint_files_exist = all(
            fold["checkpoint_saved"] and Path(fold["checkpoint_file"]).exists()
            for fold in folds
        )
        # La recomendación final necesita las predicciones OOF de cada fold.
        oof_files_exist = all(
            fold["oof_saved"]
            and Path(fold["oof_file"]).exists()
            and Path(fold["oof_meta_file"]).exists()
            for fold in folds
        )
        # Acumula inconsistencias que invalidan la comparación del experimento.
        consistency_errors: list[str] = []
        for key in (
            "neg_pos_ratio", "pos_weight", "decision_time_mode", "seed",
            "learning_rate", "weight_decay", "batch_size",
        ):
            values = {fold[key] for fold in folds}
            if len(values) > 1:
                consistency_errors.append(f"{key} inconsistente")
        if any(fold["cv_n_folds"] not in (None, expected_folds) for fold in folds):
            consistency_errors.append("cv_n_folds inconsistente")

        complete = len(folds) == expected_folds and fold_ids_valid
        # Solo una configuración completa, consistente y clínicamente elegible
        # puede competir por el primer puesto.
        eligible = bool(
            complete
            and checkpoint_files_exist
            and oof_files_exist
            and not consistency_errors
            and finite_fdr.all()
            and float(np.mean(sensitivities)) >= min_event_sensitivity
        )
        experiments.append({
            "experiment": name,
            "n_folds": len(folds),
            "complete": complete,
            "fold_ids_valid": fold_ids_valid,
            "checkpoint_files_exist": checkpoint_files_exist,
            "oof_files_exist": oof_files_exist,
            "consistency_errors": consistency_errors,
            "eligible": eligible,
            "mean_fdr_per_hour": float(np.mean(fdrs)) if finite_fdr.all() else math.inf,
            "std_fdr_per_hour": float(np.std(fdrs)) if finite_fdr.all() else math.inf,
            "mean_event_sensitivity": float(np.mean(sensitivities)),
            "std_event_sensitivity": float(np.std(sensitivities)),
            "mean_val_loss": float(np.mean([fold["val_loss"] for fold in folds])),
            "folds": folds,
        })

    # Filtra los experimentos que cumplen todas las condiciones anteriores.
    eligible = [item for item in experiments if item["eligible"]]
    # El ganador minimiza el FDR promedio, nunca una métrica de test.
    winner = min(eligible, key=lambda item: item["mean_fdr_per_hour"]) if eligible else None
    return {
        "root": str(root),
        "expected_folds": expected_folds,
        "min_event_sensitivity": min_event_sensitivity,
        "experiments": experiments,
        "winner": winner["experiment"] if winner else None,
        "selection_metric": "mean_fdr_per_hour among complete experiments with event sensitivity >= target",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar experimentos de CV sin usar test")
    parser.add_argument("--root", type=str, default=str(MODELS_DIR / "cv"))
    parser.add_argument("--expected-folds", type=int, default=4)
    parser.add_argument("--min-event-sensitivity", type=float, default=MIN_EVENT_SENSITIVITY)
    parser.add_argument("--out", type=str, default=None, help="Guardar resumen JSON opcional.")
    args = parser.parse_args()
    if args.expected_folds < 2:
        parser.error("--expected-folds debe ser al menos 2")
    if not 0.0 <= args.min_event_sensitivity <= 1.0:
        parser.error("--min-event-sensitivity debe estar entre 0 y 1")

    # Ejecuta el análisis sobre los historiales encontrados.
    summary = summarize_cv(Path(args.root), args.expected_folds, args.min_event_sensitivity)
    if not summary["experiments"]:
        print(f"No se encontraron historiales en: {args.root}")
        return

    print("experiment | folds | completa | sens_evento media | FDR media | FDR std | elegible")
    print("-" * 86)
    # Ordena para mostrar primero los FDR más bajos.
    for item in sorted(summary["experiments"], key=lambda row: row["mean_fdr_per_hour"]):
        fdr = "inf" if not math.isfinite(item["mean_fdr_per_hour"]) else f"{item['mean_fdr_per_hour']:.3f}"
        fdr_std = "inf" if not math.isfinite(item["std_fdr_per_hour"]) else f"{item['std_fdr_per_hour']:.3f}"
        print(
            f"{item['experiment']} | {item['n_folds']} | {item['complete']} | "
            f"{item['mean_event_sensitivity']:.3f} | {fdr} | {fdr_std} | {item['eligible']}"
        )
        if item["consistency_errors"]:
            print(f"  [WARN] {', '.join(item['consistency_errors'])}")

    if summary["winner"] is None:
        print("\nNo hay un experimento completo que cumpla el objetivo de sensibilidad.")
    else:
        print(f"\nConfiguracion seleccionada por validacion: {summary['winner']}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Resumen guardado en: {out}")


if __name__ == "__main__":
    main()
