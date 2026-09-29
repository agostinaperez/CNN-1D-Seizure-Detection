"""Construye la recomendación final a partir de predicciones out-of-fold.

Uso:
    python -m src.finalize_cv \
        --experiment-dir models/cv/ratio3_pw1 \
        --out results/ratio3_pw1_recommendation.json

Las predicciones OOF son predicciones de pacientes que no participaron del
entrenamiento de su fold. Por eso sirven para fijar threshold y época sin mirar
el test ni reutilizar una validación fija.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.config import MIN_EVENT_SENSITIVITY
from src.metrics import event_metrics, select_event_operating_point
from src.timing import validate_decision_time_mode


def _fold_number(path: Path) -> int:
    """Extrae el número de fold del nombre fold_N.oof.npz."""
    match = re.search(r"fold_(\d+)", path.name)
    if match is None:
        raise ValueError(f"No se pudo identificar el fold en {path.name}")
    return int(match.group(1))


def _load_oof(experiment_dir: Path, expected_folds: int) -> tuple[dict, list[dict]]:
    """Une predicciones y anotaciones de todos los folds en una sola línea temporal."""
    npz_paths = sorted(experiment_dir.glob("fold_*.oof.npz"), key=_fold_number)
    if len(npz_paths) != expected_folds:
        raise ValueError(
            f"Se esperaban {expected_folds} archivos OOF en {experiment_dir}, "
            f"pero se encontraron {len(npz_paths)}."
        )

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_file_ids: list[np.ndarray] = []
    all_local_ids: list[np.ndarray] = []
    valid_files: list[Path] = []
    annotations: dict[str, list] = {}
    histories: list[dict] = []
    fold_ids: list[int] = []
    file_offset = 0

    for npz_path in npz_paths:
        # El nombre debe identificar un fold único y esperado.
        fold_id = _fold_number(npz_path)
        if fold_id in fold_ids:
            raise ValueError(f"Fold duplicado en {experiment_dir}: {fold_id}")
        fold_ids.append(fold_id)
        # Carga las salidas de la mejor época del fold.
        with np.load(npz_path, allow_pickle=False) as arrays:
            probs = np.asarray(arrays["probs"]).reshape(-1)
            labels = np.asarray(arrays["labels"]).reshape(-1)
            file_ids = np.asarray(arrays["file_ids"]).reshape(-1)
            local_ids = np.asarray(arrays["local_ids"]).reshape(-1)
        if not (len(probs) == len(labels) == len(file_ids) == len(local_ids)):
            raise ValueError(f"Arrays OOF desalineados en {npz_path}")

        # Recupera los nombres de EDF y sus intervalos de crisis del mismo fold.
        meta_path = npz_path.with_suffix("").with_suffix(".oof.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fold_files = [Path(name) for name in meta["valid_files"]]
        fold_annotations = meta.get("annotations", {})
        if len(set(path.name for path in fold_files)) != len(fold_files):
            raise ValueError(f"Archivos EDF duplicados en metadata de {npz_path}")
        if file_ids.size and (int(file_ids.min()) < 0 or int(file_ids.max()) >= len(fold_files)):
            raise ValueError(f"file_ids fuera de rango en {npz_path}")

        # Reindexa file_ids para que cada fold ocupe un bloque independiente.
        all_probs.append(probs)
        all_labels.append(labels)
        all_file_ids.append(file_ids + file_offset)
        all_local_ids.append(local_ids)
        valid_files.extend(fold_files)
        annotations.update(fold_annotations)
        file_offset += len(fold_files)

        # Lee metadata para derivar época, hiperparámetros y consistencia.
        history_path = npz_path.with_name(npz_path.name.replace(".oof.npz", ".history.json"))
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        if not payload.get("checkpoint_saved", False) or not payload.get("oof_saved", False):
            raise ValueError(f"El fold {fold_id} no declara checkpoint y OOF completos.")
        if payload.get("cv_fold") != fold_id:
            raise ValueError(f"La metadata del historial no coincide con fold_{fold_id}.")
        if payload.get("cv_n_folds") != expected_folds:
            raise ValueError(
                f"La metadata de {npz_path} no coincide con expected_folds={expected_folds}."
            )
        histories.append(payload)

    if set(fold_ids) != set(range(1, expected_folds + 1)):
        raise ValueError(f"Folds encontrados {sorted(fold_ids)}; se esperaban 1..{expected_folds}.")
    if len(set(valid_file.name for valid_file in valid_files)) != len(valid_files):
        raise ValueError("El mismo EDF aparece en más de un fold OOF.")

    combined = {
        "probs": np.concatenate(all_probs),
        "labels": np.concatenate(all_labels),
        "file_ids": np.concatenate(all_file_ids),
        "local_ids": np.concatenate(all_local_ids),
        "valid_files": valid_files,
        "annotations": annotations,
    }
    return combined, histories


def _fallback_max_sensitivity(
    combined: dict,
    thresholds=None,
    decision_time_mode: str = "window_end",
) -> tuple[dict, float]:
    """Elige el máximo de sensibilidad solo para dejar una recomendación informativa."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    best = None
    best_threshold = None
    for threshold in thresholds:
        metrics = event_metrics(
            combined["probs"], combined["labels"],
            file_ids=combined["file_ids"], local_ids=combined["local_ids"],
            valid_files=combined["valid_files"], annotations=combined["annotations"],
            threshold=float(threshold),
            decision_time_mode=decision_time_mode,
        )
        if best is None or metrics["sensibility"] > best["sensibility"]:
            best = metrics
            best_threshold = float(threshold)
    return best, best_threshold


def build_recommendation(
    experiment_dir: Path,
    expected_folds: int = 4,
    min_event_sensitivity: float = MIN_EVENT_SENSITIVITY,
) -> dict:
    """Selecciona threshold global y época final usando solamente OOF."""
    combined, histories = _load_oof(experiment_dir, expected_folds)
    # Todas las predicciones OOF deben compartir la misma convención temporal.
    modes = {payload.get("decision_time_mode") for payload in histories}
    if len(modes) != 1 or None in modes:
        raise ValueError(f"decision_time_mode inconsistente entre folds: {modes}")
    decision_time_mode = validate_decision_time_mode(modes.pop())
    # El criterio de selección se aplica sobre todos los pacientes OOF juntos.
    selected, threshold = select_event_operating_point(
        combined["probs"], combined["labels"],
        file_ids=combined["file_ids"], local_ids=combined["local_ids"],
        valid_files=combined["valid_files"], annotations=combined["annotations"],
        min_sensibility=min_event_sensitivity,
        decision_time_mode=decision_time_mode,
    )
    eligible = selected is not None
    if selected is None:
        # No habilita el entrenamiento final clínicamente elegible, pero deja
        # registrado el mejor compromiso alcanzable para diagnóstico.
        selected, threshold = _fallback_max_sensitivity(combined, decision_time_mode=decision_time_mode)

    # Se usa la mediana de las mejores épocas de los folds como calendario final.
    best_epochs = [int(payload["best_epoch"]) for payload in histories if payload.get("best_epoch", 0) > 0]
    if not best_epochs:
        raise ValueError("Los historiales no contienen best_epoch válido.")
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))

    # Todos los folds de un experimento deben compartir estos hiperparámetros.
    keys = ("neg_pos_ratio", "pos_weight", "seed", "decision_time_mode", "learning_rate", "weight_decay", "batch_size")
    metadata = {}
    for key in keys:
        values = {payload.get(key) for payload in histories}
        if len(values) != 1:
            raise ValueError(f"{key} no es consistente entre folds: {values}")
        metadata[key] = values.pop()

    return {
        "format_version": 1,
        "experiment": experiment_dir.name,
        "expected_folds": expected_folds,
        "min_event_sensitivity": min_event_sensitivity,
        "eligible": eligible,
        "epochs": final_epochs,
        "threshold": float(threshold),
        "oof_event_sensitivity": selected["sensibility"],
        "oof_event_fdr_per_hour": selected["false_detection_per_hour"],
        "oof_event_latency_mean": selected["latency_mean"],
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fijar threshold y época final desde OOF")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--expected-folds", type=int, default=4)
    parser.add_argument("--min-event-sensitivity", type=float, default=MIN_EVENT_SENSITIVITY)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.expected_folds < 2:
        parser.error("--expected-folds debe ser al menos 2")
    if not 0.0 <= args.min_event_sensitivity <= 1.0:
        parser.error("--min-event-sensitivity debe estar entre 0 y 1")

    recommendation = build_recommendation(
        Path(args.experiment_dir), args.expected_folds, args.min_event_sensitivity
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    print(json.dumps(recommendation, indent=2))
    print(f"Recomendación guardada en: {out}")


if __name__ == "__main__":
    main()
