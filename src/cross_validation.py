"""
Generacion de folds de validacion cruzada agrupados por paciente.

El test definido en split.json queda intacto. Los pacientes de train y val
forman el conjunto de desarrollo y se reparten en folds completos, de modo
que nunca haya ventanas del mismo paciente en train y validacion.

Uso:
    python -m src.cross_validation --n-folds 4
    python -m src.cross_validation --split-file data/processed/split.json \
        --out data/processed/cv_splits.json --seed 42
"""

from __future__ import annotations

import argparse  # Construye la interfaz de línea de comandos.
import copy  # Copia el split base sin modificarlo accidentalmente.
import hashlib  # Calcula una firma reproducible de las listas de pacientes.
import json  # Lee y escribe los archivos de split.
from pathlib import Path

import numpy as np  # Se usa para comparar cargas de pacientes entre folds.

from src.config import PROCESSED_DIR, SEED, SPLIT_FILE


def split_signature(split: dict) -> str:
    """Firma estable de las listas de pacientes de un split."""
    # Solo se firma la pertenencia de pacientes a cada conjunto, no rutas ni metadatos.
    payload = {
        # El orden se conserva porque también forma parte del split reproducible.
        "train": list(split.get("train", [])),
        "val": list(split.get("val", [])),
        "test": list(split.get("test", [])),
    }
    # Se serializa con orden estable para obtener siempre los mismos bytes.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    # La firma permite detectar si se evalúa un checkpoint contra otro split.
    return hashlib.sha256(encoded).hexdigest()


def _patient_load(split: dict, patient: str) -> tuple[float, float, float]:
    """Devuelve (archivos, segundos de crisis, cantidad de crisis)."""
    # Busca las estadísticas precalculadas del paciente en el JSON original.
    info = split.get("patients", {}).get(patient, {})
    # Convierte todo a float para poder combinar las tres magnitudes numéricamente.
    return (
        float(info.get("n_files", 0)),
        float(info.get("seizure_seconds", 0)),
        float(info.get("n_seizures", 0)),
    )


def make_patient_folds(
    split: dict,
    n_folds: int = 4,
    seed: int = SEED,
) -> list[dict]:
    """
    Construye folds balanceados por paciente.

    Se usan train+val del split base como conjunto de desarrollo. El test
    original se copia sin cambios a cada fold y nunca participa del reparto.
    El algoritmo greedy asigna primero los pacientes mas grandes al fold que
    deja menor carga normalizada de archivos, segundos y cantidad de crisis.
    """
    # Una validación cruzada necesita al menos dos grupos de validación.
    if n_folds < 2:
        raise ValueError("n_folds debe ser al menos 2")

    # Train y val originales forman el conjunto de desarrollo para la CV.
    development = list(dict.fromkeys(split.get("train", []) + split.get("val", [])))
    # El test original se conserva aparte y jamás se reparte.
    test = list(split.get("test", []))
    if len(development) < n_folds:
        raise ValueError(
            f"Hay {len(development)} pacientes de desarrollo y se pidieron "
            f"{n_folds} folds."
        )
    if set(development) & set(test):
        raise ValueError("El test comparte pacientes con train/val.")

    # El generador fijo hace que la asignación sea repetible con la misma seed.
    rng = np.random.default_rng(seed)
    # El orden principal es por severidad. El ruido solo desempata pacientes
    # con exactamente la misma carga y mantiene el resultado reproducible.
    # Añade un desempate aleatorio reproducible entre pacientes de igual severidad.
    tie_break = {patient: float(rng.random()) for patient in development}
    ordered = sorted(
        development,
        key=lambda patient: (*_patient_load(split, patient), tie_break[patient]),
        reverse=True,
    )

    # Calcula la carga total para comparar folds en escala normalizada.
    total = np.sum([_patient_load(split, patient) for patient in development], axis=0)
    # Evita divisiones por cero si alguna magnitud es cero en un dataset artificial.
    total = np.where(total > 0, total, 1.0)
    # Guarda la carga acumulada de cada fold en archivos, segundos y crisis.
    loads = np.zeros((n_folds, 3), dtype=np.float64)
    # Guarda los pacientes asignados a cada fold.
    fold_patients: list[list[str]] = [[] for _ in range(n_folds)]
    # Fuerza folds con igual cantidad de pacientes cuando el total es divisible.
    base_size, remainder = divmod(len(development), n_folds)
    capacities = [base_size + (index < remainder) for index in range(n_folds)]

    for patient in ordered:
        load = np.asarray(_patient_load(split, patient), dtype=np.float64)
        # No se permite asignar más pacientes que la capacidad prevista del fold.
        available = np.asarray(
            [len(fold_patients[index]) < capacities[index] for index in range(n_folds)]
        )
        # Simula añadir el paciente a cada fold y suma las cargas normalizadas.
        candidate_scores = ((loads + load) / total).sum(axis=1)
        # Los folds llenos no pueden ser candidatos.
        candidate_scores[~available] = np.inf
        # En un empate se elige el fold con menos pacientes y luego el indice
        # menor para que la asignacion sea totalmente determinista.
        min_score = float(candidate_scores.min())
        candidates = np.flatnonzero(np.isclose(candidate_scores, min_score))
        fold_id = min(candidates, key=lambda idx: (len(fold_patients[idx]), int(idx)))
        # Asigna el paciente al fold menos cargado.
        fold_patients[int(fold_id)].append(patient)
        # Actualiza la carga acumulada para las siguientes asignaciones.
        loads[int(fold_id)] += load

    folds: list[dict] = []
    for fold_id, validation in enumerate(fold_patients, start=1):
        # Convierte la lista de validación en set para comprobar pertenencia rápido.
        validation_set = set(validation)
        # El train del fold contiene todos los pacientes de desarrollo restantes.
        train = [patient for patient in development if patient not in validation_set]
        # Copia el JSON base para conservar metadatos y el test original.
        fold_split = copy.deepcopy(split)
        fold_split["train"] = train
        fold_split["val"] = validation
        fold_split["test"] = test
        fold_split["cv_fold"] = fold_id
        fold_split["cv_n_folds"] = n_folds
        fold_split["cv_seed"] = seed

        # Actualiza la etiqueta de split solo para que el JSON sea auditable.
        for patient, info in fold_split.get("patients", {}).items():
            if patient in validation_set:
                info["split"] = "val"
            elif patient in train:
                info["split"] = "train"
            elif patient in test:
                info["split"] = "test"

        folds.append(fold_split)

    return folds


def make_final_split(split: dict) -> dict:
    """Crea el split final: todo desarrollo en train y test intacto."""
    # Train y val originales son los pacientes disponibles para el desarrollo final.
    development = list(dict.fromkeys(split.get("train", []) + split.get("val", [])))
    # Se copia el JSON base para conservar sus estadísticas de pacientes.
    final_split = copy.deepcopy(split)
    # El entrenamiento final usa los 16 pacientes de desarrollo.
    final_split["train"] = development
    # No hay validación interna: CV ya fijó época y threshold.
    final_split["val"] = []
    # El test sigue exactamente igual y no participa del entrenamiento.
    final_split["test"] = list(split.get("test", []))
    # Metadata para distinguirlo de un fold normal.
    final_split["final_train"] = True
    final_split["cv_fold"] = None
    final_split["cv_n_folds"] = None
    final_split["cv_seed"] = None
    # Etiquetas legibles en la tabla de pacientes.
    for patient, info in final_split.get("patients", {}).items():
        if patient in development:
            info["split"] = "train"
        elif patient in final_split["test"]:
            info["split"] = "test"
    # Recalcula los contadores globales para que el JSON final sea auditable.
    def volume(pool: list[str]) -> dict:
        selected = [final_split["patients"][patient] for patient in pool]
        return {
            "files": sum(int(info.get("n_files", 0)) for info in selected),
            "seizure_seconds": sum(
                float(info.get("seizure_seconds", 0)) for info in selected
            ),
        }

    final_split["n_train"] = len(final_split["train"])
    final_split["n_val"] = 0
    final_split["n_test"] = len(final_split["test"])
    final_split["volumen"] = {
        "train": volume(final_split["train"]),
        "val": volume([]),
        "test": volume(final_split["test"]),
        "total": volume(final_split["train"] + final_split["test"]),
    }
    return final_split


def write_cv_files(
    split_file: Path = SPLIT_FILE,
    out: Path = PROCESSED_DIR / "cv_splits.json",
    n_folds: int = 4,
    seed: int = SEED,
) -> tuple[Path, list[Path]]:
    """Genera el artefacto resumen y un split.json plano por cada fold."""
    # Carga el split base desde disco.
    split = json.loads(split_file.read_text(encoding="utf-8"))
    # Genera todas las particiones sin modificar el archivo fuente.
    folds = make_patient_folds(split, n_folds=n_folds, seed=seed)
    # Genera el split que usará el entrenamiento final con todo desarrollo.
    final_split = make_final_split(split)

    out.parent.mkdir(parents=True, exist_ok=True)
    fold_paths: list[Path] = []
    # Escribe un split plano por fold para que train/preprocessing puedan consumirlo.
    for index, fold in enumerate(folds, start=1):
        fold_path = out.with_name(f"{out.stem}_fold_{index}.json")
        fold_path.write_text(json.dumps(fold, indent=2, ensure_ascii=False), encoding="utf-8")
        fold_paths.append(fold_path)

    # Guarda el split final aparte de los folds de validación.
    final_path = out.with_name(f"{out.stem}_final.json")
    final_path.write_text(
        json.dumps(final_split, indent=2, ensure_ascii=False), encoding="utf-8"
    )


    # Además crea un resumen pequeño con la relación entre folds y archivos.
    artifact = {
        "format_version": 1,
        "source_split": str(split_file),
        "seed": seed,
        "n_folds": n_folds,
        "test": split.get("test", []),
        "development_patients": list(dict.fromkeys(split.get("train", []) + split.get("val", []))),
        "fold_files": [str(path) for path in fold_paths],
        "final_file": str(final_path),
        "folds": [
            {
                "fold": fold["cv_fold"],
                "train": fold["train"],
                "val": fold["val"],
                "test": fold["test"],
            }
            for fold in folds
        ],
    }
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, fold_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Folds inter-paciente para validacion cruzada")
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--out", type=str, default=str(PROCESSED_DIR / "cv_splits.json"))
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.n_folds < 2:
        parser.error("--n-folds debe ser al menos 2")

    out, fold_paths = write_cv_files(
        Path(args.split_file), Path(args.out), args.n_folds, args.seed
    )
    print(f"CV guardada en: {out}")
    for index, fold_path in enumerate(fold_paths, start=1):
        fold = json.loads(fold_path.read_text(encoding="utf-8"))
        print(
            f"fold {index}: train={len(fold['train'])} pacientes, "
            f"val={len(fold['val'])} pacientes -> {fold_path}"
        )
    print(f"Split final (train+val de desarrollo): {out.with_name(f'{out.stem}_final.json')}")
    print(f"Test fijo en todos los folds: {json.loads(Path(args.split_file).read_text(encoding='utf-8')).get('test', [])}")


if __name__ == "__main__":
    main()
