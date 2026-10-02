"""
Parser de las anotaciones clínicas de CHB-MIT.

CHB-MIT viene con un archivo por paciente llamado `chbXX-summary.txt` que lista, para cada archivo `.edf`, los intervalos de crisis ("Seizure Start Time" /
"Seizure End Time") en segundos. Este módulo lo convierte en estructuras Python listas para usar (por paciente -> por archivo -> lista de intervalos).
"""

from __future__ import annotations

import re
from pathlib import Path

from src.config import DATASET_DIR

# Expresiones regulares para detectar las líneas relevantes del summary.
# - "File Name: chb01_03.edf"      -> empieza un nuevo archivo.
# - "Seizure Start Time: 2996 seconds"  (formato de chb01–chb05)
# - "Seizure 1 Start Time: 1665 seconds" (formato numerado, chb06 en adelante)
# OJO: hay DOS formatos según el paciente. El "(?:\d+\s+)?" de los regex
# acepta el "1 ", "2 ", ... opcional para que matchee los dos.
_RE_FILE = re.compile(r"^File Name:\s+(.+)")  # captura el nombre del EDF
_RE_START = re.compile(r"^Seizure\s+(?:\d+\s+)?Start Time:\s+(\d+)")
_RE_END = re.compile(r"^Seizure\s+(?:\d+\s+)?End Time:\s+(\d+)")


def is_safe_edf_name(name: str) -> bool:
    """Indica si `name` es un nombre EDF relativo y sin componentes de ruta."""
    candidate = Path(name)
    return (
        bool(name)
        and not candidate.is_absolute()
        and candidate.name == name
        and candidate.suffix.lower() == ".edf"
    )


def patient_dir(data_dir: Path | str, patient: str) -> Path:
    """Resuelve un directorio de paciente sin permitir escapar del dataset."""
    root = Path(data_dir).resolve()
    resolved = (root / patient).resolve()
    if resolved.parent != root:
        raise ValueError(f"Paciente fuera del dataset: {patient!r}")
    return resolved


def edf_path(data_dir: Path | str, patient: str, filename: str) -> Path:
    """Resuelve un EDF validando paciente, nombre y extensión."""
    if not is_safe_edf_name(filename):
        raise ValueError(f"Nombre de EDF inválido: {filename!r}")
    resolved = patient_dir(data_dir, patient) / filename
    if resolved.parent != patient_dir(data_dir, patient):
        raise ValueError(f"EDF fuera del directorio del paciente: {filename!r}")
    return resolved


def discover_patients(data_dir: Path | str = DATASET_DIR) -> list[str]:
    data_dir = Path(data_dir)

    # Lista de pacientes: carpetas que empiezan con "chb" y contienen EDFs.
    patients = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.lower().startswith("chb"):
            continue
        # Un paciente cuenta si efectivamente tiene archivos EDF dentro.
        if any(child.glob("*.edf")):
            patients.append(child.name)

    if not patients:
        raise FileNotFoundError(f"No se encontraron carpetas de pacientes 'chbNN' en: {data_dir}")

    return patients


def summary_path_for(patient: str, data_dir: Path | str = DATASET_DIR) -> Path:
    #Ruta del archivo de resumen de un paciente: `<carpeta>/<paciente>-summary.txt`.
    return patient_dir(data_dir, patient) / f"{patient}-summary.txt"


def parse_summary(summary_path: Path | str) -> dict[str, list[tuple[int, int]]]:
    """
    Parsea un archivo summary y devuelve:

        {"chb01_03.edf": [(2996, 3036)], "chb01_04.edf": [(1467, 1494)], ...}

    - Cada clave es el nombre de un archivo EDF.
    - Cada valor es una lista de tuplas (inicio_seg, fin_seg) de crisis.
    - Se devuelve una lista vacía para los archivos sin crisis (así el split sabe que existen aunque no tengan eventos).
    """

    current_file: str | None = None
    # acá guardo el valor de inicio de la crisis (me falta guardarle el end time)
    pending_start: int | None = None

    # Mapa nombre->lista de intervalos
    seizures: dict[str, list[tuple[int, int]]] = {}

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()  # limpiar saltos de línea

            # la línea "File Name: ..." indica q cambié de archivo
            m = _RE_FILE.match(line)
            if m:
                if pending_start is not None and current_file is not None:
                    print(f"[WARN] {summary_path.name}: crisis sin hora de fin en {current_file}")
                candidate = m.group(1).strip()
                if not is_safe_edf_name(candidate):
                    print(f"[WARN] {summary_path.name}: nombre de EDF inválido: {candidate!r}")
                    current_file = None
                    pending_start = None
                    continue
                current_file = candidate
                # Inicializo la lista de crisis para este arvchivo (si no tiene crisis, queda vacío y listo)
                seizures.setdefault(current_file, [])
                # reinicio esto para q no queden valores cruzados
                pending_start = None
                continue

            if current_file is None:
                continue

            m = _RE_START.match(line)
            if m:
                pending_start = int(m.group(1)) #guardo el inicio de la crisis
                continue

            m = _RE_END.match(line)
            if m and pending_start is not None:
                end = int(m.group(1))
                # no puede terminar antes de empezar.
                if end >= pending_start:
                    seizures[current_file].append((pending_start, end)) #formo el invervalo de segundos de inicio y de fin de la crisis
                else:
                    print(f"[WARN] {summary_path.name}: intervalo inválido en {current_file}: {pending_start}-{end}")
                pending_start = None #limpio el valor
    if pending_start is not None and current_file is not None:
        print(f"[WARN] {summary_path.name}: crisis sin hora de fin en {current_file}")
    return {k: v for k, v in seizures.items() if k} #devuelvo solamente los valores con nombre de archivo no vacío x si quedó alguna línea rari


def load_annotations(data_dir: Path | str = DATASET_DIR) -> dict[str, dict]:
    """
    Carga las anotaciones de todos los pacientes del dataset.
    Devuelve: {"chb01": {"chb01_03.edf": [(2996, 3036), ...], ...}, ...}
    """
    data_dir = Path(data_dir)
    annotations: dict[str, dict] = {}

    for patient in discover_patients(data_dir):
        summary = summary_path_for(patient, data_dir)
        if summary.exists():
            parsed = parse_summary(summary)
            existing: dict[str, list[tuple[int, int]]] = {}
            for filename, intervals in parsed.items():
                path = edf_path(data_dir, patient, filename)
                if path.is_file():
                    existing[filename] = intervals
                else:
                    print(f"[WARN] {patient}/{filename}: anotado pero no existe en disco; se omite.")
            annotations[patient] = existing
        else:
            print(f"[WARN] {patient}: no tiene {summary.name}; se omite.")

    return annotations


def files_for_patient(data_dir: Path | str, patient: str) -> list[Path]:
    """Devuelve los EDF existentes mencionados en el summary del paciente."""
    try:
        summary = summary_path_for(patient, data_dir)
    except ValueError as exc:
        print(f"[WARN] {patient}: ruta de paciente inválida ({exc}); se omite.")
        return []
    if not summary.exists():
        print(f"[WARN] {patient}: no tiene summary; se omite.")
        return []

    files: list[Path] = []
    for filename in parse_summary(summary):
        path = edf_path(data_dir, patient, filename)
        if path.is_file():
            files.append(path)
        else:
            print(f"[WARN] {patient}/{filename}: anotado pero no existe en disco.")
    return files


def patient_seizure_stats(annotations: dict[str, dict],) -> list[dict]:
    """
    Resumen de severidad por paciente, útil para el split proporcional.

    Para cada paciente calcula:
      - patient:     código del paciente
      - n_files:     cantidad de archivos EDF anotados (cada uno ~1 hora)
      - n_seizures:  cantidad total de eventos de crisis
      - seizure_seconds: duración total de crisis en segundos

    Se usa n_files como proxy del volumen de registro de cada paciente, y
    seizure_seconds como proxy de la "cantidad de positivos" que aporta.
    """
    stats = []

    for patient, files in annotations.items():
        # n_files: cuántos EDF menciona el summary para este paciente.
        n_files = len(files)

        # Concentramos todos los intervalos de crisis del paciente.
        all_intervals = []
        for ivs in files.values():      # 1. por cada lista de intervalos
            for iv in ivs:              # 2. por cada intervalo dentro de esa lista
                all_intervals.append(iv)

        # Cantidad de eventos de crisis.
        n_seizures = len(all_intervals)

        # Suma de duraciones de todas sus crisis (en segundos).
        seizure_seconds = sum(end - start for start, end in all_intervals)

        stats.append(
            {
                "patient": patient,
                "n_files": n_files,
                "n_seizures": n_seizures,
                "seizure_seconds": seizure_seconds,
            }
        )

    return stats


if __name__ == "__main__":
    # Imprime, por paciente, la cantidad de crisis y sus segundos totales.
    import json

    annotations = load_annotations()
    stats = patient_seizure_stats(annotations)
    #imprime corte
        #paciente  archivos  crisis  segundos
        #chb01         4        8      3589
        #chb10        12        1       340
    
    print(f"{'paciente':<8} {'archivos':>8} {'crisis':>6} {'segundos':>10}")
    for s in sorted(stats, key=lambda x: x["patient"]):
        print(
            f"{s['patient']:<8} {s['n_files']:>8} "
            f"{s['n_seizures']:>6} {s['seizure_seconds']:>10}"
        )
    print("\nResumen:", json.dumps(annotations.get("chb01"), indent=2))
