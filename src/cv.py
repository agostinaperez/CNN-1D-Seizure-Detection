"""
Validación cruzada agrupada por paciente (GroupKFold estratificado).

La validación cruzada (CV) estima el error de generalización dividiendo los datos en K particiones y, para cada una, entrenando sobre K-1 y evaluando sobre la restante, hasta que todas hayan sido evaluadas exactamente una vez. El resultado
es una DISTRIBUCIÓN de medidas (una por fold), de la que se reportan media y desvío.
La unidad de agrupación para evitar el data leakage es el paciente, ya que las ventanas del mismo están correlacionadas

La CV se usa para comparar configuraciones, no como número final. La configuración ganadora se reentrena sobre todo pool de desarrollo y se evalúa una sola vez sobre el test
Por otro lado, el RobustScaler (mediana/IQR por canal) se recalcula en cada fold usando solo los pacientes de train de ese fold. Reusar el scaler global filtraría información de los pacientes de val hacia el escalado.

Flujo
-----
    build_folds        -> partición reproducible (folds.json)
    por config x fold  -> scaler del fold -> train_model -> eval del val del fold
    aggregate          -> media ± std por métrica
    select_best        -> configuración con mejor AUPRC medio (desempate: menor std)
    refit_final        -> entrena sobre todo el pool de desarrollo y evalúa UNA vez el test

    python -m src.cv --data-dir <ruta/chbmit> --folds 5 --seed 42 --out-dir results/cv
    python -m src.cv --configs bce,focal_a025 --limit-files 2 --epochs 2   # smoke-test local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.annotations import load_annotations, patient_seizure_stats
from src.config import (
    CV_CONFIGS,
    CV_FOLDS,
    CV_OUT_DIR,
    CV_SEED,
    DATASET_DIR,
    EPOCHS,
    FOCAL_GAMMA,
    SPLIT_FILE,
    THRESHOLD,
)
from src.evaluate import evaluate_checkpoint_on_split
from src.preprocessing import compute_stats_for_patients
from src.protocol import PROTOCOL_VERSION, split_id
from src.train import train_model

# Registro de configuraciones a comparar. El ÚNICO factor variable es la función de
# pérdida (BCE vs Focal con distinto alpha); todo lo demás (arquitectura, ventana,
# filtro, lr, batch, patience, neg_pos_ratio, noise_std, seed) queda fijo e idéntico
# entre configuraciones, porque cualquier diferencia adicional invalidaría la
# comparación (comparar de a un factor por vez).
CONFIG_REGISTRY: dict[str, dict] = {
    "bce": {
        "loss": "bce",
        "focal_alpha": None,
        "focal_gamma": None,
    },
    "focal_a025": {
        "loss": "focal",
        "focal_alpha": 0.25,
        "focal_gamma": FOCAL_GAMMA,
    },
    "focal_a075": {
        "loss": "focal",
        "focal_alpha": 0.75,
        "focal_gamma": FOCAL_GAMMA,
    },
}

# Métricas que se agregan con media ± std a través de los folds.
_METRIC_KEYS = [
    "auc_roc",
    "auprc",
    "sens_window",
    "spec_window",
    "event_sensibility",
    "event_fa_per_hour",
    "event_median_fa_per_hour",
    "latency_mean",
    "threshold",
    "best_epoch",
]


def _assign_severity_buckets(zsec: np.ndarray, K: int) -> np.ndarray:
    """StratifiedGroupKFold exige que cada clase tenga al menos K miembros. Se prueba primero con 3 grupos de severidad (con cuantiles) y, si
    algún cubo queda muy chico, con la mediana. Si ni así alcanza, se devuelve una única clase (agrupación simple).
    """
    zsec = np.asarray(zsec, dtype=np.float64)
    for n_buckets in (3, 2): #pruebo primero 3 grupos y sino 2
        #acá calculo los cuantiles y asigno a cada paciente un bucket de severidad (0,1,2) según la cantidad de segundos de crisis
        edges = np.unique(np.quantile(zsec, np.linspace(0.0, 1.0, n_buckets + 1)))
        #asigno a cada paciente una eriqueta de grupo
        labels = np.digitize(zsec, edges[1:-1])
        #chequeo q se hayan formado al menos 2 grupos distintos, y que el grupo más chico tenga al menos K pacientes.
        if len(np.unique(labels)) >= 2 and min((labels == b).sum() for b in np.unique(labels)) >= K:
            #si pasa, devuelvo las etiquetas. Sino, pruebo con 2 buckets
            return labels
    # Sin estratificación viable: una sola clase (equivalente a GroupKFold).
    return np.zeros(len(zsec), dtype=int)


def build_folds(dev_patients: list[str], stats: list[dict], K: int = CV_FOLDS, seed: int = CV_SEED,) -> list[dict]:
    """Particiona el pool de desarrollo en K folds agrupados por paciente.
        Todos los pacientes se particionan con StratifiedGroupKFold, estratificadopor cubos de severidad (segundos de crisis)
        Ningún paciente aparece en train y val del mismo fold, y cada paciente cae en val exactamente una vez; K folds; determinista dado `seed`.

    Devuelve [{"fold": i, "train": [...], "val": [...]}, ...].
    """
    if K <= 1:
        raise ValueError("K (folds) debe ser mayor que 1.")
    if not dev_patients:
        raise ValueError("El pool de desarrollo no contiene pacientes.")
    if K > len(dev_patients):
        raise ValueError(f"K={K} no puede ser mayor que los pacientes de desarrollo ({len(dev_patients)}).")

    severity = {s["patient"]: s["seizure_seconds"] for s in stats}
    patients_sorted = sorted(dev_patients, key=lambda p: (severity.get(p, 0.0), p))
    zsec = np.array([severity.get(p, 0.0) for p in patients_sorted], dtype=np.float64)

    # Estratificación por severidad (con fallback a agrupación simple).
    labels = _assign_severity_buckets(zsec, K)

    folds: list[dict] = []
    for i, (train_idx, val_idx) in enumerate(StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed).split(patients_sorted, labels, groups=patients_sorted)):
        train_patients = sorted(patients_sorted[j] for j in train_idx)
        val_patients = sorted(patients_sorted[j] for j in val_idx)
        folds.append({"fold": i, "train": train_patients, "val": val_patients})

    validate_folds(folds, dev_patients)
    return folds


def validate_folds(folds: list[dict], dev_patients: list[str]) -> None:
    dev = set(dev_patients)

    for fold in folds:
        train_s, val_s = set(fold["train"]), set(fold["val"])
        if train_s & val_s:
            raise ValueError(f"Fold {fold['fold']}: pacientes en train y val a la vez.")
        if train_s | val_s != dev:
            raise ValueError(f"Fold {fold['fold']}: la partición no cubre exactamente el pool.")
      
    val_counts: dict[str, int] = {}
    for fold in folds:
        for p in fold["val"]:
            val_counts[p] = val_counts.get(p, 0) + 1
    if any(c != 1 for c in val_counts.values()) or set(val_counts) != dev:
        raise ValueError("Cada paciente debe caer en val exactamente una vez.")


def build_fold_split(base_split: dict, fold_train: list[str], fold_val: list[str]) -> dict:
    #seteo oficialmente los pacientes de cada grupo del fold y hago la metadata
    fold_train_set = set(fold_train)
    fold_val_set = set(fold_val)
    test_patients = set(base_split.get("test", []))
    return {
        "protocol_version": base_split.get("protocol_version", PROTOCOL_VERSION),
        "train": list(fold_train),
        "val": list(fold_val),
        "test": list(base_split.get("test", [])),
        "test_ratio_solicitado": base_split.get("test_ratio_solicitado"),
        "n_val_patients_config": len(fold_val),
        "val_required_patients": sorted(
            set(base_split.get("val_required_patients", [])) & fold_val_set
        ),
        "forced_train_patients": sorted(
            set(base_split.get("forced_train_patients", [])) & fold_train_set
        ),
        "forced_test_patients": sorted(
            set(base_split.get("forced_test_patients", [])) & test_patients
        ),
        "split_w_zsec": base_split.get("split_w_zsec"),
    }


def validate_pools(dev_patients: list[str], test_patients: list[str]) -> None:
    """Chequea que el test  esté separado del pool de desarrollo."""
    if set(dev_patients) & set(test_patients):
        raise ValueError("El test no puede compartir pacientes con el pool de desarrollo.")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_fold(config_id: str, config: dict, fold: dict,fold_split: dict, *, data_dir: Path | str, out_dir: Path,
    device: str | None = None, seed: int = CV_SEED, epochs: int = EPOCHS, limit_files: int | None = None,) -> dict:
    """Ejecuta un fold completo y devuelve sus métricas, que aggregate después juntaen media ± std.
    """
    fold_idx = fold["fold"]
    run_dir = out_dir / config_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / f"fold{fold_idx}.pt"

    # mediana/IQR calculadas SOLO con el train de este fold.
    scaler_stats = compute_stats_for_patients(data_dir, fold["train"], split=fold_split, limit=limit_files, verbose=False)

    # mismo criterio de selección en el entrenamiento que el flujo normal.
    train_summary = train_model(data_dir=data_dir, split=fold_split, scaler_stats=scaler_stats, out=ckpt,
        device=device, loss=config["loss"], focal_alpha=config["focal_alpha"], focal_gamma=config["focal_gamma"],
        epochs=epochs, seed=seed, limit_files=limit_files, resume=True,)

    #vuelvo a cargar el checkpoint elegido y calculo las métricas finales del fold sobre su validación
    eval_res = evaluate_checkpoint_on_split(ckpt, data_dir=data_dir, split=fold_split, split_name="val", device=device,
    limit_files=limit_files, reselect_operating_point=False,)

    ev = eval_res["event"]
    return {
        "config": config_id,
        "fold": fold_idx,
        "train_patients": fold["train"],
        "val_patients": fold["val"],
        "auc_roc": eval_res["ranking"]["auc_roc"],
        "auprc": eval_res["ranking"]["auprc"],
        "sens_window": eval_res["window"]["sensibility"],
        "spec_window": eval_res["window"]["specificity"],
        "threshold": eval_res["threshold"],
        "event_sensibility": ev["sensibility"],
        "event_n_detected": ev["n_detected"],
        "event_n_seizures": ev["n_seizures"],
        "event_fa_per_hour": ev["event_false_alarms_per_hour"],
        "event_median_fa_per_hour": ev["median_false_alarms_per_hour"],
        "latency_mean": ev["latency_mean"],
        "best_epoch": train_summary["best_epoch"],
        "best_val_loss": train_summary["best_val_loss"],
    }


def aggregate(fold_results: list[dict], metric_keys: list[str] | None = None) -> dict:
    """Agrega las métricas de los folds en media ± std (más min/max y valores).

    Para la sensibilidad de evento además se reporta el "pooled" (detectadas/totales sumando a través de folds), que pondera por cantidad de crisis y no por fold, y
    siempre con los conteos (la sensibilidad con pocas crisis tiene mucha varianza).
    """
    keys = metric_keys or _METRIC_KEYS
    agg: dict = {}
    for key in keys:
        vals = [r[key] for r in fold_results if r.get(key) is not None]
        if not vals:
            agg[key] = None
            continue
        q25, q75 = np.percentile(vals, [25, 75])
        agg[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(q75 - q25),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": [float(v) for v in vals],
        }

    detected = int(sum(r["event_n_detected"] for r in fold_results))
    seizures = int(sum(r["event_n_seizures"] for r in fold_results))
    agg["event_sensibility_pooled"] = (detected / seizures) if seizures else None
    agg["event_counts"] = {"detected": detected, "seizures": seizures}
    return agg


def select_best(summary: dict, metric: str = "auprc") -> str:
    """Elige la configuración con mejor mediana de metric entre los folds de esa config.
    Desempate: menor rango intercuartílico (IQR). Nunca se elige por el mejor fold individual. devuelve el nombre de la config q ganó
    """
    scored = [
        (config_id, m[metric]["median"], m[metric].get("iqr", float("inf")))
        for config_id, m in summary.items()
        if m.get(metric) and m[metric].get("median") is not None
    ]
    if not scored:
        raise RuntimeError(
            f"No hay {metric} en ningún fold para seleccionar la configuración ganadora. "
            "Verificar que el val de cada fold tenga ventanas positivas (un smoke-test con "
            "--limit-files muy chico no las tiene)."
        )
    return max(scored, key=lambda t: (t[1], -t[2]))[0]


def refit_final(config: dict, dev_patients: list[str] ,base_split: dict, *, data_dir: Path | str, out: Path, device: str | None = None, seed: int = CV_SEED,
    epochs: int = EPOCHS, refit_threshold: float = THRESHOLD,) -> dict:
    #Refit de la configuración ganadora sobre todo el pool de desarrollo
    final_split = build_fold_split(base_split, dev_patients, [])

    scaler_stats = compute_stats_for_patients(
        data_dir, dev_patients, split=final_split, verbose=False,
    )

    train_summary = train_model(
        data_dir=data_dir,
        split=final_split,
        scaler_stats=scaler_stats,
        out=out,
        device=device,
        loss=config["loss"],
        focal_alpha=config["focal_alpha"],
        focal_gamma=config["focal_gamma"],
        epochs=epochs,
        seed=seed,
        fixed_op_threshold=refit_threshold,
        resume=True,
    )

    test_res = evaluate_checkpoint_on_split(
        out,
        data_dir=data_dir,
        split=final_split,
        split_name="test",
        device=device,
    )
    return {"train_summary": train_summary, "test": test_res}


def run_refit(config_id: str, *, data_dir: Path | str, split_file: Path | str = SPLIT_FILE, out_dir: Path | str = CV_OUT_DIR,
    device: str | None = None, seed: int = CV_SEED, epochs: int | None = None, refit_threshold: float | None = None,) -> dict:
    """Refit de una configuración sobre todo el pool de desarrollo + evaluación de test, sin reentrenar los folds.

    El umbral y la cantidad de épocas se infieren de los `fold*.result.json` ya guardados de esa configuración
    (media de umbrales y de best_epoch, igual que en `run_cv`); si se pasan `refit_threshold`/`epochs` explícitos, tienen prioridad.
    Escribe `<out_dir>/cv_final_<config>.pt` y `<out_dir>/refit_result.json`.
    """
    if config_id not in CONFIG_REGISTRY:
        raise ValueError(f"Configuración desconocida: {config_id!r}")

    base_split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    dev_patients = list(base_split.get("train", [])) + list(base_split.get("val", []))
    test_patients = list(base_split.get("test", []))
    validate_pools(dev_patients, test_patients)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict] = []
    for path in sorted((out_dir / config_id).glob("fold*.result.json")):
        fold_results.append(json.loads(path.read_text(encoding="utf-8")))

    if refit_threshold is None:
        thresholds = [r["threshold"] for r in fold_results if r.get("threshold") is not None]
        refit_threshold = round(float(np.mean(thresholds)), 2) if thresholds else THRESHOLD
    if epochs is None:
        best_epochs = [r["best_epoch"] for r in fold_results if r.get("best_epoch")]
        epochs = max(1, int(round(np.mean(best_epochs)))) if best_epochs else EPOCHS

    refit_out = out_dir / f"cv_final_{config_id}.pt"
    print(f"\nRefit de {config_id} sobre los {len(dev_patients)} pacientes del pool "
          f"({epochs} épocas, umbral={refit_threshold})")
    t0 = time.time()
    refit = refit_final(
        CONFIG_REGISTRY[config_id], dev_patients, base_split,
        data_dir=data_dir, out=refit_out, device=device,
        seed=seed, epochs=epochs, refit_threshold=refit_threshold,
    )
    refit_result = {
        "checkpoint": str(refit_out),
        "epochs": epochs,
        "threshold": refit_threshold,
        "test": refit["test"],
    }
    _write_json_atomic(out_dir / "refit_result.json", refit_result)
    refit_out.with_name(refit_out.stem + ".resume.pt").unlink(missing_ok=True)
    print(f"Refit terminado en {time.time() - t0:.1f}s -> {refit_out}")

    return {"summary": {}, "best": config_id, "refit_threshold": refit_threshold, "refit": refit_result}


def run_select_best(*, out_dir: Path | str = CV_OUT_DIR, config_ids: list[str] | None = None) -> dict:
    """Lee los `fold*.result.json` ya guardados, agrega y selecciona la configuración ganadora (sin reentrenar nada)."""
    config_ids = list(config_ids if config_ids is not None else CV_CONFIGS)
    out_dir = Path(out_dir)

    per_config: dict[str, list[dict]] = {}
    for cid in config_ids:
        results = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted((out_dir / cid).glob("fold*.result.json"))]
        if results:
            per_config[cid] = results

    if not per_config:
        raise RuntimeError(f"No hay fold*.result.json en {out_dir} para ninguna configuración.")

    summary = {cid: aggregate(per_config[cid]) for cid in per_config}
    best = select_best(summary, metric="auprc")
    return {"summary": summary, "best": best}


def run_cv(*, data_dir: Path | str, split_file: Path | str = SPLIT_FILE, out_dir: Path | str = CV_OUT_DIR,
    K: int = CV_FOLDS, seed: int = CV_SEED, config_ids: list[str] | None = None, device: str | None = None,
    epochs: int = EPOCHS, refit_epochs: int | None = None, limit_files: int | None = None, keep_checkpoints: bool = True,
    skip_refit: bool = False,) -> dict:
    """Orquesta la CV completa, devuelve el resumen

    Salidas en `out_dir`:
      - folds.json        : partición reproducible (pacientes por fold + seed).
      - summary_cv.json   : media ± std por configuración y métrica.
      - best_config.json  : configuración ganadora + regla + refit threshold.
      - <config>/fold<i>.history.json : historial de entrenamiento por fold.
      - <config>/fold<i>.pt           : checkpoint de mejor época por fold.
      - <config>/fold<i>.resume.pt    : estado por época para reanudar tras interrupciones.
      - <config>/fold<i>.result.json  : métricas persistidas al completar el fold.
    """
    base_split = json.loads(Path(split_file).read_text(encoding="utf-8"))
    test_patients = list(base_split.get("test", []))
    dev_patients = list(base_split.get("train", [])) + list(base_split.get("val", []))
    config_ids = list(config_ids if config_ids is not None else CV_CONFIGS)

    # Chequeos previos
    validate_pools(dev_patients, test_patients)
    for cid in config_ids:
        if cid not in CONFIG_REGISTRY:
            raise ValueError(f"Configuración desconocida: {cid!r}")

    # Severidad por paciente (para estratificar los folds y que queden equilibrados la cantidad de casos graves y leves)
    annotations = load_annotations(data_dir)
    stats = patient_seizure_stats(annotations)
    known = {s["patient"] for s in stats}
    if not set(dev_patients + test_patients) <= known:
        raise RuntimeError("Hay pacientes del split sin anotaciones en el dataset.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = build_folds(dev_patients, stats, K=K, seed=seed)
    fold_plan = {"seed": seed, "K": K, "dev_patients": dev_patients,"test_patients": test_patients, "folds": folds,}
    manifest = {
        "format_version": 3,
        "data_dir": str(Path(data_dir).resolve()),
        "split_id": split_id(base_split),
        "split_file_sha256": _sha256_file(Path(split_file)),
        "annotations_sha256": _sha256_json(annotations),
        "fold_plan_sha256": _sha256_json(fold_plan),
        "protocol_version": PROTOCOL_VERSION,
        "folds": K,
        "seed": seed,
        "configs": config_ids,
        "device": device,
        "epochs": epochs,
        "refit_epochs": refit_epochs,
        "limit_files": limit_files,
        "keep_checkpoints": keep_checkpoints,
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parent.parent)): _sha256_file(path)
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
    }
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise RuntimeError(
                f"La carpeta {out_dir} ya pertenece a otra configuración de CV. "
                "Elegí un --out-dir nuevo para iniciar una corrida distinta."
            )
    else:
        _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(out_dir / "folds.json", fold_plan)

    # guardo los resultados de cada configuración, agrupados en listas. X ej per_config["bce"] = [fold0_result, fold1_result, ...]
    per_config: dict[str, list[dict]] = {}
    for cid in config_ids:
        per_config[cid] = [] #seteo la congif
        for fold in folds: #para esa config, recorro cada fold
            run_dir = out_dir / cid
            fold_result_path = run_dir / f"fold{fold['fold']}.result.json"
            #El archivo existe? si sí, puede ser q el fold se haya completado en una ejecucion anterior, así q lo leo y verifico q los datos estén íntegros
            #si sí, lo reutilizo y paso al próximo fold
            if fold_result_path.is_file():
                result = json.loads(fold_result_path.read_text(encoding="utf-8"))
                if (result.get("config") != cid or result.get("fold") != fold["fold"] or result.get("train_patients") != fold["train"] or result.get("val_patients") != fold["val"]):
                    raise RuntimeError(f"Resultado de fold inválido: {fold_result_path}")
                print(f"\n[{cid}] fold {fold['fold']}: ya completo, se reutilizan sus resultados.")
                per_config[cid].append(result)
                (run_dir / f"fold{fold['fold']}.resume.pt").unlink(missing_ok=True)
                continue
            #si no existe, corro el fold completo (scaler -> train -> eval) y guardo el resultado
            fold_split = build_fold_split(base_split, fold["train"], fold["val"])
            print(f"\n[{cid}] fold {fold['fold']}: train={len(fold['train'])} pacientes, " f"val={len(fold['val'])}")
            
            result = run_fold(cid, CONFIG_REGISTRY[cid], fold, fold_split, data_dir=data_dir, out_dir=out_dir, device=device, seed=seed, epochs=epochs, limit_files=limit_files)
            per_config[cid].append(result)
            #acá guardo el resultado del fold en un json, y borro el checkpoint de resume (ya no hace falta)
            _write_json_atomic(fold_result_path, result)
            (run_dir / f"fold{fold['fold']}.resume.pt").unlink(missing_ok=True)
            if not keep_checkpoints:
                (run_dir / f"fold{fold['fold']}.pt").unlink(missing_ok=True)
    #para cada configuracion cid, aggregate junta los resultados de todos sus folds y devuelve media ± std por métrica. queda ub diccionario con un resumen por config
    summary = {cid: aggregate(per_config[cid]) for cid in config_ids}
    best = select_best(summary, metric="auprc")

    # el threshold del refit es la media de los umbrales seleccionados en los folds de la configuración ganadora
    fold_thresholds = [r["threshold"] for r in per_config[best] if r.get("threshold") is not None]
    refit_threshold = round(float(np.mean(fold_thresholds)), 2) if fold_thresholds else THRESHOLD

    result: dict = {"summary": summary, "best": best, "refit_threshold": refit_threshold}

    if not skip_refit:
        # la cantidad de epochs del refit es la media de las épocas seleccionadas en los folds de la ganadora (sin early stopping porque ya no hay validación)
        if refit_epochs is None:
            best_epochs = [r["best_epoch"] for r in per_config[best] if r.get("best_epoch")]
            refit_epochs = max(1, int(round(np.mean(best_epochs)))) if best_epochs else epochs

        refit_out = out_dir / f"cv_final_{best}.pt"
        refit_result_path = out_dir / "refit_result.json"
        if refit_result_path.is_file():
            refit_result = json.loads(refit_result_path.read_text(encoding="utf-8"))
            print("Refit/test ya completados; se reutilizan los resultados guardados.")
        else:
            print(f"\nRefit de {best} sobre los {len(dev_patients)} pacientes del pool "
                  f"({refit_epochs} épocas, umbral={refit_threshold})")
            t0 = time.time()
            refit = refit_final(
                CONFIG_REGISTRY[best], dev_patients, base_split,
                data_dir=data_dir, out=refit_out, device=device,
                seed=seed, epochs=refit_epochs, refit_threshold=refit_threshold,
            )
            refit_result = {
                "checkpoint": str(refit_out),
                "epochs": refit_epochs,
                "threshold": refit_threshold,
                "test": refit["test"],
            }
            _write_json_atomic(refit_result_path, refit_result)
            refit_out.with_name(refit_out.stem + ".resume.pt").unlink(missing_ok=True)
            print(f"Refit terminado en {time.time() - t0:.1f}s -> {refit_out}")
        result["refit"] = refit_result

    _write_json_atomic(out_dir / "summary_cv.json", {
        "criterion": "mediana de AUPRC por fold (desempate: menor IQR)",
        "seed": seed,
        "K": K,
        "configs": summary,
        "best": best,
    })
    _write_json_atomic(out_dir / "best_config.json", {
        "best_config": best,
        "criterion": "mediana de AUPRC por fold (desempate: menor IQR)",
        "refit_threshold": refit_threshold,
        "refit_checkpoint": result.get("refit", {}).get("checkpoint"),
        "config": CONFIG_REGISTRY[best],
    })

    return result


def print_cv_summary(result: dict) -> None:
    print("\n" + "=" * 78)
    print("VALIDACIÓN CRUZADA — RESUMEN")
    print("=" * 78)
    print(f"Mejor configuración (AUPRC mediana, desempate menor IQR): {result['best']}")

    header = (f"{'config':<14}{'AUPRC':>22}{'AUC-ROC':>22}"
              f"{'sens. evento':>16}{'FA/h med':>18}")
    print(header)
    print("-" * len(header))

    def fmt(agg: dict | None, key: str, decimals: int = 3) -> str:
        v = (agg or {}).get(key)
        if not v or v.get("median") is None:
            return "—"
        med = f"{v['median']:.{decimals}f}"
        if v.get("q25") is None or v.get("q75") is None:
            return med
        return f"{med} ({v['q25']:.{decimals}f}–{v['q75']:.{decimals}f})"

    for cid, agg in result["summary"].items():
        pooled = agg.get("event_sensibility_pooled")
        counts = agg.get("event_counts", {})
        sens = (f"{pooled * 100:.1f}%"
                f" ({counts.get('detected', 0)}/{counts.get('seizures', 0)})"
                if pooled is not None else "—")
        print(f"{cid:<14}{fmt(agg, 'auprc'):>22}{fmt(agg, 'auc_roc'):>22}"
              f"{sens:>16}{fmt(agg, 'event_median_fa_per_hour', 2):>18}")

    refit = result.get("refit")
    if refit:
        print("-" * len(header))
        _print_refit_block(refit, result["best"])
    print("=" * 78)


def _print_refit_block(refit: dict, config_id: str) -> None:
    """Imprime las métricas de test de un refit (checkpoint reentrenado sobre el pool de desarrollo)."""
    test_ev = refit["test"]["event"]
    test_rk = refit["test"]["ranking"]
    auc = f"{test_rk['auc_roc']:.4f}" if test_rk.get("auc_roc") is not None else "—"
    auprc = f"{test_rk['auprc']:.4f}" if test_rk.get("auprc") is not None else "—"
    print(f"Refit ({config_id}) reentrenado sobre todo el pool de desarrollo "
          f"y evaluado una vez sobre test:")
    print(f"  checkpoint: {refit['checkpoint']}  |  umbral: {refit['threshold']}")
    print(f"  AUC-ROC={auc}  AUPRC={auprc}  "
          f"sens. evento={test_ev['sensibility']:.4f} "
          f"({test_ev['n_detected']}/{test_ev['n_seizures']})  "
          f"FA/h mediana={test_ev['median_false_alarms_per_hour']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validación cruzada agrupada por paciente.")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--out-dir", type=str, default=str(CV_OUT_DIR))
    parser.add_argument("--folds", type=int, default=CV_FOLDS, help="K de la CV.")
    parser.add_argument("--seed", type=int, default=CV_SEED)
    parser.add_argument("--configs", type=str, default=",".join(CV_CONFIGS),
                        help="Configuraciones separadas por coma (bce, focal_a025, focal_a075).")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Épocas máx por fold.")
    parser.add_argument("--refit-epochs", type=int, default=None,
                        help="Épocas del refit final (default: media de las épocas por fold).")
    parser.add_argument("--refit-only", type=str, default=None, metavar="CONFIG",
                        help="Solo reentrenar CONFIG sobre todo el pool de desarrollo y evaluar test, sin correr folds.")
    parser.add_argument("--refit-threshold", type=float, default=None,
                        help="Umbral del refit (default: media de los umbrales de los folds de esa config).")
    parser.add_argument("--select-best-only", action="store_true",
                        help="Solo agregar los folds ya guardados y elegir la config ganadora, sin entrenar nada.")
    parser.add_argument("--device", type=str, default=None, help="'cuda' o 'cpu'.")
    parser.add_argument("--limit-files", type=int, default=None,
                        help="Limitar a N EDFs por split (smoke-test; NUNCA para resultados).")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--keep-checkpoints", dest="keep_checkpoints", action="store_true")
    checkpoint_group.add_argument("--discard-checkpoints", dest="keep_checkpoints", action="store_false",
                                  help="Borrar los .pt de folds ya completados.")
    parser.set_defaults(keep_checkpoints=True)
    parser.add_argument("--skip-refit", action="store_true",
                        help="Saltear el refit final y la evaluación de test.")
    args = parser.parse_args()
    if args.folds <= 1:
        parser.error("--folds debe ser mayor que 1")
    if args.epochs <= 0:
        parser.error("--epochs debe ser mayor que cero")
    if args.limit_files is not None and args.limit_files <= 0:
        parser.error("--limit-files debe ser mayor que cero")
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.select_best_only:
            result = run_select_best(out_dir=args.out_dir,
                config_ids=[c.strip() for c in args.configs.split(",") if c.strip()])
            print_cv_summary(result)
            return
        if args.refit_only:
            result = run_refit(args.refit_only, data_dir=args.data_dir, split_file=args.split_file, out_dir=args.out_dir,
                device=args.device, seed=args.seed, epochs=args.refit_epochs, refit_threshold=args.refit_threshold)
            print_cv_summary(result)
            return
        result = run_cv(data_dir=args.data_dir, split_file=args.split_file, out_dir=args.out_dir,
            K=args.folds, seed=args.seed, config_ids=[c.strip() for c in args.configs.split(",") if c.strip()],
            device=args.device, epochs=args.epochs, refit_epochs=args.refit_epochs, limit_files=args.limit_files,
            keep_checkpoints=args.keep_checkpoints,
            skip_refit=args.skip_refit,
        )
    except (RuntimeError, ValueError) as exc:
        sys.exit(str(exc))
    print_cv_summary(result)


if __name__ == "__main__":
    main()
