"""
Evaluación de un checkpoint guardado sobre val o test.

    python -m src.evaluate --checkpoint models/best.pt --split test
    python -m src.evaluate --checkpoint models/best.pt --split val

Carga un checkpoint que guardó train.py, reconstruye la CNN, arma el DataLoader del split pedido sin balancear y sin pos_weight y calcula las métricas clínicas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.config import (
    BATCH_SIZE,
    DATASET_DIR,
    EVENT_MAX_LATENCY,
    EVENT_MIN_ALARM_INTERVAL,
    WINDOW_RANGE_FOR_EVENT,
    POSITIVES_FOR_EVENT,
    MAX_FALSE_ALARMS_PER_HOUR,
    MODELS_DIR,
    NUM_WORKERS,
    SPLIT_FILE,
    THRESHOLD,
)
from src.data import build_splits_dataloaders
from src.metrics import (
    binary_metrics,
    event_metrics,
    rank_metrics,
    select_event_operating_point,
    select_lowest_false_alarm_point,
)
from src.model import SeizureCNN
from src.train import evaluate, get_device
from src.protocol import (
    THRESHOLD_GRID,
    split_id,
    validate_checkpoint_scaler,
)


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Checkpoint no encontrado: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"El checkpoint '{path}' no tiene el formato seguro esperado. "
            "Regeneralo ejecutando el flujo de validación cruzada con src.cv."
        ) from exc

    # Normaliza las stats del scaler a NumPy (apply_scaler opera sobre NumPy)
    scaler_stats = checkpoint.get("scaler_stats")
    if scaler_stats is not None:
        for key in ("median", "iqr"):
            value = scaler_stats.get(key)
            if torch.is_tensor(value):
                scaler_stats[key] = value.numpy()

    return checkpoint


def rebuild_model(checkpoint: dict) -> SeizureCNN:
    config = checkpoint["model_config"]
    model = SeizureCNN(in_channels=config["in_channels"], conv_channels=config["conv_channels"], conv_kernels=config["conv_kernels"], fc_units=config["fc_units"], dropout=config["dropout"],)
    # Carga los valores de TODOS los pesos guardados en el checkpoint.
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluar un checkpoint sobre val o test")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best.pt"), help="Ruta del .pt a evaluar.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],   help="Qué split evaluar")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR),  help="Carpeta con los EDFs crudos de CHB-MIT.")
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE),  help="Ruta del split.json.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--threshold", type=float, default=None, help="Umbral de decisión. Por defecto usa el que guardó el checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="'cuda' o 'cpu'.")
    parser.add_argument("--limit-files", type=int, default=None, help="Limitar a N EDFs del split (smoke-test rápido).")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size debe ser mayor que cero")
    if args.num_workers < 0:
        parser.error("--num-workers no puede ser negativo")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold debe estar entre 0 y 1")
    if args.limit_files is not None and args.limit_files <= 0:
        parser.error("--limit-files debe ser mayor que cero")
    return args


def evaluate_checkpoint_on_split(checkpoint_path: Path | str, *, data_dir: Path | str, split: dict, split_name: str = "val",
    device: str | None = None, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, threshold: float | None = None, limit_files: int | None = None,
    reselect_operating_point: bool = True,) -> dict:
    """Mide un checkpoint sobre un split dado, reutilizando el scaler embebido.
    Devuelve un dict con ranking, ventana y evento + la metadata del checkpoint.
    """
    device = get_device(device)

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    model = rebuild_model(checkpoint)
    model.to(device)
    model.eval()  # modo evaluar (chau dropout y eso)

    # el event config guarda n_within y n_window (tiro alarma si en n_window ventanas hay n_within positivas), tamb guarda min alarm interval, max latency, y el techo de falsas alarmas
    event_config = checkpoint.get("event_config", {})
    checkpoint_threshold_grid = [float(t) for t in checkpoint.get("threshold_grid", THRESHOLD_GRID)]
    if not checkpoint_threshold_grid or any(not 0.0 <= t <= 1.0 for t in checkpoint_threshold_grid):
        raise RuntimeError("threshold_grid inválida en el checkpoint.")

    max_false_alarms_per_hour = event_config.get("max_false_alarms_per_hour", MAX_FALSE_ALARMS_PER_HOUR)
    positives_for_event = event_config.get("n_within", POSITIVES_FOR_EVENT)
    window_range_for_event = event_config.get("n_window", WINDOW_RANGE_FOR_EVENT)
    event_min_alarm_interval = event_config.get("min_alarm_interval", EVENT_MIN_ALARM_INTERVAL)
    event_max_latency = event_config.get("max_latency", EVENT_MAX_LATENCY)
    if max_false_alarms_per_hour <= 0:
        raise RuntimeError("max_false_alarms_per_hour debe ser mayor que cero")
    if positives_for_event <= 0 or window_range_for_event <= 0 or positives_for_event > window_range_for_event:
        raise RuntimeError("Configuración de ventanas de evento inválida")
    if event_min_alarm_interval < 0 or event_max_latency < 0:
        raise RuntimeError("Los intervalos de evento no pueden ser negativos")

    thr = threshold if threshold is not None else checkpoint.get("op_threshold")
    if thr is None:
        if split_name == "test":
            raise RuntimeError("El checkpoint no contiene op_threshold seleccionado en validación.")
        thr = checkpoint.get("threshold", THRESHOLD)
#tests de consistencia
    checkpoint_split_id = checkpoint.get("split_id")
    current_split_id = split_id(split)
    if checkpoint_split_id != current_split_id:
        raise RuntimeError(
            f"El checkpoint y el split no coinciden: checkpoint={checkpoint_split_id}, split={current_split_id}."
        )
    # El scaler_stats viene adentro del checkpoint y lo uso para escalar las ventanas de val/test EXACTAMENTE igual que en train
    scaler_stats = checkpoint.get("scaler_stats")
    if scaler_stats is None:
        raise RuntimeError("El checkpoint no contiene estadísticas del scaler.")
    validate_checkpoint_scaler(checkpoint, scaler_stats)
    if scaler_stats["split_id"] != current_split_id:
        raise RuntimeError("El scaler embebido en el checkpoint no corresponde al split solicitado.")

    # armo train/val/test. Para val/test recorre TODAS las ventanas en orden (sin undersampling, sin shuffle)
    # Construye datasets con el scaler embebido en el checkpoint.
    loaders = build_splits_dataloaders(data_dir, split, scaler_stats,
                                       batch_size=batch_size, num_workers=num_workers,
                                       limit_files=limit_files)
    dataset = loaders[split_name]["dataset"]
    loader = loaders[split_name]["dataloader"]

    n_pos = int(dataset.n_positive)
    n_total = len(dataset)

    # evaluar
    # mido el error sobre la distribución natural, x ende el Binary Cross entropy va sin pos-weight
    criterion = nn.BCEWithLogitsLoss()
    avg_loss, probs, labels = evaluate(model, loader, criterion, device)

    # Métricas clínicas sobre TODAS las ventanas del split
    #Construyo el índice global ventana -> (archivo, ventana local) y el vector de label
    dataset._ensure_index()
    m = binary_metrics(probs, labels, thr, file_ids=dataset.file_ids, local_ids=dataset.local_ids)
    rk = rank_metrics(probs, labels)

    event_selection = "checkpoint_threshold"
    event_warning: str | None = None
    if split_name == "test" or not reselect_operating_point:
        # En test y en evaluaciones que piden umbral fijo, no se recalibra.
        ev = event_metrics(
            probs, labels,
            file_ids=dataset.file_ids, local_ids=dataset.local_ids,
            valid_files=dataset.valid_files, annotations=dataset.annotations,
            threshold=thr,
            n_within=positives_for_event, n_window=window_range_for_event,
            min_alarm_interval=event_min_alarm_interval, max_latency=event_max_latency,
        )
    else:
        ev, ev_threshold = select_event_operating_point(
            probs, labels,
            file_ids=dataset.file_ids, local_ids=dataset.local_ids,
            valid_files=dataset.valid_files, annotations=dataset.annotations,
            max_false_alarms_per_hour=max_false_alarms_per_hour,
            n_within=positives_for_event, n_window=window_range_for_event,
            min_alarm_interval=event_min_alarm_interval, max_latency=event_max_latency,
            thresholds=checkpoint_threshold_grid,
        )
        event_selection = "operating_point"
        if ev is None:
            # Ningún umbral respeta el techo: fallback explícito al de menor mediana.
            ev, ev_threshold = select_lowest_false_alarm_point(
                probs, labels,
                file_ids=dataset.file_ids, local_ids=dataset.local_ids,
                valid_files=dataset.valid_files, annotations=dataset.annotations,
                n_within=positives_for_event, n_window=window_range_for_event,
                min_alarm_interval=event_min_alarm_interval, max_latency=event_max_latency,
                thresholds=checkpoint_threshold_grid,
            )
            event_selection = "fallback"
            event_warning = (f"Ningún umbral respetó el techo de {max_false_alarms_per_hour:.0f} "
                             f"falsas alarmas/h. Mostrando el de menor mediana (umbral {ev_threshold:.2f}).")
        thr = ev_threshold

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "split": split_name,
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "pos_weight": checkpoint.get("pos_weight"),
        "seed": checkpoint.get("seed"),
        "threshold": float(thr),
        "event_selection": event_selection,
        "event_warning": event_warning,
        "event_config": {
            "n_within": positives_for_event,
            "n_window": window_range_for_event,
            "min_alarm_interval": event_min_alarm_interval,
            "max_latency": event_max_latency,
            "max_false_alarms_per_hour": max_false_alarms_per_hour,
        },
        "n_windows": n_total,
        "n_positive": n_pos,
        "loss": avg_loss,
        "ranking": rk,
        "window": m,
        "event": ev,
    }


def print_evaluation_report(result: dict) -> None:
    """Imprime el reporte de `evaluate_checkpoint_on_split` (salida histórica del CLI)."""
    ec = result["event_config"]
    m = result["window"]
    rk = result["ranking"]
    ev = result["event"]
    thr = result["threshold"]
    split_name = result["split"]

    print("=" * 70)
    print("EVALUACIÓN DE CHECKPOINT")
    print("=" * 70)
    print(f"  Checkpoint:      {result['checkpoint_name']}")
    print(f"  Época guardada:  {result['epoch'] if result['epoch'] is not None else '?'}")
    print(f"  best_val_loss:   {result['best_val_loss'] if result['best_val_loss'] is not None else float('nan'):.6f}")
    print(f"  pos_weight:      {result['pos_weight'] if result['pos_weight'] is not None else '?'}")
    print(f"  seed:            {result['seed'] if result['seed'] is not None else '?'}")
    print(f"  threshold:       {thr}")
    print(f"  split a evaluar: {split_name}")
    print(f"  Ventanas {split_name}: {result['n_windows']}  "
          f"({result['n_positive']} positivas, {result['n_windows'] - result['n_positive']} negativas)")

    print("-" * 70)
    print(f"  Loss promedio ({split_name}):  {result['loss']:.6f}")
    if rk["auc_roc"] is not None:
        print(f"  AUC-ROC:                        {rk['auc_roc']:.4f}")
    if rk["auprc"] is not None:
        print(f"  AUPRC (average precision):      {rk['auprc']:.4f}")
    print(f"  [NIVEL VENTANA @ umbral {thr:.2f}]")
    print(f"  Sensibilidad (recall):          {m['sensibility']:.4f}")
    print(f"  Especificidad:                  {m['specificity']:.4f}")
    print(f"  FPR (tasa falsos positivos):    {m['false_positive_rate']:.4f}")
    print(f"  Falsos positivos por hora:      {m['false_positive_per_hour']:.4f}")
    print(f"  Accuracy:                       {m['accuracy']:.4f}")
    print(f"  F1:                             {m['f1']:.4f}")
    print(f"  MCC:                            {m['mcc']:.4f}")
    print(f"  Conteos -> TP={m['actual_positive']}  FP={m['false_positive']}  "
          f"TN={m['actual_negative']}  FN={m['false_negative']}")

    if result["event_warning"]:
        print(f"\n  [WARN] {result['event_warning']}")
    if split_name == "test":
        print("\n  [NIVEL EVENTO]")
        print(f"  Threshold aplicado: {thr:.2f}")
    else:
        print("\n  [NIVEL EVENTO (crisis)]")
        if result["event_selection"] == "operating_point":
            print(f"  Punto de operación (mediana de falsas alarmas <= {ec['max_false_alarms_per_hour']:.1f}/h, mayor sensibilidad): "
                  f"umbral={thr:.2f}  mediana={ev['median_false_alarms_per_hour']:.2f}/h")

    print(f"  Sensibilidad:              {ev['sensibility']:.4f}  ({ev['n_detected']}/{ev['n_seizures']} crisis detectadas)")
    print(f"  Falsas alarmas de evento/h: {ev['event_false_alarms_per_hour']:.4f}  (mediana por paciente: {ev['median_false_alarms_per_hour']:.4f})  ({ev['n_false_alarms']} falsas alarmas en {ev['total_hours']:.1f} h cubiertas)")
    print(f"  Latencia media:            {ev['latency_mean']:.2f} s")
    print(f"  Latencia mediana:          {ev['latency_median']:.2f} s")
    print(f"  Postprocesado:             >= {ec['n_within']} positivos en {ec['n_window']} ventanas consecutivas")
    print("  Desglose por paciente:")
    for patient, patient_metrics in sorted(ev["by_patient"].items()):
        print(
            f"    {patient}: sens={patient_metrics['sensitivity']:.4f} "
            f"({patient_metrics['n_detected']}/{patient_metrics['n_seizures']}), "
            f"falsas alarmas/h={patient_metrics['false_alarms_per_hour']:.4f}"
        )
    print("=" * 70)


def main() -> None:
    args = parse_args()

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))

    try:
        result = evaluate_checkpoint_on_split(
            args.checkpoint,
            data_dir=args.data_dir,
            split=split,
            split_name=args.split,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            threshold=args.threshold,
            limit_files=args.limit_files,
        )
    except RuntimeError as exc:
        sys.exit(str(exc))

    print_evaluation_report(result)


if __name__ == "__main__":
    main()
