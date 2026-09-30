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
import numpy as np

from src.config import (
    BATCH_SIZE,
    DATASET_DIR,
    EVENT_MAX_LATENCY,
    EVENT_MIN_ALARM_INTERVAL,
    WINDOW_RANGE_FOR_EVENT,
    POSITIVES_FOR_EVENT,
    MIN_EVENT_SENSITIVITY,
    MODELS_DIR,
    NUM_WORKERS,
    SPLIT_FILE,
    THRESHOLD,
)
from src.data import build_splits_dataloaders
from src.metrics import (
    binary_metrics,
    event_metrics,
    select_event_operating_point,
)
from src.model import SeizureCNN
from src.train import evaluate, get_device


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Checkpoint no encontrado: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"El checkpoint '{path}' no tiene el formato seguro esperado. "
            "Regeneralo entrenando nuevamente con src.train."
        ) from exc

    if checkpoint.get("format_version") != 3:
        raise RuntimeError(
            f"Formato de checkpoint no soportado: {checkpoint.get('format_version')!r}. "
            "Regeneralo entrenando nuevamente con src.train."
        )

    scaler_stats = checkpoint.get("scaler_stats")
    if scaler_stats is not None:
        # apply_scaler opera sobre arrays NumPy; la serialización segura usa
        # tensores para evitar objetos pickle, por eso se convierten aquí.
        scaler_stats["median"] = scaler_stats["median"].numpy()
        scaler_stats["iqr"] = scaler_stats["iqr"].numpy()

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
    parser.add_argument("--min-event-sensitivity", type=float, default=None, help="Sensibilidad objetivo a nivel EVENTO (crisis): elige el umbral con menor FDR que la alcance.")
    parser.add_argument("--positives-for-event", type=int, default=None, help="Alarma si hay >= N positivos dentro de las últimas N_WINDOW ventanas (nivel evento).")
    parser.add_argument("--window-range-for-event", type=int, default=None, help="Cantidad de ventanas consecutivas consideradas para la regla de evento (nivel evento).")
    parser.add_argument("--event-min-alarm-interval", type=float, default=None, help="Segundos mínimos entre dos alarmas (nivel evento).")
    parser.add_argument("--event-max-latency", type=float, default=None, help="Ventana de latencia [onset, onset+max] para dar una crisis por detectada.")
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


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    model = rebuild_model(checkpoint)
    model.to(device)
    model.eval()  # modo evaluar (chau dropout y eso)

    # el event config guarda n_within y n_window (tiro alarma si en n_window ventanas hay n_within positivas), tamb guarda min alarm interval, max latency, y min event sensitivity
    event_config = checkpoint.get("event_config", {})
    
    min_event_sensitivity = ( args.min_event_sensitivity if args.min_event_sensitivity is not None else event_config.get("min_event_sensitivity", MIN_EVENT_SENSITIVITY))
    positives_for_event = args.positives_for_event if args.positives_for_event is not None else event_config.get("n_within", POSITIVES_FOR_EVENT)
    window_range_for_event = args.window_range_for_event if args.window_range_for_event is not None else event_config.get("n_window", WINDOW_RANGE_FOR_EVENT)
    event_min_alarm_interval = args.event_min_alarm_interval if args.event_min_alarm_interval is not None else event_config.get("min_alarm_interval", EVENT_MIN_ALARM_INTERVAL)
    event_max_latency = args.event_max_latency if args.event_max_latency is not None else event_config.get("max_latency", EVENT_MAX_LATENCY)
    if not 0.0 <= min_event_sensitivity <= 1.0:
        raise ValueError("min_event_sensitivity debe estar entre 0 y 1")
    if positives_for_event <= 0 or window_range_for_event <= 0 or positives_for_event > window_range_for_event:
        raise ValueError("Configuración de ventanas de evento inválida")
    if event_min_alarm_interval < 0 or event_max_latency < 0:
        raise ValueError("Los intervalos de evento no pueden ser negativos")

    threshold = args.threshold if args.threshold is not None else checkpoint.get("op_threshold", checkpoint.get("threshold", THRESHOLD))

    # Metadata del checkpoint 
    print("=" * 70)
    print("EVALUACIÓN DE CHECKPOINT")
    print("=" * 70)
    print(f"  Checkpoint:      {checkpoint_path.name}")
    print(f"  Época guardada:  {checkpoint.get('epoch', '?')}")
    print(f"  best_val_loss:   {checkpoint.get('best_val_loss', float('nan')):.6f}")
    print(f"  pos_weight:      {checkpoint.get('pos_weight', '?')}")
    print(f"  seed:            {checkpoint.get('seed', '?')}")
    print(f"  threshold:       {threshold}")
    print("  decision time:   fin de ventana (unica convencion)")
    print(f"  split a evaluar: {args.split}")

    # Carga el split que se quiere evaluar.
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    # El scaler_stats viene adentro del checkpoint y lo uso para escalar las ventanas de val/test EXACTAMENTE igual que en train
    scaler_stats = checkpoint.get("scaler_stats")
    if scaler_stats is None:
        sys.exit("El checkpoint no contiene estadísticas del scaler.")

    # armo train/val/test. Para val/test recorre TODAS las ventanas en orden (sin undersampling, sin shuffle)
    # Construye datasets con el scaler embebido en el checkpoint.
    loaders = build_splits_dataloaders(args.data_dir, split, scaler_stats,batch_size=args.batch_size, num_workers=args.num_workers, limit_files=args.limit_files,)
    dataset = loaders[args.split]["dataset"] #acá corro --split test o --split val t elijo
    loader = loaders[args.split]["dataloader"]

    n_pos = int(dataset.n_positive)
    n_total = len(dataset)
    print(f"  Ventanas {args.split}: {n_total}  ({n_pos} positivas, {n_total - n_pos} negativas)")

    # evaluar
    # mido el error sobre la distribución natural, x ende el Binary Cross entropy va sin pos-weight
    criterion = nn.BCEWithLogitsLoss()
    avg_loss, probs, labels = evaluate(model, loader, criterion, device)

    # Métricas clínicas sobre TODAS las ventanas del split
    #Construyo el índice global ventana -> (archivo, ventana local) y el vector de label
    dataset._ensure_index()#pq hago esto acpa?
    m = binary_metrics(probs, labels, threshold, file_ids=dataset.file_ids, local_ids=dataset.local_ids,)

    print("-" * 70)
    print(f"  Loss promedio ({args.split}):  {avg_loss:.6f}")
    print(f"  [NIVEL VENTANA @ umbral {threshold:.2f}]")
    print(f"  Sensibilidad (recall):          {m['sensibility']:.4f}")
    print(f"  Especificidad:                  {m['specificity']:.4f}")
    print(f"  FPR (tasa falsos positivos):    {m['false_positive_rate']:.4f}")
    print(f"  Falsos positivos por hora:      {m['false_positive_per_hour']:.4f}")
    print(f"  Accuracy:                       {m['accuracy']:.4f}")
    print(f"  F1:                             {m['f1']:.4f}")
    print(f"  MCC:                            {m['mcc']:.4f}")
    print(f"  Conteos -> TP={m['actual_positive']}  FP={m['false_positive']}  "
          f"TN={m['actual_negative']}  FN={m['false_negative']}")

    # 7) Métricas a nivel EVENTO (crisis) + punto de operación clínico
    if args.split == "test":
        # Test queda reservado para medir: usa el threshold elegido en val.
        ev_threshold = threshold
        ev = event_metrics(
            probs,
            labels,
            file_ids=dataset.file_ids,
            local_ids=dataset.local_ids,
            valid_files=dataset.valid_files,
            annotations=dataset.annotations,
            threshold=ev_threshold,
            n_within=positives_for_event,
            n_window=window_range_for_event,
            min_alarm_interval=event_min_alarm_interval,
            max_latency=event_max_latency,
        )
        print("\n  [NIVEL EVENTO (crisis)]")
        print(f"  Métricas sobre test usando threshold fijado en validación: {ev_threshold:.2f}")
    else:
        ev, ev_threshold = select_event_operating_point(
            probs, labels,
            file_ids=dataset.file_ids, local_ids=dataset.local_ids,
            valid_files=dataset.valid_files, annotations=dataset.annotations,
            min_sensibility=min_event_sensitivity,
            n_within=positives_for_event, n_window=window_range_for_event,
            min_alarm_interval=event_min_alarm_interval,
            max_latency=event_max_latency,
        )

    if ev is None:
        # ningún umbral alcanza el objetivo de sensibilidad de crisis -> mostramos el
        # umbral con la máxima sensibilidad de crisis alcanzable, a título informativo.
        for _t in np.arange(0.05, 1.0, 0.05):
            _e = event_metrics(probs, labels,
                                file_ids=dataset.file_ids, local_ids=dataset.local_ids,
                                valid_files=dataset.valid_files, annotations=dataset.annotations,
                               threshold=_t,
                                 n_within=positives_for_event, n_window=window_range_for_event,
                                 min_alarm_interval=event_min_alarm_interval,
                                 max_latency=event_max_latency)
            if ev is None or _e["sensibility"] > ev["sensibility"]:
                ev = _e
                ev_threshold = float(_t)
        print("\n  [NIVEL EVENTO (crisis)]")
        print(f"  [WARN] Ningún umbral alcanzó sens de crisis >= {min_event_sensitivity}. "
              f"Mostrando el de máxima sensibilidad (umbral {ev_threshold:.2f}).")
    elif args.split == "test":
        print(f"  Threshold aplicado (fijado en validación): {ev_threshold:.2f}")
    else:
        print("\n  [NIVEL EVENTO (crisis)]")
        print(f"  Punto de operación (sens de crisis >= {min_event_sensitivity}, menor FDR): "
              f"umbral={ev_threshold:.2f}  FDR={ev['false_detection_per_hour']:.2f}/h")

    print(f"  Sensibilidad:              {ev['sensibility']:.4f}  ({ev['n_detected']}/{ev['n_seizures']} crisis detectadas)")
    print(f"  Falsas detecciones/hora:   {ev['false_detection_per_hour']:.4f}  ({ev['n_false_alarms']} falsas alarmas en {ev['total_hours']:.1f} h cubiertas)")
    print(f"  Latencia media:            {ev['latency_mean']:.2f} s")
    print(f"  Latencia mediana:          {ev['latency_median']:.2f} s")
    print(f"  Postprocesado:             >= {positives_for_event} positivos en {window_range_for_event} ventanas consecutivas")
    print("=" * 70)


if __name__ == "__main__":
    main()
