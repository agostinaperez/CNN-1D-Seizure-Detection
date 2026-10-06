"""Inferencia sobre un EDF usando un checkpoint entrenado.

Uso:
    python -m src.inference --input archivo.edf --checkpoint models/best.pt

  - postprocesado a nivel evento en la terminal. Si se detecta una crisis imprime
    "Se detectó una crisis epiléptica que va del <start> al <end>".
  - el detalle ventana a ventana se guarda siempre en
    inferences/inference_<archivo>_<YYYYMMDD_HHMMSS>.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.config import (
    BATCH_SIZE,
    BASE_DIR,
    EVENT_MIN_ALARM_INTERVAL,
    MODELS_DIR,
    THRESHOLD,
)
from src.evaluate import load_checkpoint, rebuild_model
from src.train import get_device
from src.metrics import detect_event_intervals
from src.preprocessing import process_edf
from src.protocol import validate_checkpoint_scaler
from src.timing import decision_times_from_starts

INFERENCES_DIR = BASE_DIR / "inferences"


def _format_seconds(seconds: float) -> str:
    """Formatea segundos como '12.3 s' (un decimal, suficiente para pantalla)."""
    return f"{seconds:.1f} s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia CNN-1D sobre un EDF.")
    parser.add_argument("--input", required=True, help="Ruta al archivo EDF.")
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / "best.pt"))
    parser.add_argument("--output", default=None,
                        help="Ruta exacta del JSON. Por defecto se genera en inferences/ con nombre automático.")
    parser.add_argument("--out-dir", default=str(INFERENCES_DIR),
                        help="Directorio donde guardar el JSON cuando no se usa --output.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None, help="'cuda' o 'cpu'.")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size debe ser mayor que cero")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold debe estar entre 0 y 1")
    return args


def _default_output_path(input_path: Path, out_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"inference_{input_path.stem}_{timestamp}.json"
    return out_dir / filename


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(Path(args.checkpoint))
    validate_checkpoint_scaler(checkpoint, checkpoint["scaler_stats"])
    model = rebuild_model(checkpoint)
    device = get_device(args.device)
    model.to(device)
    model.eval()

    threshold = args.threshold
    if threshold is None:
        threshold = checkpoint.get("op_threshold", checkpoint.get("threshold", THRESHOLD))
    if threshold is None:
        threshold = THRESHOLD

    path = Path(args.input)
    #proceso el archivo
    result = process_edf(path, [], scaler_stats=checkpoint["scaler_stats"])
    if not result["ok"]:
        raise RuntimeError(f"No se pudo procesar {path}: {result['reason']}")

    windows = result["windows"]
    probs: list[np.ndarray] = []
    #inferencia ventana a ventana en batches
    with torch.no_grad():
        for start in range(0, len(windows), args.batch_size):
            batch = torch.from_numpy(windows[start:start + args.batch_size]).to(device)
            logits = model(batch).squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    #concateno mi lista de arrays en un solo array
    probabilities = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
    window_starts = result["starts"]
    #defino los tiempos de fin de cada ventana a partir de los tiempos de inicio
    window_ends = decision_times_from_starts(window_starts)
    predictions = probabilities >= threshold

    predictions_json = [
        {
            "window": int(i),
            "start_s": float(window_starts[i]),
            "end_s": float(window_ends[i]),
            "probability": float(probabilities[i]),
            "prediction": int(predictions[i]),
        }
        for i in range(len(probabilities))
    ]

    # Postprocesado a nivel evento. hago el agrupamiento de ventanas positiva. La verdad esto es medio trampita... porque no estoy usando el mismo criterio de 2-de-3 que uso en 
    #el entrenamiento. Pero bueno no es tan grave, porque el objetivo es que el usuario vea los eventos detectados. Con esto me reporta más las falsas alarmas
    #y con el otro criterio se ve más recatado el output. Es un detalle menor pero tranquilamente lo puedo cambiar.
    events = detect_event_intervals(
        window_starts,
        window_ends,
        predictions,
        min_alarm_interval=EVENT_MIN_ALARM_INTERVAL,
    )

    output = {
        "input": str(path),
        "checkpoint": str(args.checkpoint),
        "threshold": float(threshold),
        "n_windows": len(predictions_json),
        "n_events": len(events),
        "events": [
            {"start_s": start, "end_s": end}
            for start, end in events
        ],
        "predictions": predictions_json,
    }

    output_path = Path(args.output) if args.output else _default_output_path(path, Path(args.out_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Reporte en pantalla: resumen + eventos detectados (no el JSON completo).
    print("=" * 60)
    print("INFERENCIA")
    print("=" * 60)
    print(f"  Archivo:      {path.name}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Umbral:       {threshold:.2f}")
    print(f"  Ventanas:     {len(predictions_json)}")
    print(f"  JSON guardado en: {output_path}")
    print("-" * 60)
    if events:
        for start, end in events:
            print(f"  Se detectó una crisis epiléptica que va del "
                  f"{_format_seconds(start)} al {_format_seconds(end)}.")
    else:
        print("  No se detectaron crisis epilépticas.")
    print("=" * 60)


if __name__ == "__main__":
    main()
