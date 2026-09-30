"""
CLI de inferencia: clasifica las ventanas de un EDF con un modelo ya entrenado.

    python -m src.inference --input ej.edf --checkpoint models/best.pt

Flujo (tal cual la Fase 7 del PLAN_MVP):
  1. Carga el checkpoint (pesos + config + stats del escalador + threshold).
  2. Procesa el EDF: 16 canales TUEV -> filtro 0.5-50 Hz -> ventaneo 5.12 s / 50%
     -> escalado robusto con las stats GUARDADAS en el checkpoint (las de train,
     para escalar EXACTAMENTE igual que en el entrenamiento).
  3. Clasifica cada ventana (forward, sin gradientes) -> probabilidad de crisis.
  4. Aplica el threshold -> etiqueta binaria por ventana.
   5. Imprime una tabla (ventana, inicio, tiempo de decisión, probabilidad,
      predicción)
      + un resumen de alarmas tras el postprocesado temporal.

A diferencia de evaluate.py, acá NO hay etiquetas reales: es para datos NUEVOS
(predicción pura). El EDF se procesa offline completo. Por eso process_edf recibe
seizures=[] (no se usa para etiquetar, solo para obtener las ventanas y sus tiempos
de inicio).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import (
    BATCH_SIZE,
    EVENT_MIN_ALARM_INTERVAL,
    WINDOW_RANGE_FOR_EVENT,
    POSITIVES_FOR_EVENT,
    MODELS_DIR,
    THRESHOLD,
    WIN_SECONDS_EFFECTIVE,
)
# Reuso lo que ya escribí en evaluate.py: cargar el checkpoint y reconstruir la CNN
# idéntica. (Evito duplicar esa lógica en dos archivos.)
from src.evaluate import load_checkpoint, rebuild_model
from src.preprocessing import process_edf
from src.train import get_device
from src.timing import decision_times_from_starts
from src.protocol import preprocessing_config


def classify_windows(model, windows: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """
    Pasa TODAS las ventanas por la red (en batches para no reventar la memoria)
    y devuelve la PROBABILIDAD de crisis de cada una, como array numpy (n,).

    - model: la CNN en modo eval.
    - windows: array (n_windows, 16, 1310) ya filtrado y escalado.
    """
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero")
    if len(windows) == 0:
        return np.empty(0, dtype=np.float32)

    model.eval()
    probs: list[torch.Tensor] = []

    # torch.no_grad(): no construyo grafo de gradientes, total no voy a entrenar.
    # Ahorra memoria y es más rápido (igual que evaluate() en train.py).
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]               # (batch, 16, 1310)
            x = torch.from_numpy(chunk).float().to(device)  # numpy -> tensor en CPU/GPU
            logits = model(x)                               # (batch, 1) logit crudo
            # sigmoide sobre el logit -> probabilidad en (0, 1); .cpu() para seguir en numpy.
            probs.append(torch.sigmoid(logits.squeeze(-1)).cpu())

    return torch.cat(probs).numpy()  # (n_windows,)


def detect_alarms(preds: np.ndarray, starts: np.ndarray, *,
                  n_within: int = POSITIVES_FOR_EVENT, n_window: int = WINDOW_RANGE_FOR_EVENT,
                  min_alarm_interval: float = EVENT_MIN_ALARM_INTERVAL) -> list[float]:
    """
    Postprocesado a nivel EVENTO (el MISMO que usa event_metrics en la evaluación):
    dispara una alarma cuando hay >= `n_within` predicciones positivas dentro de las
    últimas `n_window` ventanas consecutivas, y suprime alarmas separadas por menos
    de `min_alarm_interval` segundos.

    `starts` representa el tiempo de decisión de cada ventana, no su inicio.
    Devuelve los tiempos (segundos) de cada alarma disparada. A diferencia de
    group_events, acá una sola ventana positiva aislada NO dispara nada: hace falta
    una ráfaga. Así el umbral bajo (ej. op_threshold=0.10) no inunda de falsas alarmas.
    """
    alarm_times: list[float] = []
    last_alarm = -np.inf
    recent: list[bool] = []
    for i, t in enumerate(starts):
        recent.append(bool(preds[i]))
        if len(recent) > n_window:
            recent.pop(0)
        if len(recent) >= n_window and sum(recent) >= n_within:
            if t - last_alarm >= min_alarm_interval:
                alarm_times.append(float(t))
                last_alarm = float(t)
    return alarm_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia sobre un EDF con un checkpoint entrenado")
    parser.add_argument("--input", type=str, required=True,
                        help="Ruta del EDF a clasificar (CHB-MIT).")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_DIR / "best.pt"),
                        help="Ruta del .pt entrenado (lo que guardó train.py).")
    parser.add_argument("--device", type=str, default=None, help="'cuda' o 'cpu'.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Cuántas ventanas se clasifican por pasada (memoria).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Umbral de decisión. Por defecto usa el del checkpoint.")
    parser.add_argument("--event-n-within", type=int, default=None,
                        dest="positives_for_event",
                        help="Alarma si hay >= N positivos dentro de las últimas N_WINDOW ventanas.")
    parser.add_argument("--event-n-window", type=int, default=None,
                        dest="window_range_for_event")
    parser.add_argument("--event-min-alarm-interval", type=float, default=None,
                        help="Segundos mínimos entre dos alarmas.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limitar la tabla impresa a N ventanas (smoke-test).")
    parser.add_argument("--limit-windows", type=int, default=None,
                        help="Analizar solo las primeras N ventanas del EDF (smoke-test).")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size debe ser mayor que cero")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold debe estar entre 0 y 1")
    if args.max_rows is not None and args.max_rows < 0:
        parser.error("--max-rows no puede ser negativo")
    if args.limit_windows is not None and args.limit_windows <= 0:
        parser.error("--limit-windows debe ser mayor que cero")
    return args


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    # 1) Checkpoint + modelo ---------------------------------------------------------
    # Carga pesos, arquitectura, scaler y metadata del modelo.
    ckpt = load_checkpoint(Path(args.checkpoint))
    model = rebuild_model(ckpt)
    model.to(device)
    model.eval()

    # Umbral: el punto de operación elegido en validación (op_threshold), salvo que lo
    # pise --threshold. Si el checkpoint es viejo y no trae op_threshold, uso el fijo.
    event_config = ckpt.get("event_config", {})
    positives_for_event = args.positives_for_event if args.positives_for_event is not None else event_config.get("n_within", POSITIVES_FOR_EVENT)
    window_range_for_event = args.window_range_for_event if args.window_range_for_event is not None else event_config.get("n_window", WINDOW_RANGE_FOR_EVENT)
    event_min_alarm_interval = args.event_min_alarm_interval if args.event_min_alarm_interval is not None else event_config.get("min_alarm_interval", EVENT_MIN_ALARM_INTERVAL)
    if positives_for_event <= 0 or window_range_for_event <= 0 or positives_for_event > window_range_for_event:
        raise ValueError("Configuración de ventanas de evento inválida")
    if event_min_alarm_interval < 0:
        raise ValueError("event_min_alarm_interval no puede ser negativo")

    threshold = args.threshold if args.threshold is not None else ckpt.get("op_threshold", ckpt.get("threshold", THRESHOLD))
    # Stats del escalador GUARDADAS en el checkpoint (median/iqr de train).
    scaler_stats = ckpt.get("scaler_stats")
    if scaler_stats is None:
        sys.exit("El checkpoint no contiene estadísticas del scaler.")
    if scaler_stats.get("preprocessing_config") and scaler_stats["preprocessing_config"] != preprocessing_config():
        raise RuntimeError("El checkpoint fue creado con otro preprocesamiento.")

    # 2) Procesar el EDF ------------------------------------------------------------
    path = Path(args.input)
    if not path.exists():
        sys.exit(f"No existe: {path}")

    # seizures=[] porque acá no tenemos etiquetas reales (predicción pura). process_edf
    # igual nos devuelve las ventanas filtradas + escaladas y sus tiempos de inicio.
    res = process_edf(path, seizures=[], scaler_stats=scaler_stats, do_scale=True)
    if not res["ok"]:
        sys.exit(f"[FAIL] {res['reason']}")

    windows = res["windows"]       # (n_windows, 16, 1310)
    starts = res["starts"]         # (n_windows,) inicio de cada ventana en SEGUNDOS
    # Convierte los inicios de ventana en tiempos de decisión compartidos con metrics.py.
    decision_times = decision_times_from_starts(starts)

    if args.limit_windows is not None:
        windows = windows[:args.limit_windows]
        starts = starts[:args.limit_windows]
        decision_times = decision_times[:args.limit_windows]

    n = len(windows)

    # 3) Clasificar ------------------------------------------------------------------
    probs = classify_windows(model, windows, device, args.batch_size)
    preds = (probs >= threshold).astype(int)  # aplica el umbral -> 0/1

    # 4) Tabla por ventana -----------------------------------------------------------
    print("=" * 72)
    print(f"INFERENCIA — {path.name}")
    print(f"Checkpoint: {Path(args.checkpoint).name}  |  threshold: {threshold}")
    print("=" * 72)
    print(f"{'ventana':>7} | {'inicio(s)':>9} | {'decision(s)':>11} | {'prob':>6} | predicción")
    print("-" * 72)

    max_rows = n if args.max_rows is None else min(args.max_rows, n)
    for i in range(max_rows):
        label = "crisis" if preds[i] == 1 else "normal"
        print(f"{i:>7} | {starts[i]:>9.2f} | {decision_times[i]:>11.2f} | {probs[i]:>6.3f} | {label}")

    if args.max_rows is not None and n > args.max_rows:
        print(f"... (se omiten {n - args.max_rows} ventanas)")

    # 5) Resumen ---------------------------------------------------------------------
    n_pos = int(preds.sum())
    total_duration = float(starts[-1]) + WIN_SECONDS_EFFECTIVE if n > 0 else 0.0  # fin del último tramo

    print("-" * 72)
    print(f"Ventanas totales:      {n}")
    print(f"Marcadas como crisis:  {n_pos}  ({100 * n_pos / n:.2f}%)" if n > 0 else "Sin ventanas")
    print(f"Duración del registro: {total_duration:.2f} s  ({total_duration / 60:.2f} min)")

    alarms = detect_alarms(preds, decision_times,
                            n_within=positives_for_event, n_window=window_range_for_event,
                            min_alarm_interval=event_min_alarm_interval)
    if alarms:
        print(f"\nAlarmas disparadas ({len(alarms)}, postprocesado {positives_for_event}-de-{window_range_for_event}):")
        for k, t in enumerate(alarms, 1):
            print(f"  #{k:>2}: a los {t:8.2f} s")
    else:
        print("\nNo se disparó ninguna alarma (postprocesado incluido).")
    print("=" * 72)


if __name__ == "__main__":
    main()
