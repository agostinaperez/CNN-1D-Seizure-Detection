"""Selección del mejor escenario a partir de las métricas de validación.

Lee el historial que guardó ``src.train`` (``<checkpoint>.history.json``) de
cada escenario entrenado y compara su punto de operación en VALIDACIÓN
(menor FDR entre los que alcanzan la sensibilidad de crisis objetivo). El
test NO se usa acá: se evalúa una sola vez, después, sobre el ganador.

Uso:
    python -m src.select_best \
        --checkpoints models/ratio3_pw1.pt models/ratio4_pw1.pt \
                       models/ratio1_pw1.pt models/ratio3_pw15.pt \
        --min-event-sensitivity 0.9 \
        --out results/best_model.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.config import MIN_EVENT_SENSITIVITY


def _history_path(checkpoint: Path) -> Path:
    """Deriva el historial JSON hermano del checkpoint (nombre.history.json)."""
    return checkpoint.with_name(checkpoint.stem + ".history.json")


def _load_row(checkpoint: Path) -> dict | None:
    """Lee el historial y devuelve las métricas de validación del checkpoint."""
    history_path = _history_path(checkpoint)
    if not checkpoint.exists():
        return {
            "checkpoint": str(checkpoint),
            "history_file": str(history_path),
            "checkpoint_saved": False,
            "fdr_per_hour": math.inf,
            "event_sensitivity": 0.0,
            "error": "checkpoint no existe",
        }
    if not history_path.exists():
        return {
            "checkpoint": str(checkpoint),
            "history_file": str(history_path),
            "checkpoint_saved": False,
            "fdr_per_hour": math.inf,
            "event_sensitivity": 0.0,
            "error": "history.json no existe",
        }
    payload = json.loads(history_path.read_text(encoding="utf-8"))

    experiment_config = payload.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return {
            "checkpoint": str(checkpoint),
            "history_file": str(history_path),
            "checkpoint_saved": False,
            "fdr_per_hour": math.inf,
            "event_sensitivity": 0.0,
            "error": "el historial no contiene experiment_config",
        }

    if payload.get("checkpoint_saved"):
        try:
            checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            return {
                "checkpoint": str(checkpoint),
                "history_file": str(history_path),
                "checkpoint_saved": False,
                "fdr_per_hour": math.inf,
                "event_sensitivity": 0.0,
                "error": f"no se pudo leer el checkpoint: {exc}",
            }
        if checkpoint_payload.get("experiment_config") != experiment_config:
            return {
                "checkpoint": str(checkpoint),
                "history_file": str(history_path),
                "checkpoint_saved": False,
                "fdr_per_hour": math.inf,
                "event_sensitivity": 0.0,
                "error": "history.json y checkpoint tienen configuraciones distintas",
            }

    fdr = payload.get("best_event_false_alarms_per_hour", payload.get("best_fdr_per_hour"))
    sensibility = payload.get("best_op_sensibility")
    return {
        "checkpoint": str(checkpoint),
        "history_file": str(history_path),
        "checkpoint_saved": bool(payload.get("checkpoint_saved", False)),
        "best_epoch": payload.get("best_epoch"),
        "best_val_loss": payload.get("best_val_loss"),
        "fdr_per_hour": float(fdr) if fdr is not None else math.inf,
        "event_false_alarms_per_hour": float(fdr) if fdr is not None else math.inf,
        "event_sensitivity": float(sensibility) if sensibility is not None else 0.0,
        "op_threshold": payload.get("best_op_threshold"),
        "neg_pos_ratio": payload.get("neg_pos_ratio"),
        "pos_weight": payload.get("pos_weight"),
        "noise_std": payload.get("noise_std"),
        "seed": payload.get("seed"),
        "experiment_config": experiment_config,
    }


def select_best(
    checkpoints: list[Path],
    min_event_sensitivity: float = MIN_EVENT_SENSITIVITY,
) -> dict:
    """Compara los escenarios por validación y devuelve el resumen + ganador."""
    rows = []
    for checkpoint in checkpoints:
        row = _load_row(checkpoint)
        row["eligible"] = bool(
            row.get("checkpoint_saved")
            and "error" not in row
            and math.isfinite(row["fdr_per_hour"])
            and row["event_sensitivity"] >= min_event_sensitivity
        )
        rows.append(row)

    compatibility_keys = (
        "protocol_version",
        "split_id",
        "scaler_id",
        "preprocessing",
        "model_config",
        "event_config",
        "min_event_sensitivity",
        "threshold_grid",
        "seed",
    )
    reference = None
    for row in rows:
        config = row.get("experiment_config")
        if not config or "error" in row:
            continue
        signature = {key: config.get(key) for key in compatibility_keys}
        if reference is None:
            reference = signature
        elif signature != reference:
            row["eligible"] = False
            row["error"] = "protocolo incompatible con los demás escenarios"
        if config.get("min_event_sensitivity") != float(min_event_sensitivity):
            row["eligible"] = False
            row["error"] = "sensibilidad objetivo distinta de la selección"

    eligible = [r for r in rows if r.get("eligible")]
    winner = min(eligible, key=lambda r: r["fdr_per_hour"]) if eligible else None

    return {
        "min_event_sensitivity": min_event_sensitivity,
        "results": rows,
        "winner": winner["checkpoint"] if winner else None,
        "selection_metric": "menor tasa de falsas alarmas de evento por hora en validacion entre los escenarios con sensibilidad de crisis >= objetivo",
    }


def _fmt_fdr(value: float) -> str:
    return "inf" if not math.isfinite(value) else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Elegir el mejor escenario por validación")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoints .pt a comparar (sus .history.json deben existir).")
    parser.add_argument("--min-event-sensitivity", type=float, default=MIN_EVENT_SENSITIVITY)
    parser.add_argument("--out", type=str, default=None, help="Guardar resumen JSON opcional.")
    args = parser.parse_args()
    if not 0.0 <= args.min_event_sensitivity <= 1.0:
        parser.error("--min-event-sensitivity debe estar entre 0 y 1")

    summary = select_best(
        [Path(c) for c in args.checkpoints],
        args.min_event_sensitivity,
    )

    print("checkpoint | ratio | pos_weight | noise_std | sens_val | falsas_alarmas_evento/h | umbral | elegible")
    print("-" * 96)
    for row in sorted(summary["results"], key=lambda r: r["fdr_per_hour"]):
        name = Path(row["checkpoint"]).name if "error" not in row else row["checkpoint"]
        if "error" in row:
            print(f"{name} | [ERROR] {row['error']}")
            continue
        print(
            f"{name} | {row['neg_pos_ratio']} | {row['pos_weight']} | "
            f"{row['noise_std']} | {row['event_sensitivity']:.3f} | "
            f"{_fmt_fdr(row['event_false_alarms_per_hour'])} | {row['op_threshold']} | {row['eligible']}"
        )

    if summary["winner"] is None:
        print("\nNingún escenario alcanza el objetivo de sensibilidad en validación.")
    else:
        print(f"\nGanador (evaluar este en test una sola vez): {summary['winner']}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resumen guardado en: {out}")


if __name__ == "__main__":
    main()
