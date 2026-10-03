"""Selecciona el mejor escenario usando exclusivamente métricas de validación.

Uso:
    python -m src.select_best \
        --checkpoints models/bce.pt models/focal.pt \
        --out results/best_model.json

El test no se carga ni se consulta. Cada checkpoint ya contiene las métricas
del punto operativo calculadas sobre validación durante el entrenamiento.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 4:
        raise RuntimeError(f"Checkpoint no compatible: {path} (se esperaba formato 4).")
    required = (
        "protocol_version",
        "split_id",
        "scaler_id",
        "preprocessing_config",
        "event_config",
        "threshold_grid",
        "model_config",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"Checkpoint incompleto: {path}; faltan {', '.join(missing)}.")
    return checkpoint


def _compatibility_signature(checkpoint: dict) -> tuple:
    """Campos que deben ser idénticos al comparar escenarios."""
    return tuple(
        json.dumps(checkpoint[key], sort_keys=True, separators=(",", ":"))
        for key in (
            "protocol_version",
            "split_id",
            "scaler_id",
            "preprocessing_config",
            "event_config",
            "threshold_grid",
            "model_config",
        )
    )


def _selection_key(checkpoint: dict, min_event_sensitivity: float) -> tuple:
    op = checkpoint.get("op_metrics")
    if not op:
        return (0, -float("inf"), -float("inf"), -float("inf"), -float("inf"))

    sensitivity = float(op.get("sensibility", 0.0))
    feasible = sensitivity >= min_event_sensitivity
    auprc = float(checkpoint.get("validation_auprc", float("-inf")))
    return (
        1 if feasible else 0,
        sensitivity,
        -float(op.get("median_false_alarms_per_hour", float("inf"))),
        auprc,
        -float(checkpoint.get("best_val_loss", float("inf"))),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seleccionar escenario por métricas de validación.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--min-event-sensitivity", type=float, default=0.0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.min_event_sensitivity <= 1.0:
        parser.error("--min-event-sensitivity debe estar entre 0 y 1")
    return args


def main() -> None:
    args = parse_args()
    candidates = []
    reference_signature = None
    for raw_path in args.checkpoints:
        path = Path(raw_path)
        checkpoint = _load(path)
        signature = _compatibility_signature(checkpoint)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise RuntimeError(
                "Los checkpoints no son comparables: difieren en split, scaler, "
                "preprocesamiento, arquitectura o configuración clínica."
            )
        candidates.append({
            "path": str(path),
            "loss": checkpoint.get("loss"),
            "seed": checkpoint.get("seed"),
            "epoch": checkpoint.get("epoch"),
            "op_threshold": checkpoint.get("op_threshold"),
            "op_metrics": checkpoint.get("op_metrics"),
            "best_val_loss": checkpoint.get("best_val_loss"),
            "validation_auprc": checkpoint.get("validation_auprc"),
            "selection_metric": checkpoint.get("selection_metric"),
        })

    ranked = sorted(
        candidates,
        key=lambda candidate: _selection_key(candidate, args.min_event_sensitivity),
        reverse=True,
    )
    if not ranked:
        raise RuntimeError("No se recibieron checkpoints.")

    selected = ranked[0]
    target_met = bool(
        selected.get("op_metrics")
        and float(selected["op_metrics"].get("sensibility", 0.0)) >= args.min_event_sensitivity
    )
    output = {
        "selected": selected,
        "min_event_sensitivity": args.min_event_sensitivity,
        "target_met": target_met,
        "candidates": ranked,
        "selection_source": "validation_only",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Mejor escenario según validación: {selected['path']}")
    if not target_met:
        print("[WARN] Ningún escenario alcanzó la sensibilidad mínima solicitada.")
    print(f"Resultado guardado en: {out}")


if __name__ == "__main__":
    main()
