"""
Métricas de cómputo del modelo (hardware-independientes y dependientes).

Separa lo que no cambia según la máquina de lo que sí:

  - Independientes del hardware: nº de parámetros y FLOPs/MACs por ventana.
  - Dependientes del hardware: RAM/VRAM pico, latencia de inferencia por ventana.

Se usan desde `train.py` para volcar un reporte de cómputo en el `.history.json`.
Ver `PLAN_COMPUTO.md` para la justificación de cada métrica.
"""

from __future__ import annotations

import time

import numpy as np
import psutil
import torch
import torch.nn as nn
from torchinfo import summary

from src.config import FS, STRIDE_SAMPLES


def count_parameters(model: nn.Module) -> dict:
    """Parámetros totales, entrenables y desglose por bloque (conv/fc/otros)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_block = {"conv": 0, "fc": 0, "other": 0}
    for module in model.modules():
        n = sum(p.numel() for p in module.parameters(recurse=False))
        if n == 0:
            continue
        if isinstance(module, nn.Conv1d):
            by_block["conv"] += n
        elif isinstance(module, nn.Linear):
            by_block["fc"] += n
        else:
            by_block["other"] += n
    return {
        "n_params_total": total,
        "n_params_trainable": trainable,
        "params_by_block": by_block,
    }


def measure_flops(model: nn.Module, input_shape: tuple[int, ...]) -> dict:
    """MACs y FLOPs de un forward para una muestra (batch=1) vía torchinfo.

    FLOPs ≈ 2 × MACs (multiply-accumulate = 1 mult + 1 add).
    """
    info = summary(model, input_size=input_shape, verbose=0, depth=0)
    macs = float(info.total_mult_adds) if info.total_mult_adds is not None else 0.0
    return {
        "macs_per_window": macs,
        "flops_per_window": 2.0 * macs,
    }


def peak_ram_mb() -> float:
    """Memoria RAM actual del proceso (RSS) en MB."""
    return psutil.Process().memory_info().rss / (1024.0 ** 2)


def measure_inference_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: str,
    warmup: int = 10,
    reps: int = 50,
) -> dict:
    """Latencia de inferencia por ventana (batch=1) y costo por hora de EEG.

    Hace `warmup` forwards descartados (cold-start) y mide `reps` forwards,
    reportando media y p95. Deriva los segundos que tarda en procesar una hora
    de EEG (considerando el solapamiento del 50% entre ventanas).
    """
    model.eval()
    x = sample_input.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        per_rep_ms: list[float] = []
        for _ in range(reps):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            per_rep_ms.append((time.perf_counter() - t0) * 1000.0)

    ms_mean = float(np.mean(per_rep_ms))
    ms_p95 = float(np.percentile(per_rep_ms, 95))

    # Ventanas por hora de EEG: cada ventana nueva avanza STRIDE_SAMPLES muestras.
    windows_per_hour = (3600.0 * FS) / STRIDE_SAMPLES
    seconds_per_hour = ms_mean * windows_per_hour / 1000.0

    return {
        "ms_per_window_mean": ms_mean,
        "ms_per_window_p95": ms_p95,
        "seconds_per_hour_of_eeg": seconds_per_hour,
    }
