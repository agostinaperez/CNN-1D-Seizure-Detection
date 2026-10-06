"""
Métricas de cómputo del modelo (hardware-independientes y dependientes).
  - Independientes del hardware: nº de parámetros y FLOPs/MACs por ventana.
  - Dependientes del hardware: RAM/VRAM pico, latencia de inferencia por ventana.
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
    """Parámetros totales, entrenables y desglose por bloque."""
    #total es todos los pesos de cada capa, los sesgos de cada capa, y cualquier otro tensor registado como parametro
    total = sum(p.numel() for p in model.parameters())
    #los mismos de arriba pero solo los que se van a actualizar en el entrenamiento. Algunos que no entran pueden ser los que se congelan para transfer learning o los que no tienen gradiente
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #fc es fully connectes o sea de las capas densas
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
    Los MACs (multiply-accumulate) mide la cantidad de operaciones combinadas de multiplicación y suma (input.peso + bias) que hace el modelo. Representa la cantidad
    de operaciones básicas q el modelo hace para procesar una entrada. La cantidad de MACs determina el uso de memoria de la red, porque están relacionadas directamente con el número de parámetros y activaciones en la misma
    Los FLOPs son la cantidad total de operaciones matemáticas que necesita el modelo para procesar una entrada. Permite estimar el costo computacional de una red neuronal (a + FLOPs, tarda + tiempo entrenarla)
    FLOPs ≈ 2 × MACs (1 MAC se compone de 1 mult + 1 suma).
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


def measure_inference_latency(model: nn.Module, sample_input: torch.Tensor, device: str, warmup: int = 10, reps: int = 50, ) -> dict:
    """latencia x inferencia (cuánto tarda el modelo en hacer una predicción) y costo por hora de EEG.
    """
    model.eval()
    x = sample_input.to(device)
    with torch.no_grad():
        #corro una cantidad de forwards para que el modelo se "caliente" y no haya overhead de inicialización. A esas corridas las descarto
        for _ in range(warmup):
            _ = model(x)
        #ahora sí corro el modelo y mido el tiempo con time.perf_counter() que es más preciso que time.time()
        per_rep_ms: list[float] = []
        for _ in range(reps):
            if device == "cuda":
                #la gpu es asincrónica, entonces hay que sincronizarla antes de medir el tiempo para que no haya operaciones pendientes. torch.cuda.synchronize() bloquea el hilo hasta que todas las operaciones en la GPU hayan terminado
                torch.cuda.synchronize()
            #tiempo inicial
            t0 = time.perf_counter()
            #Lo ejecuto
            _ = model(x)
            if device == "cuda":
                #sincronizo y guardo el tiempo final q tardó
                torch.cuda.synchronize()
            per_rep_ms.append((time.perf_counter() - t0) * 1000.0) #en milisegundos (ms)

    ms_mean = float(np.mean(per_rep_ms)) #promedio de cuanto tarda una ventana
    ms_p95 = float(np.percentile(per_rep_ms, 95)) #percentil 95 de cuanto tarda

    # Ventanas por hora de EEG. Stride samples es cuanto avanza cada ventana nueva. Al tener 50% de solapamiento, cada ventana avanza solo la mitad de su largo
    windows_per_hour = (3600.0 * FS) / STRIDE_SAMPLES #cuantas ventanas hay en 1 hs de señal
    seconds_per_hour = ms_mean * windows_per_hour / 1000.0 #cuántos segundos de cómputo me lleva procesar 1 hs (onda cuan rápido em responde el modelo YA ENTRENADO)
    # Si es < 3600 el modelo puede procesar en tiempo real (procesa más rápido de lo que llega la señal)
    # Si es > 3600, es más lento que el tiempo real
    return {
        "ms_per_window_mean": ms_mean,
        "ms_per_window_p95": ms_p95,
        "seconds_per_hour_of_eeg": seconds_per_hour,
    }
