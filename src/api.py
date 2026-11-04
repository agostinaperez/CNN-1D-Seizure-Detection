"""
API FastAPI

Expone:
  - GET  /api/models              -> lista de checkpoints .pt en models/
  - GET  /api/models/{name}       -> info completa del modelo (config, cómputo, historial)
  - POST /api/infer               -> inferencia sobre un EDF subido por el usuario
  - sirve la interfaz Vue en la raíz.

    python -m src.api
    # o bien: uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import (
    BASE_DIR,
    BATCH_SIZE,
    CHANNELS_TUEV,
    EVENT_MIN_ALARM_INTERVAL,
    FS,
    MODELS_DIR,
    N_CHANNELS,
    THRESHOLD,
    WIN_SAMPLES,
    WIN_SECONDS_EFFECTIVE,
)
from src.compute_metrics import count_parameters, measure_flops, measure_inference_latency
from src.evaluate import load_checkpoint, rebuild_model
from src.metrics import detect_event_intervals
from src.preprocessing import apply_scaler, filter_bandpass, load_16_channels, segment_windows
from src.protocol import validate_checkpoint_scaler
from src.timing import decision_times_from_starts
from src.train import get_device

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

app = FastAPI(title="Detección de crisis EEG — CNN-1D", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache de modelos cargados (name -> dict). Evita re-cargar y re-medir a cada request. Se usa en las dos rutas de modelo y en inferencia.
_MODEL_CACHE: dict[str, dict[str, Any]] = {}


def _list_model_files() -> list[Path]:
    """Checkpoints .pt directamente en models/ (sin subcarpetas como unused/)."""
    return sorted(p for p in MODELS_DIR.glob("*.pt") if p.is_file())


def _history_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + ".history.json")


def _validate_model_name(name: str) -> Path:
    if not name.endswith(".pt") or Path(name).name != name:
        raise HTTPException(status_code=400, detail="Nombre de modelo inválido")
    path = MODELS_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Modelo '{name}' no encontrado en models/")
    return path


def _compute_live(model: torch.nn.Module, device: str) -> dict[str, Any]:
    """Parámetros, MACs/FLOPs y latencia medidos en vivo (independiente del historial)."""
    params = count_parameters(model)
    flops = measure_flops(model, (1, N_CHANNELS, WIN_SAMPLES))
    latency = measure_inference_latency(
        model, torch.randn(1, N_CHANNELS, WIN_SAMPLES), device, warmup=5, reps=20
    )
    return {
        "device": device,
        "torch_version": torch.__version__,
        "n_params_total": params["n_params_total"],
        "n_params_trainable": params["n_params_trainable"],
        "params_by_block": params["params_by_block"],
        "macs_per_window": flops["macs_per_window"],
        "flops_per_window": flops["flops_per_window"],
        "inference_ms_per_window_mean": latency["ms_per_window_mean"],
        "inference_ms_per_window_p95": latency["ms_per_window_p95"],
        "seconds_per_hour_of_eeg": latency["seconds_per_hour_of_eeg"],
    }


def _load_model(name: str) -> dict[str, Any]:
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]

    path = _validate_model_name(name)
    checkpoint = load_checkpoint(path)
    validate_checkpoint_scaler(checkpoint, checkpoint["scaler_stats"])
    model = rebuild_model(checkpoint)
    device = get_device(None)
    model.to(device)
    model.eval()

    threshold = checkpoint.get("op_threshold", checkpoint.get("threshold", THRESHOLD))
    if threshold is None:
        threshold = THRESHOLD

    state = {
        "name": name,
        "path": path,
        "checkpoint": checkpoint,
        "model": model,
        "device": device,
        "threshold": float(threshold),
        "compute": _compute_live(model, device),
    }
    _MODEL_CACHE[name] = state
    return state


def _load_history(name: str) -> dict[str, Any] | None:
    path = _validate_model_name(name)
    hp = _history_path(path)
    if not hp.is_file():
        return None
    try:
        return json.loads(hp.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "models": [p.name for p in _list_model_files()]}


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    models = []
    for path in _list_model_files():
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        try:
            cp = load_checkpoint(path)
            threshold = cp.get("op_threshold", cp.get("threshold", THRESHOLD))
            info = {
                "name": path.name,
                "size_mb": round(size_mb, 2),
                "threshold": float(threshold) if threshold is not None else None,
                "loss": cp.get("loss"),
                "focal_alpha": cp.get("focal_alpha"),
                "focal_gamma": cp.get("focal_gamma"),
                "epoch": cp.get("epoch"),
                "selection_metric": cp.get("selection_metric"),
                "has_history": _history_path(path).is_file(),
            }
        except Exception as exc:  # checkpoint corrupto: lo listamos pero marcado
            info = {
                "name": path.name,
                "size_mb": round(size_mb, 2),
                "error": str(exc),
                "has_history": _history_path(path).is_file(),
            }
        models.append(info)
    return {"models": models}


@app.get("/api/models/{name}")
def model_info(name: str) -> dict[str, Any]:
    state = _load_model(name)
    cp = state["checkpoint"]
    history = _load_history(name)

    op_metrics = cp.get("op_metrics")
    if isinstance(op_metrics, dict):
        # by_patient puede traer claves con arrays numpy -> lo serializamos liviano
        op_metrics = _json_safe(op_metrics)

    return {
        "name": name,
        "size_mb": round(state["path"].stat().st_size / (1024.0 ** 2), 2),
        "threshold": state["threshold"],
        "loss": cp.get("loss"),
        "focal_alpha": cp.get("focal_alpha"),
        "focal_gamma": cp.get("focal_gamma"),
        "epoch": cp.get("epoch"),
        "best_val_loss": cp.get("best_val_loss"),
        "checkpoint_val_loss": cp.get("checkpoint_val_loss"),
        "pos_weight": cp.get("pos_weight"),
        "seed": cp.get("seed"),
        "selection_metric": cp.get("selection_metric"),
        "validation_auprc": cp.get("validation_auprc"),
        "event_feasible": cp.get("event_feasible"),
        "model_config": cp.get("model_config"),
        "training_config": cp.get("training_config"),
        "event_config": cp.get("event_config"),
        "op_metrics": op_metrics,
        "compute": state["compute"],
        "history": history.get("history") if history else None,
        "history_summary": (
            {k: history.get(k) for k in ("best_epoch", "best_val_loss", "best_selection_metric")}
            if history else None
        ),
        "history_compute": history.get("compute") if history else None,
    }


@app.post("/api/infer")
async def infer(
    file: UploadFile = File(...),
    model: str = Form(...),
    threshold: float | None = Form(None),
) -> dict[str, Any]:
    if not (file.filename or "").lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un .edf")

    state = _load_model(model)
    scaler_stats = state["checkpoint"]["scaler_stats"]
    nn_model = state["model"]
    device = state["device"]
    thr = float(threshold) if threshold is not None else state["threshold"]
    if not 0.0 <= thr <= 1.0:
        raise HTTPException(status_code=400, detail="El umbral debe estar entre 0 y 1")

    # Guardamos el EDF en un temporal (mne necesita una ruta de archivo).
    tmp = BASE_DIR / "inferences" / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename or "subida.edf").stem
    edf_path = tmp / f"{stem}_{uuid.uuid4().hex[:8]}.edf"
    edf_path.write_bytes(await file.read())

    try:
        t0 = time.perf_counter()

        data, reason = load_16_channels(edf_path, fs_expected=FS)
        if data is None:
            raise HTTPException(status_code=400, detail=f"No se pudo procesar el EDF: {reason}")

        filtered = filter_bandpass(data, fs=FS).astype(np.float32)
        del data

        windows, starts_samples = segment_windows(filtered)
        if windows.shape[0] == 0:
            raise HTTPException(status_code=400, detail="El EDF no contiene ventanas completas")

        windows = apply_scaler(windows, scaler_stats)
        starts = (starts_samples / FS).astype(np.float64)
        n_win = windows.shape[0]

        probs: list[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, n_win, BATCH_SIZE):
                batch = torch.from_numpy(windows[s:s + BATCH_SIZE]).to(device)
                logits = nn_model(batch).squeeze(-1)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        probabilities = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)

        ends = decision_times_from_starts(starts)
        preds = probabilities >= thr
        events = detect_event_intervals(starts, ends, preds, min_alarm_interval=EVENT_MIN_ALARM_INTERVAL)

        signal_payload = _downsample_signal(filtered)
        inference_s = time.perf_counter() - t0
    finally:
        edf_path.unlink(missing_ok=True)

    return {
        "model": model,
        "filename": file.filename,
        "threshold": thr,
        "duration_s": float(filtered.shape[1] / FS),
        "n_windows": int(n_win),
        "window_starts": starts.tolist(),
        "window_ends": ends.tolist(),
        "probabilities": probabilities.astype(np.float64).tolist(),
        "n_events": len(events),
        "events": [
            {"start_s": float(start), "end_s": float(end), "duration_s": float(end - start)}
            for start, end in events
        ],
        "signal": signal_payload,
        "compute": state["compute"],
        "inference_time_s": round(inference_s, 3),
        "windows_per_second": float(n_win / (filtered.shape[1] / FS)) if filtered.shape[1] else 0.0,
    }


def _downsample_signal(signal: np.ndarray, target_samples: int = 8000) -> dict[str, Any]:
    """Reduce la señal para poder graficarla en el navegador sin saturar el JSON."""
    n = signal.shape[1]
    factor = max(1, int(np.ceil(n / target_samples)))
    down = signal[:, ::factor]
    return {
        "channels": list(CHANNELS_TUEV),
        "sfreq": float(FS / factor),
        "n_samples": int(down.shape[1]),
        "data": down.astype(np.float32).tolist(),
    }


def _json_safe(obj: Any) -> Any:
    """Convierte numpy/scalars a tipos JSON para no romper FastAPI al serializar."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


# Interfaz web
if (FRONTEND_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "detail": "Frontend no compilado. Ejecutá 'npm install' y 'npm run build' dentro de frontend/."
        }


def main() -> None:
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
