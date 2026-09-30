"""
Entrenamiento de la CNN-1D (en mi google colab pro).

Junta `data.py` con `model.py` y entrena la red para clasificar cada ventana de EEG.

    python -m src.train                    # corrida completa (CPU o GPU según haya)
    python -m src.train --limit-files 6    # smoke de entrenamiento (pocos EDFs)

  1. Carga split.json y scaler_stats.npz
  2. Arma los DataLoaders: train balanceado y shuffleado (BalancedEpochDataset) y val natural (sin tocar la distribución).
  3. Instancia la CNN, el optimizador AdamW y el loss BCEWithLogits con pos_weight.
  4. Corre las épocas: por cada batch -> ventanas -> red -> loss -> backward -> paso del optimizador. Al final de cada época evalúa en validación.
  5. Early stopping: si el loss de val no mejora en PATIENCE épocas, corta.
  6. Guarda el mejor modelo cuando algún punto de operación cumple el objetivo
     clínico, junto con su configuración y stats de escalado.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    BATCH_SIZE,
    CONV_CHANNELS,
    CONV_KERNELS,
    DATASET_DIR,
    DROPOUT,
    EPOCHS,
    EVENT_MAX_LATENCY,
    EVENT_MIN_ALARM_INTERVAL,
    WINDOW_RANGE_FOR_EVENT,
    POSITIVES_FOR_EVENT,
    FC_UNITS,
    GRAD_CLIP,
    LEARNING_RATE,
    LR_MIN,
    MIN_EVENT_SENSITIVITY,
    MODELS_DIR,
    N_CHANNELS,
    NEG_POS_RATIO,
    NOISE_STD,
    NUM_WORKERS,
    PATIENCE,
    SCALER_STATS_FILE,
    SEED,
    SPLIT_FILE,
    THRESHOLD,
    WEIGHT_DECAY,
)
from src.data import build_splits_dataloaders, compute_positive_weight
from src.metrics import binary_metrics, event_metrics, select_event_operating_point
from src.model import SeizureCNN
from src.preprocessing import load_scaler_stats
from src.protocol import (
    PROTOCOL_VERSION,
    THRESHOLD_GRID,
    compatibility_config,
    scaler_id,
    split_id,
)



def set_seed(seed: int) -> None:
    """
    Fija TODO el azar del entrenamiento para que sea reproducible (dos corridas con el mismo SEED dan los mismos valores)
      - random / numpy: shuffle de índices y submuestreos.
      - torch.manual_seed: inicialización de pesos y orden de batches.
      - cudnn determinístico: en GPU, que las convoluciones den siempre igual.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic=True fuerza algoritmos reproducibles (algo más lentos);
        # benchmark=False evita que cuDNN elija kernels "a la carrera".
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str | None) -> str:
    if requested:
        if requested not in {"cpu", "cuda"}:
            raise ValueError("device debe ser 'cpu' o 'cuda'")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó cuda, pero CUDA no está disponible")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# una época de entrenamiento y la evaluación en validación
def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: str) -> float:
    """
    Por batch:
      1. mueve ventanas y labels al device
      2. `model(x)` = forward -> logits (B, 1);
      3. `criterion` (BCEWithLogits + pos_weight) compara logits vs. labels;
      4. `loss.backward()` = backpropagation: calcula el gradiente de cada peso;
      5. `optimizer.step()` = AdamW actualiza los pesos siguiendo esos gradientes.

    Devuelve el loss PROMEDIO de la época (suma pesada por tamaño de batch).
    """
    model.train()  # activa Dropout y las stats de batch del BatchNorm (modo entrenar)

    total_loss = 0.0
    n_seen = 0  # contador de ventanas vistas (para promediar bien el loss)

    for windows, labels in loader:
        windows = windows.to(device)   # (B, 16, 1310)
        labels = labels.to(device)     # (B,) etiquetas 0/1

        optimizer.zero_grad()          # resetea gradientes acumulados del batch anterior
       #Estas dos funciones escriben sus operaciones en el grafo de gradientes (se arma uno en cada batch), y los resultados intermedios se guardan en memoria
       #cada nodo del grafo guarda q operación lo creó (mul, add, sigmoid, log), de q tensores vino (Los inputs), los valores numéticos de los resultados (las activaciones q salieron, no los gradientes)
       #guardo los resultados parciales pq la derivada de una operación depende del valor de su input, por ende no puedo calcular el gradiente sin eso
        logits = model(windows)        # (B, 1) — el logit crudo, sin sigmoide
        loss = criterion(logits.squeeze(-1), labels)  # calculo el loss! BCEWithLogits espera logits (B,) y target (B,)
       #acá recorro ese grafo de gradientes en orden inverso, se calculan los gradientes, y libero el grafo.
        loss.backward()                # calculo todos los gradientes de todos los pesos usando backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # corta gradientes gigantes (estabilidad)
        optimizer.step()               # ACÁ el optimizador aplica la fórmula de AdamW y actualiza los pesos posta

        # loss.item() es el loss PROMEDIO del batch; lo multiplico por el tamaño para acumular la suma total y después promediar sobre todas las ventanas.
        total_loss += loss.item() * windows.size(0)
        n_seen += windows.size(0)

    if n_seen == 0:
        raise RuntimeError("El DataLoader de entrenamiento no contiene ventanas.")
    return total_loss / n_seen


@torch.no_grad()  # no construye grafo de gradientes para ahorra memoria, total no actualizo los pesos
def evaluate(model: nn.Module, loader, criterion, device: str) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evalúa el modelo sobre un DataLoader (validación o test)
    Devuelve:
      - val_loss (promedio, con el criterion SIN pos_weight);
      - probs: todas las probabilidades
      - labels: todas las etiquetas reales
    """
    model.eval()  # apaga Dropout y usa running_mean/var del BatchNorm (modo evaluar)

    total_loss = 0.0
    n_seen = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
#tensor de forma (64, 16, 1310) para windows y (64,) par alabels
    for windows, labels in loader:
        windows = windows.to(device)
        labels = labels.to(device)
        logits = model(windows)                  # (B, 1)
        loss = criterion(logits.squeeze(-1), labels)  # loss sin pos_weight (natural)
        total_loss += loss.item() * windows.size(0)
        n_seen += windows.size(0)
        
        # le saco la últma dimensión a logits, pasa de (64, 1) a (64). Convierto cada uno a prob con la sigmoide, lo muevo a la CPU, y lo vot acynulando
        all_probs.append(torch.sigmoid(logits.squeeze(-1)).cpu())
        all_labels.append(labels.cpu()) #ídem

    if n_seen == 0:
        raise RuntimeError("El DataLoader de evaluación no contiene ventanas.")

    # concatena todos los tensores de cada batch y me deja un solo tensor.
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    #total_loss/n_seen = VAL LOSS
    return total_loss / n_seen, probs, labels


def save_checkpoint(path: Path, model: nn.Module, scaler_stats: dict, *,
                    epoch: int, best_val_loss: float, pos_weight: float, seed: int,
                    neg_pos_ratio: float = NEG_POS_RATIO,
                    op_threshold: float = THRESHOLD, op_sensibility: float = 0.0,
                    op_fdr_per_hour: float = float("inf"),
                    min_event_sensitivity: float = MIN_EVENT_SENSITIVITY,
                    noise_std: float = NOISE_STD,
                    experiment_config: dict | None = None) -> None:
    """
    Guarda lo necesario para re-usar el modelo sin re-entrenar.

    El payload contiene únicamente tensores y tipos primitivos. Esto permite
    cargarlo con ``torch.load(..., weights_only=True)`` sin deserializar
    objetos pickle arbitrarios.

      - model_state_dict: los valores de TODOS los pesos (la "memoria" aprendida);
      - config del modelo: listas de canales/kernels, fc_units, dropout, etc. (para reconstruir la arquitectura idéntica al cargar);
      - scaler_stats: mediana/IQR por canal del RobustScaler (para que inference escale las ventanas nuevas EXACTAMENTE igual que en train);
      - metadata: época, best_val_loss, pos_weight, neg_pos_ratio, seed (auditoría);
      - op_*: punto de operación elegido en val a nivel EVENTO (umbral, sensibilidad
        de crisis y tasa de falsas alarmas por hora).

    Se guarda en disco el momento en que el criterio clínico (menor tasa de falsas alarmas/h con sens
    de crisis >= objetivo) fue el mejor, no el loss crudo.
    """
    # Crea la carpeta destino, por ejemplo models/ratio3_pw1.pt.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Agrupa pesos, scaler y metadata en un único archivo portable.
    checkpoint = {
        "format_version": 3,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": N_CHANNELS,
            "conv_channels": list(CONV_CHANNELS),
            "conv_kernels": list(CONV_KERNELS),
            "fc_units": FC_UNITS,
            "dropout": DROPOUT,
        },
        # Guarda el scaler del train exacto que produjo este modelo.
        "scaler_stats": {
            "channels": scaler_stats["channels"],
            "median": torch.as_tensor(scaler_stats["median"], dtype=torch.float64).cpu(),
            "iqr": torch.as_tensor(scaler_stats["iqr"], dtype=torch.float64).cpu(),
            "protocol_version": scaler_stats.get("protocol_version"),
            "split_id": scaler_stats.get("split_id"),
            "scaler_id": scaler_stats.get("scaler_id"),
            "preprocessing_config": scaler_stats.get("preprocessing_config"),
        },
        "threshold": THRESHOLD,
        "op_threshold": op_threshold,
        "op_sensibility": op_sensibility,
        "op_fdr_per_hour": op_fdr_per_hour,
        # Guarda las reglas que convierten predicciones de ventana en alarmas.
        "event_config": {
            "n_within": POSITIVES_FOR_EVENT,
            "n_window": WINDOW_RANGE_FOR_EVENT,
            "min_alarm_interval": EVENT_MIN_ALARM_INTERVAL,
            "max_latency": EVENT_MAX_LATENCY,
            "min_event_sensitivity": min_event_sensitivity,
        },
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "pos_weight": pos_weight,
        "neg_pos_ratio": neg_pos_ratio,
        "noise_std": noise_std,
        "protocol_version": PROTOCOL_VERSION,
        "split_id": experiment_config.get("split_id") if experiment_config else None,
        "scaler_id": experiment_config.get("scaler_id") if experiment_config else None,
        "experiment_config": experiment_config,
        "seed": seed,
    }
    # Serializa el checkpoint; desde este momento la ruta representa un modelo válido.
    torch.save(checkpoint, path)


def parse_args() -> argparse.Namespace:
    # Los flags permiten ajustar la corrida sin tocar config.py (para Colab, qpaso --data-dir y --split-file con las rutas de Drive)
    parser = argparse.ArgumentParser(description="Entrenamiento CNN-1D")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--scaler-stats", type=str, default=str(SCALER_STATS_FILE))
    parser.add_argument("--out", type=str, default=str(MODELS_DIR / "best.pt"),  help="Ruta del checkpoint a guardar.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--neg-pos-ratio", type=float, default=NEG_POS_RATIO)
    parser.add_argument("--noise-std", type=float, default=NOISE_STD,
                        help="Desvío del ruido gaussiano sumado a las ventanas de crisis (clase minoritaria). 0 lo desactiva.")
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="Peso de la clase positiva en el loss. Default: igual a NEG_POS_RATIO.")
    parser.add_argument("--min-event-sensitivity", type=float, default=MIN_EVENT_SENSITIVITY,
                        help="Sensibilidad objetivo a nivel EVENTO (crisis): elige el umbral con menor tasa de falsas alarmas/h que la alcance.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None, help="'cuda' o 'cpu'.")
    parser.add_argument("--limit-files", type=int, default=None, help="Limitar a N EDFs por split (smoke-test rápido).")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Carpeta espejo (ej. Google Drive montado) donde copiar el checkpoint y el historial cada vez que mejoran. Si Colab se cierra, el mejor modelo queda acá.")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs debe ser mayor que cero")
    if args.batch_size <= 0:
        parser.error("--batch-size debe ser mayor que cero")
    if args.lr <= 0:
        parser.error("--lr debe ser mayor que cero")
    if args.weight_decay < 0:
        parser.error("--weight-decay no puede ser negativo")
    if args.patience <= 0:
        parser.error("--patience debe ser mayor que cero")
    if args.num_workers < 0:
        parser.error("--num-workers no puede ser negativo")
    if args.neg_pos_ratio <= 0:
        parser.error("--neg-pos-ratio debe ser mayor que cero")
    if args.noise_std < 0:
        parser.error("--noise-std no puede ser negativo")
    if not math.isfinite(args.noise_std):
        parser.error("--noise-std debe ser finito")
    if args.pos_weight is not None and args.pos_weight <= 0:
        parser.error("--pos-weight debe ser mayor que cero")
    for name, value in (
        ("--min-event-sensitivity", args.min_event_sensitivity),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} debe estar entre 0 y 1")
    if args.limit_files is not None and args.limit_files <= 0:
        parser.error("--limit-files debe ser mayor que cero")
    return args


def backup_file(src: Path, backup_dir: str | None) -> None:
    """Copia `src` a `backup_dir` (creándola si hace falta), sin romper el
    entrenamiento si la copia falla (Drive montado puede cortarse)."""
    if not backup_dir:
        return
    try:
        dst_dir = Path(backup_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        print(f"    [backup] copiado a {dst}")
    except Exception as exc:  # no queremos que una falla de Drive mate la corrida
        print(f"    [backup][WARN] no se pudo copiar {src.name} a {backup_dir}: {exc}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}  (semilla={args.seed})")

    # Carga del split y de las stats del escalador
    # Carga el reparto de pacientes usado para esta corrida.
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    scaler_stats = load_scaler_stats(Path(args.scaler_stats))
    if scaler_stats is None:
        sys.exit(
            f"No existe el scaler indicado: {args.scaler_stats}. "
            "Corré primero python -m src.preprocessing --compute-stats."
        )
    expected_split_id = split_id(split)
    expected_scaler_id = scaler_id(scaler_stats)
    if scaler_stats.get("split_id") != expected_split_id:
        sys.exit(
            "El scaler no corresponde al split indicado "
            f"(esperado {expected_split_id}, recibido {scaler_stats.get('split_id')}). "
            "Regenerá scaler_stats.npz con ese split."
        )
    if scaler_stats.get("scaler_id") != expected_scaler_id:
        sys.exit("El scaler_stats.npz está internamente inconsistente; regeneralo.")

    loaders = build_splits_dataloaders(args.data_dir, split, scaler_stats, batch_size=args.batch_size,
        num_workers=args.num_workers, neg_pos_ratio=args.neg_pos_ratio, seed=args.seed,
        noise_std=args.noise_std, limit_files=args.limit_files,)
    #solo uso esto para tener info del dataset e imprimirlo. Sino uso el dataloader
    train_ds = loaders["train"]["dataset"]
    train_loader = loaders["train"]["dataloader"]
    val_ds = loaders["val"]["dataset"]
    val_loader = loaders["val"]["dataloader"]

    audit = {
        name: bundle["dataset"].audit_summary()
        for name, bundle in loaders.items()
    }
    for name, summary in audit.items():
        print(
            f"{name}: {summary['valid_files']}/{summary['input_files']} archivos válidos, "
            f"{summary['windows']} ventanas ({summary['positive_windows']} positivas)"
        )
        if summary["skipped_files"]:
            print(f"  [WARN] {name}: {len(summary['skipped_files'])} archivos descartados")
    audit_path = Path(args.out).with_name(Path(args.out).stem + ".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    backup_file(audit_path, args.backup_dir)

    if len(train_ds) == 0:
        sys.exit("El split TRAIN no contiene ventanas procesables.")
    if train_ds.n_positive == 0:
        sys.exit("El split TRAIN no contiene ventanas positivas.")
    if len(val_ds) == 0:
        sys.exit("El split VAL no contiene ventanas procesables.")
    if val_ds.n_positive == 0:
        sys.exit("El split VAL no contiene ventanas positivas para seleccionar el punto de operación.")

    pos_weight = args.pos_weight if args.pos_weight is not None else compute_positive_weight(train_ds, neg_pos_ratio=args.neg_pos_ratio)

    model_config = {
        "in_channels": N_CHANNELS,
        "conv_channels": list(CONV_CHANNELS),
        "conv_kernels": list(CONV_KERNELS),
        "fc_units": FC_UNITS,
        "dropout": DROPOUT,
    }
    event_config = {
        "n_within": POSITIVES_FOR_EVENT,
        "n_window": WINDOW_RANGE_FOR_EVENT,
        "min_alarm_interval": EVENT_MIN_ALARM_INTERVAL,
        "max_latency": EVENT_MAX_LATENCY,
    }
    experiment_config = compatibility_config(
        split,
        scaler_stats,
        model_config=model_config,
        event_config=event_config,
        min_event_sensitivity=args.min_event_sensitivity,
        threshold_grid=THRESHOLD_GRID,
        seed=args.seed,
    )
    experiment_config["scenario"] = {
        "neg_pos_ratio": float(args.neg_pos_ratio),
        "pos_weight": float(pos_weight),
        "noise_std": float(args.noise_std),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
    }

    print(f"\nConfiguración de entrenamiento")
    print(f"Train: {len(train_ds)} ventanas ({train_ds.n_positive} positivas, "
          f"{len(train_ds) - train_ds.n_positive} negativas)")
    print(f"pos_weight: {pos_weight:.1f}  |  NEG_POS_RATIO: {args.neg_pos_ratio}")
    print(f"split_id: {experiment_config['split_id']}  |  scaler_id: {experiment_config['scaler_id']}")
    print(f"batch_size={args.batch_size}  lr={args.lr}  weight_decay={args.weight_decay}  "
          f"epochs={args.epochs}  patience={args.patience}")

    model = SeizureCNN().to(device)

    # AdamW es Adam + weight decay DESACOPLADO (hat dos formas de meterlo: acoplado, agreganfo el castigo al gradiente, o el desacoplado)
    # El adam solito usa momentum y tasa adaptativa. El weight decay sirve para evitar el overfitting. Penaliza pesos
    # grandes (en vada paso achico cada peso hacia 0) haciendolo DIRECTAMENTE en el peso, sin mezclarse con el momento adaptativo (regularización L2, en donde L2= cuadrado de los pesos. Entoncees
    # mi loss queda L_total= L + (weight_decay/2) * sumatoria de(weights^2))
    #entonces, acá creo el optimizador AdamW, le paso la lista de todos los pesos, el learning rate inicial, y el weight decay de cuánto achico los pesos.
    #NO HACE NADA TODAVÍA, solo lo creo!
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler coseno: el lr baja de `args.lr` a LR_MIN a lo largo del entrenamiento.
    # Estabiliza las épocas finales (que es donde el modelo me oscila más)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=LR_MIN)

   #La BCE mide el error de mi predicción con la fórmula L = −[ y·log(p) + (1−y)·log(1−p) ], con p la probabilidad de crisis q predije e y=0 o y=1 según si hay o no
   #la probabilidad p es básicamente la sigmoide de mis logits (los logits son la salida real de mi cnn, yo con la sigmoide q es 1/(1+e^(-logit)) lo llevo a p entre 0 y 1)
   #el BCE with logits me hace la transformación a probabilidad internamente así no la hago yop
   #toma el promedio de los errores individuales de cada ventana del batch
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    # Loss de VALIDACIÓN. Independientemente si en la función de pérdida usé o no el pos_weight, acá NO lo uso. Necesito saber si la red generaliza, de forma estable
    val_criterion = nn.BCEWithLogitsLoss()

    # Loop de entrenamiento con early stopping si no mejora en X cantidad de épocas.
    # El criterio de "mejor" es CLÍNICO y a nivel EVENTO, no el loss crudo: entre los
    # umbrales cuya sensibilidad de CRISIS >= MIN_EVENT_SENSITIVITY se elige el de
    # MENOR tasa de falsas alarmas por hora.
    best_val_loss = float("inf")   # solo de referencia (se sigue guardando en el log)
    best_fdr = float("inf")
    best_op_threshold = THRESHOLD  # umbral del mejor punto de operación
    best_op_sensibility = 0.0      # sensibilidad de crisis de ese punto
    best_epoch = 0
    # Permite distinguir una corrida que no encontró un punto clínicamente válido
    # de una corrida que sí produjo un checkpoint nuevo.
    checkpoint_saved = False
    patience_left = args.patience
    start = time.time()
    history: list[dict] = []  # métricas por época (para las curvas de loss/accuracy en la tesis)

    print(f"\n=== Entrenamiento ({args.epochs} épocas máx.) ===")
    print(f"Criterio de selección: menor tasa de falsas alarmas/h con sens de crisis >= {args.min_event_sensitivity}")
    print(f"La tabla de abajo es a nivel VENTANA @ umbral {THRESHOLD} (solo referencia).")
    print(f"{'Ep':>3} | {'train_loss':>12} | {'val_loss':>12} | {'sens':>10} | {'spec':>10} | "
          f"{'fpr':>10} | {'fp/h':>10} | {'acc':>10} | {'tiempo':>8}")

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        #TRAIN LOSS SE CALCULA SOBRE LOS DATOS DE TRAIN, Q ESTÁN BALANCEADOS. SE MIDE MIENTRAS EL MODELO SE ACTUALIZA
        #VAL LOSS SE CALCULA SOBRE LOS PACIENTES DE VALIDACIÓN (NO LOS DE TEST TODAVÍA), ENTONCES NO TIENEN UNDERSAMPLING, NI POS WEIGHT,
        #NI DROPOUT. O SEA VAL LOSS ME DICE Q TAN BIEN GENERALIZÓ EN PROMEDIO FRENTE A DATOS CON LOS Q NO ENTRENÉ
        train_loss = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        val_loss, probs, labels = evaluate(model, val_loader, val_criterion, device)

        # métricas por ventana a umbral fijo (para la tabla) + curva sens <-> fp/h (referencia)
        m = binary_metrics(
            probs,
            labels,
            THRESHOLD,
            file_ids=val_ds.file_ids,
            local_ids=val_ds.local_ids,
        )

        # --- PUNTO DE OPERACIÓN A NIVEL EVENTO (criterio clínico de selección) ---
            # Entre los umbrales cuya sensibilidad de CRISIS >= objetivo, el de MENOR tasa de falsas alarmas/h.
        op_ev, op_threshold = select_event_operating_point(
            probs, labels,
            file_ids=val_ds.file_ids, local_ids=val_ds.local_ids,
            valid_files=val_ds.valid_files, annotations=val_ds.annotations,
            min_sensibility=args.min_event_sensitivity,
            n_within=POSITIVES_FOR_EVENT, n_window=WINDOW_RANGE_FOR_EVENT,
            min_alarm_interval=EVENT_MIN_ALARM_INTERVAL, max_latency=EVENT_MAX_LATENCY,
        )

        if op_ev is not None:
            ev = op_ev
            op_fdr = op_ev["event_false_alarms_per_hour"]
            op_sensibility = op_ev["sensibility"]
            warn_msg = None
        else:
            # ningún umbral alcanza el objetivo de sensibilidad de CRISIS -> fallback:
            # mostramos el umbral con la MÁXIMA sensibilidad de crisis alcanzable, y
            # marcamos la tasa=inf para que este modelo NO pueda ser seleccionado.
            ev = None
            for _t in THRESHOLD_GRID:
                _e = event_metrics(probs, labels,
                                   file_ids=val_ds.file_ids, local_ids=val_ds.local_ids,
                                   valid_files=val_ds.valid_files, annotations=val_ds.annotations,
                                   threshold=_t,
                                   n_within=POSITIVES_FOR_EVENT, n_window=WINDOW_RANGE_FOR_EVENT,
                                   min_alarm_interval=EVENT_MIN_ALARM_INTERVAL,
                                   max_latency=EVENT_MAX_LATENCY)
                if ev is None or _e["sensibility"] > ev["sensibility"]:
                    ev = _e
                    op_threshold = float(_t)
            op_fdr = float("inf")
            op_sensibility = ev["sensibility"]
            warn_msg = (f"ningún umbral alcanzó sens de crisis >= {args.min_event_sensitivity} "
                        f"(máximo logrado: {op_sensibility:.3f} @ umbral {op_threshold:.2f})")

        elapsed_time_for_epoch = time.time() - t_epoch

        # --- log por época ---
        print(f"{epoch:>3} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
              f"{m['sensibility']:>10.4f} | {m['specificity']:>10.4f} | {m['false_positive_rate']:>10.4f} | "
              f"{m['false_positive_per_hour']:>10.4f} | {m['accuracy']:>10.4f} | {elapsed_time_for_epoch:>7.1f}s")
        print(f"      ventana @ {THRESHOLD:.2f} : sens={m['sensibility']:.3f}  spec={m['specificity']:.3f}  "
              f"fp/h={m['false_positive_per_hour']:.1f}  F1={m['f1']:.3f}  MCC={m['mcc']:.3f}")
        print(f"      evento  @ {op_threshold:.2f} : sens={op_sensibility:.3f} "
              f"({ev['n_detected']}/{ev['n_seizures']} crisis)  falsas alarmas={ev['event_false_alarms_per_hour']:.1f}/h  "
              f"latencia={ev['latency_mean']:.1f}s")
        if warn_msg is not None:
            print(f"      [WARN] {warn_msg}")
        print()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "sensibility": m["sensibility"],
            "specificity": m["specificity"],
            "false_positive_rate": m["false_positive_rate"],
            "false_positive_per_hour": m["false_positive_per_hour"],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "mcc": m["mcc"],
            "op_threshold": op_threshold,
            "op_sensibility": op_sensibility,
            "op_fdr_per_hour": op_fdr,
            "event_sensibility": ev["sensibility"],
            "event_false_alarms_per_hour": ev["event_false_alarms_per_hour"],
            "event_fdr_per_hour": ev["event_false_alarms_per_hour"],
            "event_latency_mean": ev["latency_mean"],
            "time_s": elapsed_time_for_epoch,
        })

        # si el punto de operación EVENTO mejoró (menor tasa de falsas alarmas/h con sens >= objetivo)
        improved = op_ev is not None and op_fdr < best_fdr
        if improved:
            best_fdr = op_fdr
            best_val_loss = val_loss
            best_op_threshold = op_threshold
            best_op_sensibility = op_sensibility
            best_epoch = epoch
            patience_left = args.patience
            save_checkpoint(
                Path(args.out), model, scaler_stats,
                epoch=epoch, best_val_loss=best_val_loss,
                pos_weight=pos_weight, neg_pos_ratio=args.neg_pos_ratio, seed=args.seed,
                op_threshold=op_threshold, op_sensibility=op_sensibility,
                op_fdr_per_hour=op_fdr,
                min_event_sensitivity=args.min_event_sensitivity,
                noise_std=args.noise_std,
                experiment_config=experiment_config,
            )
            # Solo se marca como guardado después de que torch.save terminó bien.
            checkpoint_saved = True
            backup_file(Path(args.out), args.backup_dir)
            print(f"    -> mejor punto de operación (evento): sens={op_sensibility:.3f}, "
                  f"falsas alarmas={op_fdr:.3f}/h @ umbral {op_threshold:.2f}. Checkpoint en {args.out}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping: no mejoró la tasa de falsas alarmas/h (sens de crisis >= {args.min_event_sensitivity}) "
                      f"en {args.patience} épocas. Cortando.")
                break

        scheduler.step()  # baja el lr un escalón (coseno), sin importar si mejoró o no

    total_time = time.time() - start
    print(f"\n=== Fin del entrenamiento ===")
    print(f"Mejor época (por menor tasa de falsas alarmas/h con sens de crisis >= {args.min_event_sensitivity}): {best_epoch}")
    print(f"  val_loss de esa época: {best_val_loss:.6f}")
    print(f"  punto de operación: umbral={best_op_threshold:.2f}  sens={best_op_sensibility:.3f}  falsas alarmas={best_fdr:.3f}/h")
    print(f"Tiempo total: {total_time / 60:.1f} min")
    # No se imprime una ruta si esta corrida no creó un checkpoint.
    if checkpoint_saved:
        print(f"Checkpoint: {args.out}")
    else:
        print("Checkpoint: no guardado (ningún punto de operación alcanzó el objetivo).")

    # Guarda el historial por época (curvas loss/accuracy para la tesis). Se persiste
    # junto al checkpoint como <nombre>.history.json para que el notebook lo pueda leer sin re-entrenar.
    history_path = Path(args.out).with_name(Path(args.out).stem + ".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    # El historial se guarda incluso si no hubo checkpoint, para auditar el motivo.
    history_path.write_text(json.dumps({
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_fdr_per_hour": best_fdr,
        "best_event_false_alarms_per_hour": best_fdr,
        "best_op_threshold": best_op_threshold,
        "best_op_sensibility": best_op_sensibility,
        "min_event_sensitivity": args.min_event_sensitivity,
        "threshold": THRESHOLD,
        "pos_weight": pos_weight,
        "neg_pos_ratio": args.neg_pos_ratio,
        "noise_std": args.noise_std,
        "protocol_version": PROTOCOL_VERSION,
        "split_id": experiment_config["split_id"],
        "scaler_id": experiment_config["scaler_id"],
        "experiment_config": experiment_config,
        "checkpoint_saved": checkpoint_saved,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"Historial de entrenamiento: {history_path}")
    backup_file(history_path, args.backup_dir)


if __name__ == "__main__":
    main()
