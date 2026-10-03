"""
Entrenamiento de la CNN-1D (en mi google colab pro).

Junta `data.py` con `model.py` y entrena la red para clasificar cada ventana de EEG.

    python -m src.train                    # corrida completa (CPU o GPU según haya)
    python -m src.train --limit-files 6    # smoke de entrenamiento (pocos EDFs)

  1. Carga split.json y scaler_stats.npz
  2. Arma los DataLoaders: train balanceado y shuffleado (BalancedEpochDataset) y val natural (sin tocar la distribución).
   3. Instancia la CNN, el optimizador AdamW y BCEWithLogits o Focal Loss según CLI.
   4. Corre las épocas: por cada batch -> ventanas -> red -> loss -> backward -> paso del optimizador. Al final de cada época evalúa en validación.
   5. Early stopping: si el loss de val no mejora en PATIENCE épocas, corta.
   6. Guarda el mejor checkpoint clínico según validación (sensibilidad de evento
      bajo el techo de falsas alarmas), junto con la configuración y el umbral.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.config import (
    BATCH_SIZE,
    CONV_CHANNELS,
    CONV_KERNELS,
    DATASET_DIR,
    DROPOUT,
    EVENT_MAX_LATENCY,
    EVENT_MIN_ALARM_INTERVAL,
    EVENT_PRE_ONSET_TOLERANCE,
    EPOCHS,
    FC_UNITS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LEARNING_RATE,
    LOSS,
    MAX_FALSE_ALARMS_PER_HOUR,
    MODELS_DIR,
    N_CHANNELS,
    NEG_POS_RATIO,
    NOISE_STD,
    NUM_WORKERS,
    PATIENCE,
    POSITIVES_FOR_EVENT,
    SCALER_STATS_FILE,
    SEED,
    SPLIT_FILE,
    THRESHOLD,
    WEIGHT_DECAY,
    WINDOW_RANGE_FOR_EVENT,
    WIN_SAMPLES,
)
from src.compute_metrics import (
    count_parameters,
    measure_flops,
    measure_inference_latency,
    peak_ram_mb,
)
from src.data import build_splits_dataloaders, compute_positive_weight
from src.metrics import (
    binary_metrics,
    rank_metrics,
    select_event_operating_point,
    select_lowest_false_alarm_point,
)
from src.model import SeizureCNN
from src.preprocessing import load_scaler_stats
from src.protocol import (
    THRESHOLD_GRID,
    compatibility_config,
    preprocessing_config,
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
        optimizer.step()               # ACÁ el optimizador aplica la fórmula de AdamW y actualiza los pesos posta

        # loss.item() es el loss PROMEDIO del batch; lo multiplico por el tamaño para acumular la suma total y después promediar sobre todas las ventanas.
        total_loss += loss.item() * windows.size(0)
        n_seen += windows.size(0)

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

    #concatena todos los tensores de cada batch y me deja un solo tensor. 
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    #total_loss/n_seen = VAL LOSS
    return total_loss / n_seen, probs, labels


class FocalLoss(nn.Module):
    """Focal Loss binario (Lin et al., 2017) sobre logits crudos.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weightea los ejemplos bien clasificados (fáciles) para que el gradiente
    se concentre en los difíciles. `alpha` es el factor de la clase positiva;
    con alpha=0.25, positivos pesan 0.25 y negativos 0.75, siguiendo la
    convención del paper original. El muestreo y los hiperparámetros se comparan
    en validación, porque el factor focal no garantiza por sí solo mejor recall.
    """
    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    scaler_stats: dict,
    *,
    split: dict,
    epoch: int,
    best_val_loss: float,
    checkpoint_val_loss: float,
    pos_weight: float,
    seed: int,
    event_config: dict,
    op_threshold: float | None,
    op_metrics: dict | None,
    event_feasible: bool,
    validation_auprc: float | None,
    selection_metric: str,
    training_config: dict,
    loss: str = LOSS,
    focal_gamma: float = FOCAL_GAMMA,
    focal_alpha: float = FOCAL_ALPHA,
) -> None:
    """
    Guarda lo necesario para re-usar el modelo sin re-entrenar:

      - model_state_dict: los valores de TODOS los pesos (la "memoria" aprendida);
      - config del modelo: listas de canales/kernels, fc_units, dropout, etc. (para reconstruir la arquitectura idéntica al cargar);
      - scaler_stats: mediana/IQR por canal del RobustScaler (para que inference escale las ventanas nuevas EXACTAMENTE igual que en train);
       - metadata: época, losses de selección, pos_weight y seed (auditoría).

    El checkpoint guarda el momento seleccionado por el criterio clínico de
    validación. `best_val_loss` es el mínimo global usado para early stopping;
    `checkpoint_val_loss` es el loss de la época clínica seleccionada.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = {
        "in_channels": N_CHANNELS,
        "conv_channels": list(CONV_CHANNELS),
        "conv_kernels": list(CONV_KERNELS),
        "fc_units": FC_UNITS,
        "dropout": DROPOUT,
    }
    compatibility = compatibility_config(
        split,
        scaler_stats,
        model_config=model_config,
        event_config=event_config,
        max_false_alarms_per_hour=MAX_FALSE_ALARMS_PER_HOUR,
        threshold_grid=THRESHOLD_GRID,
        seed=seed,
    )
    checkpoint = {
        "format_version": 4,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "scaler_stats": {
            "scaler": scaler_stats.get("scaler", "robust"),
            "channels": scaler_stats["channels"],
            # Se guarda como tensor para que el checkpoint se pueda cargar con
            # weights_only=True (sin deserializar objetos numpy vía pickle).
            "median": torch.as_tensor(scaler_stats["median"], dtype=torch.float64).cpu(),
            "iqr": torch.as_tensor(scaler_stats["iqr"], dtype=torch.float64).cpu(),
            "protocol_version": scaler_stats.get("protocol_version"),
            "split_id": scaler_stats.get("split_id") or split_id(split),
            "scaler_id": scaler_id(scaler_stats),
            "preprocessing_config": scaler_stats.get("preprocessing_config") or preprocessing_config(),
        },
        "compatibility": compatibility,
        "protocol_version": compatibility["protocol_version"],
        "split_id": compatibility["split_id"],
        "scaler_id": compatibility["scaler_id"],
        "preprocessing_config": compatibility["preprocessing"],
        "event_config": event_config,
        "threshold_grid": compatibility["threshold_grid"],
        "threshold": float(op_threshold if op_threshold is not None else THRESHOLD),
        "op_threshold": op_threshold,
        "op_metrics": op_metrics,
        "event_feasible": event_feasible,
        "validation_auprc": validation_auprc,
        "selection_metric": selection_metric,
        "training_config": training_config,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_val_loss": checkpoint_val_loss,
        "pos_weight": pos_weight,
        "loss": loss,
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "seed": seed,
    }
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
                        help="Desvío del ruido agregado solo a ventanas positivas durante train.")
    parser.add_argument("--loss", type=str, default=LOSS, choices=["bce", "focal"],
                        help="Función de pérdida de entrenamiento: 'bce' (BCE+pos_weight) o 'focal' (Focal Loss).")
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA, help="Gamma del Focal Loss.")
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_ALPHA, help="Alpha (peso de la clase crisis) del Focal Loss.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None, help="'cuda' o 'cpu'.")
    parser.add_argument("--limit-files", type=int, default=None, help="Limitar a N EDFs por split (smoke-test rápido).")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs debe ser mayor que cero")
    if args.batch_size <= 0:
        parser.error("--batch-size debe ser mayor que cero")
    if args.patience <= 0:
        parser.error("--patience debe ser mayor que cero")
    if args.neg_pos_ratio < 0:
        parser.error("--neg-pos-ratio no puede ser negativo")
    if args.noise_std < 0:
        parser.error("--noise-std no puede ser negativo")
    if args.focal_gamma < 0:
        parser.error("--focal-gamma no puede ser negativo")
    if not 0.0 < args.focal_alpha <= 1.0:
        parser.error("--focal-alpha debe estar en (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}  (semilla={args.seed})")

    # Carga del split y de las stats del escalador
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    scaler_stats = load_scaler_stats(Path(args.scaler_stats))
    if scaler_stats is None:
        sys.exit("No existe scaler_stats.npz. Corré primero:\n""    python -m src.preprocessing --compute-stats\n")
    current_split_id = split_id(split)
    if not scaler_stats.get("split_id"):
        sys.exit("El scaler_stats.npz no contiene split_id. Regeneralo con el split actual.")
    if scaler_stats["split_id"] != current_split_id:
        sys.exit("El scaler_stats.npz no corresponde al split solicitado. Regeneralo sobre este split.")
    if not scaler_stats.get("scaler_id"):
        sys.exit("El scaler_stats.npz no contiene scaler_id. Regeneralo.")
    if scaler_stats["scaler_id"] != scaler_id(scaler_stats):
        sys.exit("El scaler_stats.npz tiene un scaler_id inconsistente. Regeneralo.")
    if scaler_stats.get("preprocessing_config") != preprocessing_config():
        sys.exit("El scaler_stats.npz fue creado con otro preprocesamiento. Regeneralo.")

    loaders = build_splits_dataloaders(args.data_dir, split, scaler_stats, batch_size=args.batch_size,
        num_workers=args.num_workers, neg_pos_ratio=args.neg_pos_ratio, seed=args.seed,
        noise_std=args.noise_std, limit_files=args.limit_files,)
    #solo uso esto para tener info del dataset e imprimirlo. Sino uso el dataloader
    train_ds = loaders["train"]["dataset"]
    train_loader = loaders["train"]["dataloader"]
    val_ds = loaders["val"]["dataset"]
    val_loader = loaders["val"]["dataloader"]
    # El índice se reutiliza para métricas por archivo y evaluación de eventos.
    val_ds._ensure_index()

    pos_weight = compute_positive_weight(train_ds, neg_pos_ratio=args.neg_pos_ratio)

    print(f"\nConfiguración de entrenamiento")
    print(f"Train: {len(train_ds)} ventanas ({train_ds.n_positive} positivas, "
          f"{len(train_ds) - train_ds.n_positive} negativas)")
    print(f"loss: {args.loss}  |  pos_weight: {pos_weight:.1f}  |  NEG_POS_RATIO: {args.neg_pos_ratio}")
    print(f"noise_std={args.noise_std}")
    if args.loss == "focal":
        print(f"focal_loss: gamma={args.focal_gamma}  alpha={args.focal_alpha}")
    print(f"batch_size={args.batch_size}  lr={args.lr}  weight_decay={args.weight_decay}  "
          f"epochs={args.epochs}  patience={args.patience}")

    model = SeizureCNN().to(device)

    # --- Reporte de cómputo (independiente del hardware) ---
    params_info = count_parameters(model)
    flops_info = measure_flops(model, (1, N_CHANNELS, WIN_SAMPLES))
    hardware_info = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if device == "cuda" else None,
    }
    print(f"\n[computo] params={params_info['n_params_total']:,} "
          f"(trainable={params_info['n_params_trainable']:,})  "
          f"FLOPs/ventana={flops_info['flops_per_window'] / 1e6:.1f}M  "
          f"MACs/ventana={flops_info['macs_per_window'] / 1e6:.1f}M")

    # AdamW es Adam + weight decay DESACOPLADO (hat dos formas de meterlo: acoplado, agreganfo el castigo al gradiente, o el desacoplado)
    # El adam solito usa momentum y tasa adaptativa. El weight decay sirve para evitar el overfitting. Penaliza pesos
    # grandes (en vada paso achico cada peso hacia 0) haciendolo DIRECTAMENTE en el peso, sin mezclarse con el momento adaptativo (regularización L2, en donde L2= cuadrado de los pesos. Entoncees
    # mi loss queda L_total= L + (weight_decay/2) * sumatoria de(weights^2))
    #entonces, acá creo el optimizador AdamW, le paso la lista de todos los pesos, el learning rate inicial, y el weight decay de cuánto achico los pesos.
    #NO HACE NADA TODAVÍA, solo lo creo!
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

   #La BCE mide el error de mi predicción con la fórmula L = −[ y·log(p) + (1−y)·log(1−p) ], con p la probabilidad de crisis q predije e y=0 o y=1 según si hay o no
   #la probabilidad p es básicamente la sigmoide de mis logits (los logits son la salida real de mi cnn, yo con la sigmoide q es 1/(1+e^(-logit)) lo llevo a p entre 0 y 1)
   #el BCE with logits me hace la transformación a probabilidad internamente así no la hago yop
   #toma el promedio de los errores individuales de cada ventana del batch
    if args.loss == "focal":
        train_criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    # Loss de VALIDACIÓN. Independientemente si en la función de pérdida usé o no el pos_weight, acá NO lo uso. Necesito saber si la red generaliza, de forma estable
    val_criterion = nn.BCEWithLogitsLoss()

    event_config = {
        "n_within": POSITIVES_FOR_EVENT,
        "n_window": WINDOW_RANGE_FOR_EVENT,
        "min_alarm_interval": EVENT_MIN_ALARM_INTERVAL,
        "max_latency": EVENT_MAX_LATENCY,
        "pre_onset_tolerance": EVENT_PRE_ONSET_TOLERANCE,
        "max_false_alarms_per_hour": MAX_FALSE_ALARMS_PER_HOUR,
    }
    training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "neg_pos_ratio": args.neg_pos_ratio,
        "noise_std": args.noise_std,
        "num_workers": args.num_workers,
    }

    # `val_loss` controla la paciencia porque es más estable entre épocas.
    # El checkpoint final se elige por el criterio clínico de validación:
    # mayor sensibilidad de evento dentro del techo de falsas alarmas, con
    # desempate por FDR y luego AUPRC. Si ninguna época cumple el techo, se
    # usa val_loss como fallback y queda registrado en el historial.
    best_val_loss = float("inf")
    best_selection_key: tuple | None = None
    best_epoch = 0
    best_op_threshold: float | None = None
    best_op_metrics: dict | None = None
    best_selection_metric = "event_validation"
    patience_left = args.patience
    start = time.time()
    history: list[dict] = []  # métricas por época (para las curvas de loss/accuracy en la tesis)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    peak_ram = peak_ram_mb()
    best_epoch_time_s: float | None = None

    print(f"\n=== Entrenamiento ({args.epochs} épocas máx.) ===")
    print(f"{'Ep':>3} | {'train_loss':>12} | {'val_loss':>12} | {'sens':>10} | {'spec':>10} | "
          f"{'fpr':>10} | {'fp/h':>10} | {'acc':>10} | {'tiempo':>8}")

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        #TRAIN LOSS SE CALCULA SOBRE LOS DATOS DE TRAIN, Q ESTÁN BALANCEADOS. SE MIDE MIENTRAS EL MODELO SE ACTUALIZA
        #VAL LOSS SE CALCULA SOBRE LOS PACIENTES DE VALIDACIÓN (NO LOS DE TEST TODAVÍA), ENTONCES NO TIENEN UNDERSAMPLING, NI POS WEIGHT,
        #NI DROPOUT. O SEA VAL LOSS ME DICE Q TAN BIEN GENERALIZÓ EN PROMEDIO FRENTE A DATOS CON LOS Q NO ENTRENÉ
        train_loss = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        val_loss, probs, labels = evaluate(model, val_loader, val_criterion, device)
        m = binary_metrics(
            probs,
            labels,
            THRESHOLD,
            file_ids=val_ds.file_ids,
            local_ids=val_ds.local_ids,
        )
        rk = rank_metrics(probs, labels)
        op_metrics, op_threshold = select_event_operating_point(
            probs,
            labels,
            file_ids=val_ds.file_ids,
            local_ids=val_ds.local_ids,
            valid_files=val_ds.valid_files,
            annotations=val_ds.annotations,
            max_false_alarms_per_hour=MAX_FALSE_ALARMS_PER_HOUR,
            n_within=POSITIVES_FOR_EVENT,
            n_window=WINDOW_RANGE_FOR_EVENT,
            min_alarm_interval=EVENT_MIN_ALARM_INTERVAL,
            max_latency=EVENT_MAX_LATENCY,
            pre_onset_tolerance=EVENT_PRE_ONSET_TOLERANCE,
            thresholds=THRESHOLD_GRID,
        )
        event_feasible = op_metrics is not None
        if not event_feasible:
            # Se guarda un fallback explícito para que test siga siendo
            # reproducible, pero nunca se etiqueta como clínicamente factible.
            op_metrics, op_threshold = select_lowest_false_alarm_point(
                probs,
                labels,
                file_ids=val_ds.file_ids,
                local_ids=val_ds.local_ids,
                valid_files=val_ds.valid_files,
                annotations=val_ds.annotations,
                n_within=POSITIVES_FOR_EVENT,
                n_window=WINDOW_RANGE_FOR_EVENT,
                min_alarm_interval=EVENT_MIN_ALARM_INTERVAL,
                max_latency=EVENT_MAX_LATENCY,
                pre_onset_tolerance=EVENT_PRE_ONSET_TOLERANCE,
                thresholds=THRESHOLD_GRID,
            )

        # La prioridad clínica es explícita. AUPRC no reemplaza el evento,
        # pero ayuda a desempatar modelos con igual desempeño clínico.
        if event_feasible:
            auprc_for_key = rk["auprc"] if rk["auprc"] is not None else -float("inf")
            selection_key = (
                1,
                op_metrics["sensibility"],
                -op_metrics["median_false_alarms_per_hour"],
                auprc_for_key,
                -val_loss,
            )
        else:
            selection_key = (0, -val_loss)

        elapsed_time_for_epoch = time.time() - t_epoch
        peak_ram = max(peak_ram, peak_ram_mb())
        print(f"{epoch:>3} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
              f"{m['sensibility']:>10.4f} | {m['specificity']:>10.4f} | {m['false_positive_rate']:>10.4f} | "
              f"{m['false_positive_per_hour']:>10.4f} | {m['accuracy']:>10.4f} | {elapsed_time_for_epoch:>7.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "sensibility": m["sensibility"],
            "specificity": m["specificity"],
            "false_positive_rate": m["false_positive_rate"],
            "false_positive_per_hour": m["false_positive_per_hour"],
            "accuracy": m["accuracy"],
            "val_auc_roc": rk["auc_roc"],
            "val_auprc": rk["auprc"],
            "event_threshold": op_threshold,
            "event_sensibility": op_metrics["sensibility"] if op_metrics is not None else None,
            "event_false_alarms_per_hour": (
                op_metrics["event_false_alarms_per_hour"] if op_metrics is not None else None
            ),
            "event_median_false_alarms_per_hour": (
                op_metrics["median_false_alarms_per_hour"] if op_metrics is not None else None
            ),
            "event_latency_median": op_metrics["latency_median"] if op_metrics is not None else None,
            "event_feasible": event_feasible,
            "time_s": elapsed_time_for_epoch,
        })

        # Early stopping independiente de la elección del checkpoint. No se
        # corta todavía: la época actual puede ser el mejor checkpoint clínico.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = args.patience
        else:
            patience_left -= 1
        stop_after_epoch = patience_left <= 0

        # Guardo el checkpoint que maximiza el objetivo clínico de validación.
        # Esto evita que un mínimo de BCE sobre la clase negativa decida solo
        # cuál modelo llega a test.
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_epoch = epoch
            best_op_threshold = op_threshold
            best_op_metrics = op_metrics
            best_epoch_time_s = time.time() - start
            best_selection_metric = "event_validation" if event_feasible else "val_loss_fallback"
            save_checkpoint(
                Path(args.out),
                model,
                scaler_stats,
                split=split,
                epoch=epoch,
                best_val_loss=best_val_loss,
                checkpoint_val_loss=val_loss,
                pos_weight=pos_weight,
                seed=args.seed,
                event_config=event_config,
                op_threshold=op_threshold,
                op_metrics=op_metrics,
                event_feasible=event_feasible,
                validation_auprc=rk["auprc"],
                selection_metric=best_selection_metric,
                training_config=training_config,
                loss=args.loss,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
            )
            print(
                f"    -> mejor checkpoint clínico: época={epoch}, "
                f"threshold={op_threshold if op_threshold is not None else 'N/A'}, "
                f"val_auprc={rk['auprc'] if rk['auprc'] is not None else 'N/A'}"
            )

        if stop_after_epoch:
            print(f"Early stopping: val_loss no mejoró en {args.patience} épocas. Cortando.")
            break

    total_time = time.time() - start

    # --- Resumen de cómputo (dependiente del hardware) ---
    sample_input = torch.randn(1, N_CHANNELS, WIN_SAMPLES)
    latency_info = measure_inference_latency(model, sample_input, device)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2) if device == "cuda" else None

    compute = {
        "hardware": hardware_info,
        "hardware_independent": {
            "n_params_total": params_info["n_params_total"],
            "n_params_trainable": params_info["n_params_trainable"],
            "params_by_block": params_info["params_by_block"],
            "macs_per_window": flops_info["macs_per_window"],
            "flops_per_window": flops_info["flops_per_window"],
        },
        "hardware_dependent": {
            "peak_ram_mb": round(peak_ram, 1),
            "peak_vram_mb": peak_vram_mb,
            "inference_ms_per_window_mean": latency_info["ms_per_window_mean"],
            "inference_ms_per_window_p95": latency_info["ms_per_window_p95"],
            "seconds_per_hour_of_eeg": latency_info["seconds_per_hour_of_eeg"],
            "train": {
                "batch_size": args.batch_size,
                "epochs_run": epoch,
                "best_epoch": best_epoch,
                "total_min_total": total_time / 60.0,
                "total_min_to_best": (best_epoch_time_s / 60.0) if best_epoch_time_s is not None else None,
            },
        },
    }

    print(f"\n=== Fin del entrenamiento ===")
    print(f"Mejor val_loss (early stopping): {best_val_loss:.6f}")
    print(f"Mejor época clínica: {best_epoch}  | criterio: {best_selection_metric}")
    print(f"Threshold operativo guardado: {best_op_threshold if best_op_threshold is not None else 'N/A'}")
    print(f"Tiempo total: {total_time / 60:.1f} min")
    print(f"Checkpoint: {args.out}")
    print(f"[computo] RAM pico={peak_ram:.0f} MB  "
          f"VRAM pico={peak_vram_mb if peak_vram_mb is not None else 'N/A'}  "
          f"latencia={latency_info['ms_per_window_mean']:.2f} ms/ventana  "
          f"({latency_info['seconds_per_hour_of_eeg']:.1f} s por hora de EEG)")

    # Guarda el historial por época (curvas loss/accuracy para la tesis). Se persiste
    # junto al checkpoint como <nombre>.history.json para que el notebook lo pueda leer sin re-entrenar.
    history_path = Path(args.out).with_name(Path(args.out).stem + ".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_selection_metric": best_selection_metric,
        "op_threshold": best_op_threshold,
        "op_metrics": best_op_metrics,
        "event_config": event_config,
        "training_config": training_config,
        "threshold_grid": [float(t) for t in THRESHOLD_GRID],
        "pos_weight": pos_weight,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "compute": compute,
        "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"Historial de entrenamiento: {history_path}")


if __name__ == "__main__":
    main()
