"""Entrena el modelo final con todos los pacientes de desarrollo.

La configuración no se decide aquí: se recibe desde una recomendación creada
por ``src.finalize_cv`` con predicciones out-of-fold.

Uso:
    python -m src.train_final \
        --data-dir "C:/ruta/chbmit/1.0.0" \
        --split-file data/processed/cv_splits_final.json \
        --scaler-stats data/processed/scaler_final.npz \
        --recommendation results/ratio3_pw1_recommendation.json \
        --out models/final.pt \
        --backup-dir "C:/ruta/backup/final"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    BATCH_SIZE,
    DATASET_DIR,
    GRAD_CLIP,
    LR_MIN,
    MODELS_DIR,
    NUM_WORKERS,
    SCALER_STATS_FILE,
    SPLIT_FILE,
    WEIGHT_DECAY,
)
from src.data import build_splits_dataloaders
from src.model import SeizureCNN
from src.preprocessing import load_scaler_stats
from src.train import backup_file, get_device, save_checkpoint, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento final sobre todo el desarrollo")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--scaler-stats", type=str, default=str(SCALER_STATS_FILE))
    parser.add_argument("--recommendation", type=str, required=True)
    parser.add_argument("--out", type=str, default=str(MODELS_DIR / "final.pt"))
    parser.add_argument("--backup-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--allow-ineligible", action="store_true")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size debe ser mayor que cero")
    return args


def main() -> None:
    args = parse_args()
    recommendation = json.loads(Path(args.recommendation).read_text(encoding="utf-8"))
    if not recommendation.get("eligible", False) and not args.allow_ineligible:
        sys.exit(
            "La recomendación no alcanzó la sensibilidad objetivo. "
            "Usá --allow-ineligible solo para un modelo diagnóstico."
        )

    # La recomendación fija la configuración que ya fue seleccionada con OOF.
    epochs = int(recommendation["epochs"])
    threshold = float(recommendation["threshold"])
    neg_pos_ratio = float(recommendation["neg_pos_ratio"])
    pos_weight = float(recommendation["pos_weight"])
    # Por defecto se conserva la seed elegida durante la CV; se puede sobrescribir
    # explícitamente para repetir un entrenamiento final alternativo.
    seed = int(recommendation["seed"] if args.seed is None else args.seed)
    learning_rate = float(recommendation["learning_rate"])
    weight_decay = float(recommendation["weight_decay"])
    batch_size = args.batch_size or int(recommendation["batch_size"])

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    if not split.get("final_train", False) or split.get("val"):
        sys.exit(
            "El split final debe tener final_train=true y val vacío. "
            "Generalo con src.cross_validation."
        )
    if not split.get("train") or set(split["train"]) & set(split.get("test", [])):
        sys.exit("El split final tiene train vacío o pacientes compartidos con test.")
    scaler_stats = load_scaler_stats(Path(args.scaler_stats))
    if scaler_stats is None:
        sys.exit(f"No existe el scaler indicado: {args.scaler_stats}")

    set_seed(seed)
    device = get_device(args.device)
    loaders = build_splits_dataloaders(
        args.data_dir,
        split,
        scaler_stats,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        neg_pos_ratio=neg_pos_ratio,
        seed=seed,
    )
    train_ds = loaders["train"]["dataset"]
    train_loader = loaders["train"]["dataloader"]
    if len(train_ds) == 0 or train_ds.n_positive == 0:
        sys.exit("El split final no contiene suficientes ventanas positivas de train.")

    model = SeizureCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR_MIN
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    print(f"Device: {device}  (semilla={seed})")
    print(f"Train final: {len(train_ds)} ventanas ({train_ds.n_positive} positivas)")
    print(f"Pacientes de train final: {len(split['train'])}")
    print(f"Épocas fijadas por CV: {epochs}  threshold OOF: {threshold:.3f}")

    history: list[dict] = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "time_s": time.time() - epoch_start,
        })
        print(f"{epoch:>3}/{epochs} | train_loss={train_loss:.6f}")
        scheduler.step()

    output = Path(args.out)
    save_checkpoint(
        output,
        model,
        scaler_stats,
        epoch=epochs,
        best_val_loss=float("nan"),
        pos_weight=pos_weight,
        neg_pos_ratio=neg_pos_ratio,
        cv_fold=None,
        cv_n_folds=None,
        op_threshold=threshold,
        op_sensibility=float(recommendation["oof_event_sensitivity"]),
        op_fdr_per_hour=float(recommendation["oof_event_fdr_per_hour"]),
        min_event_sensitivity=float(recommendation["min_event_sensitivity"]),
        final_train=True,
    )
    backup_file(output, args.backup_dir)

    history_path = output.with_name(output.stem + ".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {
                "history": history,
                "final_train": True,
                "checkpoint_saved": True,
                "recommendation": str(Path(args.recommendation)),
                "epochs": epochs,
                "threshold": threshold,
                "neg_pos_ratio": neg_pos_ratio,
                "pos_weight": pos_weight,
                "seed": seed,
                "total_time_s": time.time() - start,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    backup_file(history_path, args.backup_dir)
    print(f"Checkpoint final: {output}")
    print(f"Historial final: {history_path}")


if __name__ == "__main__":
    main()
