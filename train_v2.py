#!/usr/bin/env python3
"""
Model B v2 Training Pipeline — Real-Webcam Domain Adaptation & Fine-Tuning
Architecture: YOLO11s-cls
Base Weights: models/best.pt (Model B V1)
Target: single_student_8class_v2_webcam/weights/best.pt
"""

import os
import sys
import json
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "model_b_webcam_v2"
V1_MODEL_PATH = BASE_DIR / "models" / "best.pt"
OUTPUT_DIR = BASE_DIR / "runs" / "single_student_8class_v2_webcam"

CANONICAL_CLASSES = [
    "handrise",      # 0
    "look_forward",  # 1
    "read",          # 2
    "sleep",         # 3
    "stand",         # 4
    "turn_head",     # 5
    "using_device",  # 6
    "write"          # 7
]

def check_dataset_balance(min_samples_per_class: int = 15):
    print("=" * 60)
    print("CHECKING DATASET INTEGRITY & CLASS BALANCE")
    print("=" * 60)
    
    splits = ["train", "val", "test"]
    counts = {s: {} for s in splits}
    has_insufficient = False

    csv_lines = ["class,train,val,test,total"]

    for c in CANONICAL_CLASSES:
        t_c = len(list((DATA_DIR / "train" / c).glob("*.jpg"))) + len(list((DATA_DIR / "train" / c).glob("*.png")))
        v_c = len(list((DATA_DIR / "val" / c).glob("*.jpg"))) + len(list((DATA_DIR / "val" / c).glob("*.png")))
        te_c = len(list((DATA_DIR / "test" / c).glob("*.jpg"))) + len(list((DATA_DIR / "test" / c).glob("*.png")))
        total = t_c + v_c + te_c

        counts["train"][c] = t_c
        counts["val"][c] = v_c
        counts["test"][c] = te_c

        print(f"  {c:<15} | Train: {t_c:>4} | Val: {v_c:>4} | Test: {te_c:>4} | Total: {total:>4}")
        csv_lines.append(f"{c},{t_c},{v_c},{te_c},{total}")

        if t_c < min_samples_per_class:
            has_insufficient = True
            print(f"  ⚠️ Warning: Class '{c}' has only {t_c} training samples (minimum required: {min_samples_per_class})")

    # Save class distribution CSV
    dist_csv_path = DATA_DIR / "class_distribution.csv"
    with open(dist_csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"Saved class distribution to {dist_csv_path}")

    return counts, has_insufficient

def run_v2_training(
    epochs: int = 50,
    batch_size: int = 16,
    lr0: float = 0.001,
    patience: int = 10,
    min_samples: int = 15,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
):
    print("=" * 60)
    print("STARTING MODEL B V2 FINE-TUNING")
    print("=" * 60)
    print(f"Base model checkpoint: {V1_MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Dataset root: {DATA_DIR}")

    counts, has_insufficient = check_dataset_balance(min_samples_per_class=min_samples)
    if has_insufficient:
        print("\n❌ ERROR: Insufficient samples per class. Collect more real webcam data via http://localhost:8000/collect before launching training.")
        sys.exit(1)

    # Prepare training configuration
    train_config = {
        "model": str(V1_MODEL_PATH),
        "data": str(DATA_DIR),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": 224,
        "device": device,
        "workers": 2,
        "patience": patience,
        "lr0": lr0,
        "lrf": 0.01,
        "seed": 42,
        # Moderate augmentations preserving hands, phone, pen, notebook
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.08,
        "scale": 0.1,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "project": str(BASE_DIR / "runs"),
        "name": "single_student_8class_v2_webcam",
        "exist_ok": True
    }

    config_yaml_path = DATA_DIR / "training_config.yaml"
    with open(config_yaml_path, "w") as f:
        yaml.dump(train_config, f)
    print(f"Saved training configuration to {config_yaml_path}")

    print("\nInitializing YOLO11s-cls fine-tuning...")
    model = YOLO(str(V1_MODEL_PATH))

    results = model.train(
        data=str(DATA_DIR),
        epochs=epochs,
        batch=batch_size,
        imgsz=224,
        device=device,
        workers=2,
        patience=patience,
        lr0=lr0,
        lrf=0.01,
        seed=42,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.08,
        scale=0.1,
        fliplr=0.5,
        project=str(BASE_DIR / "runs"),
        name="single_student_8class_v2_webcam",
        exist_ok=True
    )

    print("=" * 60)
    print("V2 TRAINING COMPLETE!")
    print("=" * 60)
    
    best_pt = BASE_DIR / "runs" / "single_student_8class_v2_webcam" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"✅ V2 Best Checkpoint: {best_pt}")
    else:
        print("⚠️ Checkpoint not found at default path, inspect runs directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model B v2 Training Pipeline")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--min_samples", type=int, default=15, help="Minimum train samples per class required")
    parser.add_argument("--check_only", action="store_true", help="Only check dataset counts without training")
    args = parser.parse_args()

    if args.check_only:
        check_dataset_balance(min_samples_per_class=args.min_samples)
    else:
        run_v2_training(
            epochs=args.epochs,
            batch_size=args.batch,
            lr0=args.lr,
            patience=args.patience,
            min_samples=args.min_samples
        )
