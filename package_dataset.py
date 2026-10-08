#!/usr/bin/env python3
"""
Model B V2 Dataset Packager & Auditor
Validates, deduplicates, audits, visualizes, and zips the real-webcam dataset.
Strictly preserves all user-collected photos in raw/ without ever unlinking them.
"""

import os
import sys
import json
import hashlib
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Any

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "model_b_webcam_v2"
RAW_DIR = DATA_DIR / "raw"
ZIP_OUTPUT_PATH = BASE_DIR / "Model_B_Webcam_V2_Dataset.zip"
AUDIT_IMG_PATH = BASE_DIR / "model_b_webcam_v2_audit.png"
DATA_AUDIT_IMG_PATH = DATA_DIR / "model_b_webcam_v2_audit.png"

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

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def compute_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def ensure_directory_structure():
    """Creates directory structure for 8 canonical classes in train, val, and test."""
    for split in ["train", "val", "test"]:
        for c in CANONICAL_CLASSES:
            (DATA_DIR / split / c).mkdir(parents=True, exist_ok=True)
            (RAW_DIR / c).mkdir(parents=True, exist_ok=True)

def populate_splits_from_raw(train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """
    Populates train, val, and test splits directly from crops in RAW_DIR.
    Uses temporal block splitting to prevent adjacent video-frame leakage.
    NEVER deletes or unlinks any files in RAW_DIR.
    """
    ensure_directory_structure()
    np.random.seed(seed)
    
    # Check if raw files exist
    total_raw_crops = 0
    raw_crops_by_class: Dict[str, List[Path]] = {}
    
    for c in CANONICAL_CLASSES:
        raw_c_dir = RAW_DIR / c
        if not raw_c_dir.exists():
            raw_crops_by_class[c] = []
            continue
        # Find all crop images
        crops = sorted([
            p for p in raw_c_dir.glob("*_crop.jpg") 
            if p.is_file() and p.stat().st_size > 0
        ])
        raw_crops_by_class[c] = crops
        total_raw_crops += len(crops)
        
    print(f"[Packager] Found {total_raw_crops} total raw crops across 8 classes.")
    
    # Clean train, val, test splits before repopulating
    for sp in ["train", "val", "test"]:
        for c in CANONICAL_CLASSES:
            sp_dir = DATA_DIR / sp / c
            sp_dir.mkdir(parents=True, exist_ok=True)
            for old_f in sp_dir.glob("*"):
                if old_f.is_file():
                    old_f.unlink()
                    
    # Copy crops into train, val, test with leakage-safe block splits
    for c in CANONICAL_CLASSES:
        crops = raw_crops_by_class.get(c, [])
        n = len(crops)
        if n == 0:
            continue
            
        # Deduplicate identical MD5 hashes in memory (do not delete from raw)
        unique_crops = []
        seen_md5 = set()
        for p in crops:
            try:
                h = compute_file_hash(p)
                if h in seen_md5:
                    continue
                seen_md5.add(h)
                unique_crops.append(p)
            except Exception:
                continue
                
        n_unique = len(unique_crops)
        # Block split: Group into 10 temporal blocks
        num_blocks = 10
        block_size = max(1, n_unique // num_blocks)
        
        for i, p in enumerate(unique_crops):
            block_idx = min(num_blocks - 1, i // block_size)
            if block_idx in [7]:
                sp = "val"
            elif block_idx in [8, 9]:
                sp = "test"
            else:
                sp = "train"
                
            dst = DATA_DIR / sp / c / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(p, dst)
            except Exception as e:
                print(f"Error copying {p.name} to {sp}: {e}")

def generate_visual_audit_sheet(output_path: Path = AUDIT_IMG_PATH) -> bool:
    """Generates an 8-class visual contact sheet grid from the real dataset crops."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    axes = axes.flatten()
    
    found_any = False
    for idx, c in enumerate(CANONICAL_CLASSES):
        ax = axes[idx]
        candidates = []
        for sp in ["train", "val", "test"]:
            c_dir = DATA_DIR / sp / c
            if c_dir.exists():
                candidates.extend([p for p in c_dir.glob("*.jpg") if p.is_file()])
        
        if not candidates and (RAW_DIR / c).exists():
            candidates.extend([p for p in (RAW_DIR / c).glob("*_crop.jpg") if p.is_file()])
            
        if candidates:
            found_any = True
            chosen = candidates[np.random.randint(0, len(candidates))]
            img_bgr = cv2.imread(str(chosen))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(f"Class {idx}: {c.upper()}\n({len(candidates)} real webcam imgs)", fontsize=11, fontweight="bold", pad=8)
            else:
                ax.text(0.5, 0.5, f"{c}\n(read error)", ha="center", va="center", color="red")
        else:
            ax.text(0.5, 0.5, f"Class {idx}: {c}\n(EMPTY)", ha="center", va="center", color="#ef4444", fontsize=11, fontweight="bold")
            ax.set_title(f"Class {idx}: {c} [MISSING]", fontsize=11, color="#ef4444", pad=8)
            
        ax.axis("off")
        
    plt.suptitle("Model B V2 — Real-Webcam Dataset Visual Audit Contact Sheet (8 Canonical Classes)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.savefig(str(DATA_AUDIT_IMG_PATH), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return found_any

def create_dataset_zip(zip_path: Path = ZIP_OUTPUT_PATH) -> Tuple[int, float]:
    """Creates the Model_B_Webcam_V2_Dataset.zip archive containing train, val, and test."""
    if zip_path.exists():
        zip_path.unlink()
        
    total_images_zipped = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for split in ["train", "val", "test"]:
            split_dir = DATA_DIR / split
            if not split_dir.exists():
                continue
            for c in CANONICAL_CLASSES:
                c_dir = split_dir / c
                if not c_dir.exists():
                    continue
                arc_dir = f"model_b_webcam_v2/{split}/{c}/"
                zf.writestr(arc_dir, "")
                for img_file in sorted(c_dir.glob("*.jpg")):
                    if not img_file.is_file():
                        continue
                    arcname = f"model_b_webcam_v2/{split}/{c}/{img_file.name}"
                    zf.write(img_file, arcname=arcname)
                    total_images_zipped += 1
                    
    zip_size_mb = round(zip_path.stat().st_size / (1024 * 1024), 2)
    return total_images_zipped, zip_size_mb

def verify_zip_integrity(zip_path: Path) -> Dict[str, Any]:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file {zip_path} does not exist")
        
    verification = {
        "is_valid": True,
        "splits_found": set(),
        "classes_found": {s: set() for s in ["train", "val", "test"]},
        "image_count": 0,
        "errors": []
    }
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        if zf.testzip() is not None:
            verification["is_valid"] = False
            verification["errors"].append("Corrupt ZIP archive")
        for name in zf.namelist():
            parts = name.strip("/").split("/")
            if len(parts) >= 3 and parts[0] == "model_b_webcam_v2":
                sp, cls = parts[1], parts[2]
                if sp in ["train", "val", "test"]:
                    verification["splits_found"].add(sp)
                    if cls in CANONICAL_CLASSES:
                        verification["classes_found"][sp].add(cls)
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                verification["image_count"] += 1
                
    for sp in ["train", "val", "test"]:
        missing = set(CANONICAL_CLASSES) - verification["classes_found"][sp]
        if missing:
            verification["errors"].append(f"Split '{sp}' missing classes: {missing}")
            
    return verification

def run_pipeline() -> Dict[str, Any]:
    print("=" * 60)
    print("MODEL B V2 — DATASET VALIDATION, AUDIT & PACKAGING")
    print("=" * 60)
    
    ensure_directory_structure()
    
    # 1. Populate Splits from RAW
    print("[1/4] Populating train / val / test splits from raw photos...")
    populate_splits_from_raw()
    
    # 2. Count images across splits
    counts = {s: {c: 0 for c in CANONICAL_CLASSES} for s in ["train", "val", "test"]}
    for s in ["train", "val", "test"]:
        for c in CANONICAL_CLASSES:
            c_dir = DATA_DIR / s / c
            if c_dir.exists():
                counts[s][c] = len(list(c_dir.glob("*.jpg")))
                
    train_total = sum(counts["train"].values())
    val_total = sum(counts["val"].values())
    test_total = sum(counts["test"].values())
    grand_total = train_total + val_total + test_total

    # 3. Visual audit contact sheet
    print("[2/4] Generating visual audit sheet...")
    generate_visual_audit_sheet(AUDIT_IMG_PATH)

    # 4. Check for empty classes
    empty_classes = [c for c in CANONICAL_CLASSES if (counts["train"][c] + counts["val"][c] + counts["test"][c]) == 0]
    
    # Print Audit Table
    print("\n" + "=" * 48)
    print("MODEL B V2 DATASET AUDIT")
    print("=" * 48)
    print(f"{'Class':<20} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("-" * 48)
    for c in CANONICAL_CLASSES:
        t_c = counts["train"][c]
        v_c = counts["val"][c]
        te_c = counts["test"][c]
        tot_c = t_c + v_c + te_c
        print(f"{c:<20} {t_c:>6} {v_c:>6} {te_c:>6} {tot_c:>7}")
    print("-" * 48)
    print(f"{'Total':<20} {train_total:>6} {val_total:>6} {test_total:>6} {grand_total:>7}")
    print("=" * 48)
    print(f"Total image count:           {grand_total}")
    print()

    if grand_total == 0 or len(empty_classes) > 0:
        if ZIP_OUTPUT_PATH.exists():
            ZIP_OUTPUT_PATH.unlink()
        print("DATASET NOT READY: Empty classes detected:", empty_classes)
        return {
            "status": "not_ready",
            "message": "DATASET NOT READY",
            "empty_classes": empty_classes,
            "counts": counts,
            "grand_total": grand_total
        }

    # 5. Packaging into ZIP
    print("[3/4] Packaging real webcam dataset into ZIP...")
    img_count, zip_size_mb = create_dataset_zip(ZIP_OUTPUT_PATH)
    print(f"  ZIP created: {ZIP_OUTPUT_PATH} ({zip_size_mb} MB, {img_count} images)")
    
    print("[4/4] Verifying ZIP integrity...")
    verify_res = verify_zip_integrity(ZIP_OUTPUT_PATH)
    if not verify_res["is_valid"] or verify_res["errors"]:
        print(f"⚠️ ZIP Warnings: {verify_res['errors']}")
    else:
        print("✅ ZIP structure and integrity 100% verified.")
        
    print()
    print("DATASET CREATION COMPLETE")
    print(f"Dataset root:       {DATA_DIR}")
    print(f"ZIP:                {ZIP_OUTPUT_PATH}")
    print(f"ZIP size:           {zip_size_mb} MB")
    print(f"Train images:       {train_total}")
    print(f"Validation images:  {val_total}")
    print(f"Test images:        {test_total}")
    print(f"Total:              {grand_total}")
    print(f"Audit image:        {AUDIT_IMG_PATH}")
    print()
    print("READY FOR GOOGLE COLAB V2 TRAINING")
    print("=" * 60)
    
    return {
        "status": "success",
        "dataset_root": str(DATA_DIR),
        "zip_path": str(ZIP_OUTPUT_PATH),
        "zip_size_mb": zip_size_mb,
        "audit_img_path": str(AUDIT_IMG_PATH),
        "counts": counts,
        "grand_total": grand_total,
        "train_total": train_total,
        "val_total": val_total,
        "test_total": test_total
    }

if __name__ == "__main__":
    run_pipeline()
