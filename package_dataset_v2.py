#!/usr/bin/env python3
"""
Model B V2 Dataset Packager - Bulletproof Edition
==================================================
Reads ONLY *_crop.jpg from raw/<class>/, splits 70/15/15,
copies with per-file error handling, then zips the result.
Output: data/model_b_webcam_v2/model_b_v2_dataset.zip
"""
import os, sys, shutil, random, hashlib, zipfile, time
from pathlib import Path
from collections import defaultdict

BASE      = Path("/Users/nithink/Desktop/m0oel/data/model_b_webcam_v2")
RAW       = BASE / "raw"
STAGE     = BASE / "staged"
ZIP_OUT   = BASE / "model_b_v2_dataset.zip"

CLASSES   = ["handrise", "look_forward", "read", "sleep",
             "stand", "turn_head", "using_device", "write"]
SPLITS    = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED      = 42

def md5(path):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return ""

def safe_copy(src, dst):
    for attempt in range(3):
        try:
            if not src.exists():
                return False
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except (FileNotFoundError, PermissionError, OSError) as e:
            if attempt < 2:
                time.sleep(0.05)
            else:
                print(f"  SKIP (3 tries): {src.name} - {e}")
                return False
    return False

def main():
    print("=" * 60)
    print("MODEL B V2 DATASET PACKAGER  (bulletproof edition)")
    print("=" * 60)

    class_files = {}
    for cls in CLASSES:
        cls_dir = RAW / cls
        if not cls_dir.is_dir():
            print(f"FATAL: missing directory {cls_dir}")
            sys.exit(1)
        crops = sorted([
            p for p in cls_dir.glob("*_crop.jpg")
            if p.is_file() and p.stat().st_size > 0
        ])
        class_files[cls] = crops
        print(f"  {cls:16s}: {len(crops):4d} crops")

    total = sum(len(v) for v in class_files.values())
    print(f"\n  TOTAL CROPS: {total}")

    empty = [c for c, f in class_files.items() if len(f) == 0]
    if empty:
        print(f"\nFATAL: empty classes: {empty}")
        sys.exit(1)

    # Deduplicate by MD5
    dedup = {"kept": 0, "dupes": 0, "bad": 0}
    for cls in CLASSES:
        seen = set()
        unique = []
        for p in class_files[cls]:
            h = md5(p)
            if not h:
                dedup["bad"] += 1
                continue
            if h in seen:
                dedup["dupes"] += 1
                continue
            seen.add(h)
            unique.append(p)
        class_files[cls] = unique
        dedup["kept"] += len(unique)
    print(f"\n  Dedup: kept={dedup['kept']}  dupes={dedup['dupes']}  bad={dedup['bad']}")

    # Clean staging
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for split in SPLITS:
        for cls in CLASSES:
            (STAGE / split / cls).mkdir(parents=True, exist_ok=True)

    # Split & copy
    random.seed(SEED)
    ok = 0
    fail = 0
    sc = defaultdict(lambda: defaultdict(int))

    for cls in CLASSES:
        files = class_files[cls][:]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        assignments = (
            [("train", f) for f in files[:n_train]] +
            [("val",   f) for f in files[n_train:n_train + n_val]] +
            [("test",  f) for f in files[n_train + n_val:]]
        )
        for split, src in assignments:
            dst = STAGE / split / cls / src.name
            if safe_copy(src, dst):
                ok += 1
                sc[split][cls] += 1
            else:
                fail += 1

    print(f"\n  Copied: {ok} OK,  {fail} skipped")
    print(f"\n{'CLASS':16s} {'TRAIN':>6s} {'VAL':>6s} {'TEST':>6s} {'TOTAL':>6s}")
    print("-" * 50)
    for cls in CLASSES:
        t, v, te = sc["train"][cls], sc["val"][cls], sc["test"][cls]
        print(f"{cls:16s} {t:6d} {v:6d} {te:6d} {t+v+te:6d}")
    tot_t  = sum(sc["train"][c] for c in CLASSES)
    tot_v  = sum(sc["val"][c]   for c in CLASSES)
    tot_te = sum(sc["test"][c]  for c in CLASSES)
    print("-" * 50)
    print(f"{'TOTAL':16s} {tot_t:6d} {tot_v:6d} {tot_te:6d} {tot_t+tot_v+tot_te:6d}")

    # Create ZIP
    print(f"\n  Creating ZIP -> {ZIP_OUT} ...")
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, filenames in os.walk(STAGE):
            for fn in sorted(filenames):
                abs_path = Path(root) / fn
                arc_name = abs_path.relative_to(STAGE)
                zf.write(abs_path, arc_name)
    zip_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"  ZIP created: {zip_mb:.1f} MB")

    shutil.rmtree(STAGE)
    print("  Staging cleaned up.")
    print("=" * 60)
    print("  DONE - download from /api/dataset/download-zip")
    print("=" * 60)

if __name__ == "__main__":
    main()
