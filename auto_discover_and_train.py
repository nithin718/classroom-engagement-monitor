import os, sys, glob, shutil, yaml, hashlib, random, zipfile
from pathlib import Path
from collections import Counter
from PIL import Image

# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

ROOT = "/content/drive/MyDrive/AI_Classroom_Engagement"
LOCAL_DATASET = "/content/dataset"
RAW_ROOT = Path(f"{ROOT}/raw_datasets")

print(f"=== INSPECTING RAW DATASETS IN {RAW_ROOT} ===")
if not RAW_ROOT.exists():
    # Check alternate known directory names
    alt = Path("/content/drive/MyDrive/AI_Classroom_Engagement_System/raw_datasets")
    if alt.exists():
        RAW_ROOT = alt
        print(f"Using alternative path: {RAW_ROOT}")

# 2. AUTO-EXTRACT ANY ZIP FILES FIRST
zip_files = list(RAW_ROOT.rglob("*.zip"))
if zip_files:
    print(f"Found {len(zip_files)} .zip archives. Extracting...")
    for z in zip_files:
        extract_target = z.parent / z.stem
        if not extract_target.exists() or len(list(extract_target.rglob("*"))) < 5:
            print(f"  Extracting {z.name} -> {extract_target}...")
            try:
                with zipfile.ZipFile(z, 'r') as zip_ref:
                    zip_ref.extractall(extract_target)
            except Exception as e:
                print(f"  Error extracting {z}: {e}")

# 3. REPORT EXTENSIONS AND FILE COUNTS
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
for d in sorted(RAW_ROOT.iterdir()):
    if d.is_dir():
        all_f = [f for f in d.rglob("*") if f.is_file()]
        img_f = [f for f in all_f if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        lbl_f = [f for f in all_f if f.suffix == ".txt"]
        exts = set(f.suffix for f in all_f)
        print(f"Folder '{d.name}': {len(img_f)} images, {len(lbl_f)} txt labels. Extensions: {exts}")

TARGET_CLASSES = {
    0: "handrise", 1: "look_forward", 2: "read", 3: "sleep",
    4: "stand", 5: "turn_head", 6: "using_device", 7: "write"
}
CLASS_TO_ID = {v: k for k, v in TARGET_CLASSES.items()}

SOURCE_MAPPINGS = {
    "kaggle": {c: c for c in TARGET_CLASSES.values()},
    "scb": {"hand-raising": "handrise", "read": "read", "reading": "read", "write": "write", "writing": "write"},
    "mywork": {"Using_phone": "using_device", "phone": "using_device", "hand-raising": "handrise", "reading": "read", "sleep": "sleep", "turn_head": "turn_head", "writing": "write"},
    "neu": {"raise_hand": "handrise", "stand": "stand", "turn_head": "turn_head", "using_computer": "using_device", "using_phone": "using_device", "sleep": "sleep"},
    "effortless": {"Hand-raising": "handrise", "Looking-forward": "look_forward", "Reading": "read", "Sleeping": "sleep", "Turning-around": "turn_head"},
    "eaglenanao": {"phone": "using_device", "sleep": "sleep"}
}

# Clean local staging
if os.path.exists(LOCAL_DATASET):
    shutil.rmtree(LOCAL_DATASET)
for split in ["train", "val", "test"]:
    os.makedirs(f"{LOCAL_DATASET}/{split}/images", exist_ok=True)
    os.makedirs(f"{LOCAL_DATASET}/{split}/labels", exist_ok=True)

# 4. PARSE & NORMALIZE
staging_items = []
neu_items = []
std_items = []
seen_hashes = set()

for src_folder in sorted(RAW_ROOT.iterdir()):
    if not src_folder.is_dir():
        continue
    
    # Identify mapping key
    map_key = None
    for k in SOURCE_MAPPINGS:
        if k in src_folder.name.lower():
            map_key = k
            break
    if not map_key:
        map_key = "kaggle"
    mapping = SOURCE_MAPPINGS[map_key]
    
    # Look for YAML class map
    src_id_map = {}
    for yf in list(src_folder.rglob("*.yaml")) + list(src_folder.rglob("*.yml")):
        try:
            y = yaml.safe_load(open(yf))
            names = y.get("names", [])
            if isinstance(names, dict):
                src_id_map = names
            elif isinstance(names, list):
                src_id_map = {i: n for i, n in enumerate(names)}
            if src_id_map:
                break
        except Exception:
            pass
            
    # Case-insensitive image discovery
    all_imgs = [f for f in src_folder.rglob("*") if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    print(f"\nProcessing {src_folder.name}: {len(all_imgs)} images found...")
    
    valid_count = 0
    for img_file in all_imgs:
        try:
            with open(img_file, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        except Exception:
            continue
            
        lbl_candidates = list(src_folder.rglob(f"{img_file.stem}.txt"))
        if not lbl_candidates:
            continue
        lbl_file = lbl_candidates[0]
        
        valid_lines = []
        try:
            with open(lbl_file) as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid_raw, x, y, w, h_box = parts
                    x, y, w, h_box = float(x), float(y), float(w), float(h_box)
                    if w <= 0 or h_box <= 0 or not (0 <= x <= 1 and 0 <= y <= 1):
                        continue
                        
                    target_cid = None
                    if map_key == "kaggle":
                        if 0 <= int(cid_raw) <= 7:
                            target_cid = int(cid_raw)
                    else:
                        src_name = src_id_map.get(int(cid_raw) if cid_raw.isdigit() else -1, cid_raw)
                        if src_name in mapping:
                            target_cid = CLASS_TO_ID[mapping[src_name]]
                            
                    if target_cid is not None:
                        valid_lines.append(f"{target_cid} {x:.6f} {y:.6f} {w:.6f} {h_box:.6f}\n")
        except Exception:
            continue
            
        if valid_lines:
            valid_count += 1
            unique_stem = f"{src_folder.name}_{img_file.stem}"
            item = (unique_stem, img_file, valid_lines)
            if "neu" in src_folder.name.lower():
                neu_items.append(item)
            else:
                std_items.append(item)
                
    print(f"  -> {valid_count} valid annotated images extracted from {src_folder.name}")

# 5. SPLIT
random.seed(42)
random.shuffle(std_items)

splits_data = {"train": [], "val": [], "test": []}

if neu_items:
    neu_items.sort(key=lambda x: x[0])
    n_neu = len(neu_items)
    t_neu = int(n_neu * 0.70)
    v_neu = int(n_neu * 0.15)
    splits_data["train"].extend(neu_items[:t_neu])
    splits_data["val"].extend(neu_items[t_neu:t_neu+v_neu])
    splits_data["test"].extend(neu_items[t_neu+v_neu:])

n_std = len(std_items)
t_std = int(n_std * 0.80)
v_std = int(n_std * 0.10)
splits_data["train"].extend(std_items[:t_std])
splits_data["val"].extend(std_items[t_std:t_std+v_std])
splits_data["test"].extend(std_items[t_std+v_std:])

print("\nCopying files into local /content/dataset/...")
for split_name, items in splits_data.items():
    for stem, src_img, lines in items:
        dest_img = f"{LOCAL_DATASET}/{split_name}/images/{stem}{src_img.suffix}"
        dest_lbl = f"{LOCAL_DATASET}/{split_name}/labels/{stem}.txt"
        shutil.copy(src_img, dest_img)
        with open(dest_lbl, "w") as out_f:
            out_f.writelines(lines)

# 6. WRITE DATA.YAML
yaml_path = f"{LOCAL_DATASET}/data.yaml"
data_yaml_dict = {
    "path": LOCAL_DATASET,
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": TARGET_CLASSES
}
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml_dict, f, sort_keys=False)

train_imgs = list(Path(f"{LOCAL_DATASET}/train/images").glob("*"))
val_imgs = list(Path(f"{LOCAL_DATASET}/val/images").glob("*"))
test_imgs = list(Path(f"{LOCAL_DATASET}/test/images").glob("*"))

print("\n" + "="*50)
print(f"FINAL PREPARED: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
print("="*50)

assert len(train_imgs) > 0, "STOP: 0 train images! Check the folder extensions printed above."

# 7. LAUNCH 280-EPOCH TRAINING
from ultralytics import YOLO
import torch

torch.cuda.empty_cache()
EXP_NAME = "yolo11s_six_dataset_t4_final"
PROJECT_DIR = f"{ROOT}/experiments"
LAST_PT = f"{PROJECT_DIR}/{EXP_NAME}/weights/last.pt"

if os.path.exists(LAST_PT):
    print(f"\n🔄 Found checkpoint at {LAST_PT}. Resuming training...")
    model = YOLO(LAST_PT)
    results = model.train(resume=True)
else:
    print("\n🚀 Starting clean 280-epoch training from pretrained yolo11s.pt...")
    model = YOLO("yolo11s.pt")
    results = model.train(
        data=yaml_path,
        project=PROJECT_DIR,
        name=EXP_NAME,
        exist_ok=True,
        epochs=280,
        imgsz=512,
        batch=32,
        device=0,
        workers=4,
        amp=True,
        cache="disk",
        patience=0,
        optimizer="auto",
        lr0=0.008,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
        copy_paste=0.1,
        mixup=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        cls=0.7,
        box=7.5,
        dfl=1.5,
        seed=42,
        save=True,
        save_period=1,
        plots=True,
        verbose=True
    )
