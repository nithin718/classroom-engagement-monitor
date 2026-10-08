import os, glob, shutil, yaml, random
from pathlib import Path
from PIL import Image

# 1. Target 8-class taxonomy (STRICT)
TARGET_CLASSES = {
    0: "handrise", 1: "look_forward", 2: "read", 3: "sleep",
    4: "stand", 5: "turn_head", 6: "using_device", 7: "write"
}
CLASS_TO_ID = {v: k for k, v in TARGET_CLASSES.items()}

# Exact directories discovered from your Colab runtime:
SOURCES = {
    "scb": {
        "dir": "/content/drive/MyDrive/AI_Classroom_Engagement/raw_datasets/scb/actual/extracted/SCB5-Handrise-Read-write-2024-9-17",
        # In SCB: 0=handrise, 1=read, 2=write
        "raw_map": {0: 0, 1: 2, 2: 7}
    },
    "student_behaviour": {
        "dir": "/content/Student-Behaviour-Detection-6",
        "names": ['Using_phone', 'bend', 'book', 'bow_head', 'hand-raising', 'phone', 'raise_head', 'reading', 'sleep', 'turn_head', 'upright', 'writing'],
        "map": {
            "Using_phone": "using_device", "phone": "using_device",
            "hand-raising": "handrise", "reading": "read",
            "sleep": "sleep", "turn_head": "turn_head", "writing": "write"
        }
    },
    "neu": {
        "dir": "/content/neu-classroom-detected-1",
        "names": ['-', 'discuss', 'lean', 'raise_hand', 'raise_head', 'stand', 'turn_head', 'using_computer', 'using_phone'],
        "map": {
            "raise_hand": "handrise", "stand": "stand", "turn_head": "turn_head",
            "using_computer": "using_device", "using_phone": "using_device", "sleep": "sleep"
        }
    },
    "effortless": {
        "dir": "/content/Effortless-Classroom-Behavior-1",
        "names": ['Hand-raising', 'Looking-forward', 'Reading', 'Sleeping', 'Turning-around'],
        "map": {
            "Hand-raising": "handrise", "Looking-forward": "look_forward",
            "Reading": "read", "Sleeping": "sleep", "Turning-around": "turn_head"
        }
    },
    "eaglenanao": {
        "dir": "/content/new-student-classroom-activity-3-hand-raise-phone-sleep-2-1",
        "names": ['phone', 'sleep', 'study'],
        "map": {"phone": "using_device", "sleep": "sleep"}
    }
}

# Optional Kaggle anchor if present on Drive
kaggle_candidates = glob.glob("/content/drive/MyDrive/**/kaggle_classroom_engagement*", recursive=True)
if kaggle_candidates:
    SOURCES["kaggle"] = {"dir": kaggle_candidates[0], "raw_map": {i: i for i in range(8)}}

# Clean & create local staging directories
LOCAL_DATASET = "/content/dataset"
if os.path.exists(LOCAL_DATASET):
    shutil.rmtree(LOCAL_DATASET)
for split in ["train", "val", "test"]:
    os.makedirs(f"{LOCAL_DATASET}/{split}/images", exist_ok=True)
    os.makedirs(f"{LOCAL_DATASET}/{split}/labels", exist_ok=True)

neu_items = []
std_items = []

print("=== PARSING & NORMALIZING ALL VERIFIED DATASETS ===")
for src_key, info in SOURCES.items():
    s_dir = Path(info["dir"])
    if not s_dir.exists():
        print(f"⚠️ Directory {s_dir} not found. Skipping {src_key}.")
        continue
    
    # Find all images
    imgs = [f for f in s_dir.rglob("*") if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    print(f"\nProcessing {src_key} ({len(imgs)} candidate images)...")
    
    valid_count = 0
    for img_p in imgs:
        lbl_p = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            lbl_candidates = list(s_dir.rglob(f"{img_p.stem}.txt"))
            if not lbl_candidates:
                continue
            lbl_p = lbl_candidates[0]
            
        valid_lines = []
        try:
            with open(lbl_p, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid_str, x, y, w, h = parts
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    if w <= 0 or h <= 0 or not (0 <= x <= 1 and 0 <= y <= 1):
                        continue
                        
                    target_cid = None
                    if "raw_map" in info:
                        c_int = int(cid_str)
                        target_cid = info["raw_map"].get(c_int)
                    elif "map" in info and "names" in info:
                        c_int = int(cid_str)
                        if 0 <= c_int < len(info["names"]):
                            src_name = info["names"][c_int]
                            if src_name in info["map"]:
                                target_cid = CLASS_TO_ID[info["map"][src_name]]
                    
                    if target_cid is not None:
                        valid_lines.append(f"{target_cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception:
            continue
            
        if valid_lines:
            valid_count += 1
            unique_stem = f"{src_key}_{img_p.stem}"
            item = (unique_stem, img_p, valid_lines)
            if src_key == "neu":
                neu_items.append(item)
            else:
                std_items.append(item)
                
    print(f"  -> {valid_count} annotated images ready from {src_key}")

# Leakage-Free Splitting
random.seed(42)
random.shuffle(std_items)

splits = {"train": [], "val": [], "test": []}

# NEU video frames: Contiguous sequence split (70/15/15) to prevent adjacent frame leakage
if neu_items:
    neu_items.sort(key=lambda x: x[0])
    n = len(neu_items)
    t = int(n * 0.70)
    v = int(n * 0.15)
    splits["train"].extend(neu_items[:t])
    splits["val"].extend(neu_items[t:t+v])
    splits["test"].extend(neu_items[t+v:])

# Standard datasets: 80% train, 10% val, 10% test
n_s = len(std_items)
t_s = int(n_s * 0.80)
v_s = int(n_s * 0.10)
splits["train"].extend(std_items[:t_s])
splits["val"].extend(std_items[t_s:t_s+v_s])
splits["test"].extend(std_items[t_s+v_s:])

print("\nWriting files to /content/dataset/...")
for split_name, items in splits.items():
    for stem, src_img, lines in items:
        shutil.copy(src_img, f"{LOCAL_DATASET}/{split_name}/images/{stem}{src_img.suffix}")
        with open(f"{LOCAL_DATASET}/{split_name}/labels/{stem}.txt", "w") as f:
            f.writelines(lines)

# Generate data.yaml
yaml_dict = {
    "path": LOCAL_DATASET,
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": TARGET_CLASSES
}
yaml_path = f"{LOCAL_DATASET}/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yaml_dict, f, sort_keys=False)

train_count = len(list(Path(f"{LOCAL_DATASET}/train/images").glob("*")))
val_count = len(list(Path(f"{LOCAL_DATASET}/val/images").glob("*")))
test_count = len(list(Path(f"{LOCAL_DATASET}/test/images").glob("*")))

print("\n" + "="*50)
print(f"🎉 DATASET READY FOR TRAINING:")
print(f"  Train Images: {train_count}")
print(f"  Val Images:   {val_count}")
print(f"  Test Images:  {test_count}")
print("="*50)

assert train_count > 0, "STOP: 0 train images!"

# Launch 280-Epoch Training Saving Directly to Google Drive
from ultralytics import YOLO
import torch

torch.cuda.empty_cache()
EXP_NAME = "yolo11s_six_dataset_t4_final"
PROJECT_DIR = "/content/drive/MyDrive/AI_Classroom_Engagement/experiments"
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
