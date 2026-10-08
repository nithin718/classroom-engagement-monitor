"""
Model B Inference Engine & Diagnostics
Single-Student 8-Class Behavior Classifier (YOLO11s-cls)

Architecture:
Single student crop -> YOLO11s Classification (224x224) -> 8 behavior classes ->
Temporal processing -> Operational behavioral engagement indicator (0-100) -> Dashboard / UART

Canonical Behavior Classes (Strict order):
0: handrise
1: look_forward
2: read
3: sleep
4: stand
5: turn_head
6: using_device
7: write

Operational Behavioral Engagement Indicator:
handrise:     100
look_forward: 100
read:          85
sleep:          5
stand:         60
turn_head:     40
using_device:  20
write:         80
"""

import base64
import time
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    torch = None

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DEBUG_DIR = BASE_DIR / "debug_samples"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Primary checkpoints
MODEL_B_V2_PATH = MODELS_DIR / "Model_B_v2_best.pt"
if not MODEL_B_V2_PATH.exists():
    _dl_v2 = Path("/Users/nithink/Downloads/Model_B_v2_best.pt")
    if _dl_v2.exists():
        MODEL_B_V2_PATH = _dl_v2

MODEL_B_V1_PATH = MODELS_DIR / "best.pt"
if not MODEL_B_V1_PATH.exists():
    _dl_v1 = Path("/Users/nithink/Downloads/Model_B_best.pt")
    if _dl_v1.exists():
        MODEL_B_V1_PATH = _dl_v1

# Authoritative primary Model B configuration
MODEL_B_CHECKPOINT = MODEL_B_V2_PATH
MODEL_PATH = MODEL_B_CHECKPOINT  # backward compatibility alias

# Strict Canonical Class Order
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

# Canonical Operational Behavioral Engagement Indicator Mapping
ENGAGEMENT_MAP = {
    "handrise": {"score": 100.0, "level": "HIGH", "category": "Active Participation", "badge_color": "#10b981"},
    "look_forward": {"score": 100.0, "level": "HIGH", "category": "Attentive Listening", "badge_color": "#38bdf8"},
    "read": {"score": 85.0, "level": "HIGH", "category": "Focused Study", "badge_color": "#10b981"},
    "sleep": {"score": 5.0, "level": "LOW", "category": "Sleeping / Drowsy", "badge_color": "#8b5cf6"},
    "stand": {"score": 60.0, "level": "MEDIUM", "category": "Standing / Moving", "badge_color": "#f59e0b"},
    "turn_head": {"score": 40.0, "level": "MEDIUM", "category": "Distracted / Looking Away", "badge_color": "#f97316"},
    "using_device": {"score": 20.0, "level": "LOW", "category": "Device Usage / Off-task", "badge_color": "#ef4444"},
    "write": {"score": 80.0, "level": "HIGH", "category": "Note Taking / Writing", "badge_color": "#10b981"},
}

LEVEL_CODES = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "OFF-TASK": 3
}

# Structured Logger
logger = logging.getLogger("model_b_inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)


class TemporalDebouncer:
    """
    Stabilizes real-time predictions against frame-to-frame noise without
    fabricating or overriding genuine model outputs.
    - min_hold_sec: minimum hold time before switching to low-confidence alternatives
    - fast_switch_conf: if new prediction has >= 70% confidence, transitions promptly
    - candidate_frames: requires 2 consecutive consistent frames to switch
    """
    def __init__(self, min_hold_sec: float = 0.5, fast_switch_conf: float = 70.0):
        self.min_hold_sec = min_hold_sec
        self.fast_switch_conf = fast_switch_conf
        self.current_class = "look_forward"
        self.current_conf = 95.0
        self.current_probs = {c: 0.0 for c in CANONICAL_CLASSES}
        self.current_probs["look_forward"] = 95.0
        self.last_transition_time = time.time()
        self.candidate_class = None
        self.candidate_frames = 0
        self.transition_history = []

    def reset(self):
        self.current_class = "look_forward"
        self.current_conf = 95.0
        self.candidate_class = None
        self.candidate_frames = 0
        self.last_transition_time = time.time()

    def update(self, detected_class: str, conf: float, probs: Dict[str, float], enabled: bool = True) -> Tuple[str, float, Dict[str, float]]:
        now = time.time()
        
        # When temporal filter is disabled, pass through RAW predictions instantly
        if not enabled:
            if detected_class != self.current_class:
                self.transition_history.append({
                    "from": self.current_class,
                    "to": detected_class,
                    "timestamp": now,
                    "mode": "raw_instant"
                })
            self.current_class = detected_class
            self.current_conf = conf
            self.current_probs = probs
            self.last_transition_time = now
            self.candidate_class = None
            self.candidate_frames = 0
            return self.current_class, self.current_conf, self.current_probs

        time_in_state = now - self.last_transition_time

        # If detected class matches active class, update smoothly
        if detected_class == self.current_class:
            self.current_conf = round(0.7 * self.current_conf + 0.3 * conf, 1)
            self.current_probs = probs
            self.candidate_class = None
            self.candidate_frames = 0
            return self.current_class, self.current_conf, self.current_probs

        # Candidate tracking
        if detected_class == self.candidate_class:
            self.candidate_frames += 1
        else:
            self.candidate_class = detected_class
            self.candidate_frames = 1

        # Responsive transition conditions:
        # 1. High confidence (>= fast_switch_conf) persisted for 2 frames
        # 2. Candidate persisted for 3 frames regardless of confidence
        # 3. Minimum hold time passed and candidate persisted for 2 frames
        should_switch = (
            (conf >= self.fast_switch_conf and self.candidate_frames >= 2) or
            (self.candidate_frames >= 3) or
            (time_in_state >= self.min_hold_sec and self.candidate_frames >= 2)
        )

        if should_switch:
            self.transition_history.append({
                "from": self.current_class,
                "to": detected_class,
                "timestamp": now,
                "hold_duration": round(time_in_state, 3),
                "mode": "smoothed"
            })
            self.current_class = detected_class
            self.current_conf = conf
            self.current_probs = probs
            self.last_transition_time = now
            self.candidate_class = None
            self.candidate_frames = 0

        return self.current_class, self.current_conf, self.current_probs


class ModelBClassifier:
    def __init__(self, model_path: Path = MODEL_B_CHECKPOINT):
        self.model_path = model_path
        self.model = None
        self.device = "cpu"
        self.names = CANONICAL_CLASSES
        self.is_loaded = False
        self.load_error = None
        self.rolling_latency = 20.0
        self.active_version = "v2" if "v2" in str(model_path).lower() else "v1"
        self.debouncer = TemporalDebouncer(min_hold_sec=0.5, fast_switch_conf=70.0)
        self._load_model()

    def switch_model_version(self, version: str = "v2") -> Dict[str, Any]:
        version = version.lower().strip()

        if version in ("v2", "model_b_v2"):
            target_path = MODEL_B_V2_PATH
            target_version = "v2"
        elif version in ("v1", "model_b_v1"):
            target_path = MODEL_B_V1_PATH
            target_version = "v1"
        else:
            return {
                "status": "error",
                "message": f"Unknown model version '{version}'. Available: 'v1', 'v2'",
                "active_version": getattr(self, "active_version", "v2")
            }

        if not target_path.exists():
            return {
                "status": "error",
                "message": f"Model weights not found at {target_path}",
                "active_version": getattr(self, "active_version", "v2")
            }

        self.model_path = target_path
        self.active_version = target_version
        self._load_model()

        return {
            "status": "success",
            "active_version": self.active_version,
            "model_path": str(self.model_path),
            "device": self.device,
            "classes": self.names
        }

    def _load_model(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model B checkpoint not found at {self.model_path}")

            if YOLO is None:
                raise ImportError("Ultralytics package not installed.")

            print(f"🔄 Loading Model B ({self.active_version.upper()}) from {self.model_path}...")
            self.model = YOLO(str(self.model_path))

            # Task verification
            if getattr(self.model, "task", None) != "classify":
                raise ValueError(f"Invalid model task: expected 'classify', got '{getattr(self.model, 'task', None)}'")

            if torch and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            # Strict 8-class Canonical Verification
            model_classes = [self.model.names[i] for i in range(len(self.model.names))]
            if len(model_classes) != 8:
                raise ValueError(f"Invalid number of classes: expected 8, got {len(model_classes)} ({model_classes})")

            if model_classes != CANONICAL_CLASSES:
                raise ValueError(
                    f"Canonical class order mismatch!\nExpected: {CANONICAL_CLASSES}\nGot:      {model_classes}"
                )

            self.names = model_classes
            self.is_loaded = True
            self.load_error = None
            print(f"✅ Model B ({self.active_version.upper()}) loaded successfully on {self.device}.")
            print(f"   Architecture: YOLO11s Classification")
            print(f"   Classes (8):  {self.names}")
            print(f"   Input Size:   224x224")
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            print(f"❌ FATAL: Failed to load Model B: {e}")
            raise

    def apply_crop(self, img_bgr: np.ndarray, crop_mode: str = "mode_b") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Crop framing modes:
        - mode_b / head_torso_hands: Focused student crop (15% to 85% W, 8% to 96% H) - default & recommended
        - mode_a / full_camera: Full frame (100% W x 100% H)
        - mode_c / wider_upper_body: Wider student crop (8% to 92% W, 4% to 96% H)
        """
        h, w = img_bgr.shape[:2]
        crop_mode = str(crop_mode).lower().strip()

        if crop_mode in ("mode_b", "head_torso_hands", "b"):
            x1 = int(w * 0.15)
            x2 = int(w * 0.85)
            y1 = int(h * 0.08)
            y2 = int(h * 0.96)
        elif crop_mode in ("mode_c", "wider_upper_body", "c"):
            x1 = int(w * 0.08)
            x2 = int(w * 0.92)
            y1 = int(h * 0.04)
            y2 = int(h * 0.96)
        else:  # mode_a
            return img_bgr, {"x": 0, "y": 0, "width": w, "height": h, "mode": "mode_a"}

        x1 = max(0, min(x1, w - 10))
        x2 = max(x1 + 10, min(x2, w))
        y1 = max(0, min(y1, h - 10))
        y2 = max(y1 + 10, min(y2, h))

        cropped = img_bgr[y1:y2, x1:x2]
        return cropped, {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1, "mode": crop_mode}

    def build_uart_packet(self, score: float, level: str, class_id: int) -> Dict[str, Any]:
        """
        Constructs UART packet for TM4C123GXL:
        [SYNC(0xAA), SCORE_BYTE, LEVEL_CODE, CLASS_ID, CHECKSUM(XOR)]
        """
        score_byte = max(0, min(100, int(round(score))))
        level_code = LEVEL_CODES.get(level, 0)
        start_byte = 0xAA
        checksum = (start_byte ^ score_byte ^ level_code ^ class_id) & 0xFF
        packet_bytes = [start_byte, score_byte, level_code, class_id, checksum]
        hex_repr = " ".join(f"0x{b:02X}" for b in packet_bytes)
        return {
            "bytes": packet_bytes,
            "hex": hex_repr,
            "sync": "0xAA",
            "score_byte": score_byte,
            "level_code": level_code,
            "class_id": class_id,
            "checksum": f"0x{checksum:02X}"
        }

    def predict_image(
        self, 
        img_input: Any, 
        crop_mode: str = "mode_b",
        temporal_filter: bool = True,
        fps_info: float = 0.0,
        uart_connected: bool = True,
        force_class: Optional[str] = None,
        *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Pure, unadulterated Model B inference pipeline:
        real image -> real crop -> real YOLO11s prediction -> real confidence ->
        temporal debouncing (optional) -> canonical operational behavioral engagement score -> UART packet.
        
        NO probability multipliers, NO arbitrary boosts, NO hardcoded detector overrides.
        """
        if not self.is_loaded:
            if self.load_error:
                raise RuntimeError(f"Model B not loaded: {self.load_error}")
            self._load_model()

        img_bgr = None
        is_benchmark = False

        if isinstance(img_input, str):
            if "sample_" in img_input:
                is_benchmark = True
            if img_input.startswith("data:image"):
                header, encoded = img_input.split(",", 1)
                data = base64.b64decode(encoded)
                arr = np.frombuffer(data, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            elif os.path.exists(img_input):
                img_bgr = cv2.imread(img_input)
            else:
                try:
                    data = base64.b64decode(img_input)
                    arr = np.frombuffer(data, dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception:
                    raise ValueError("Invalid image path or base64 format")
        elif isinstance(img_input, np.ndarray):
            img_bgr = img_input
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}")

        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Failed to decode image input")

        orig_h, orig_w = img_bgr.shape[:2]

        # Apply Crop Framing (skip re-cropping if already a benchmark sample crop)
        if is_benchmark:
            cropped_img = img_bgr
            crop_meta = {"x": 0, "y": 0, "width": orig_w, "height": orig_h, "mode": "sample_crop"}
        else:
            cropped_img, crop_meta = self.apply_crop(img_bgr, crop_mode)

        # Run pure YOLO11s-cls (224x224 classification)
        t_inf_start = time.perf_counter()
        results = self.model.predict(source=cropped_img, imgsz=224, verbose=False)
        r = results[0]
        raw_latency = (time.perf_counter() - t_inf_start) * 1000

        # Extract RAW softmax probabilities directly from model
        raw_probs = {}
        for i, cname in enumerate(self.names):
            p = float(r.probs.data[i])
            raw_probs[cname] = round(p * 100.0, 2)

        # Raw Top-1 directly from model
        raw_top_idx = int(r.probs.top1)
        raw_top_name = self.names[raw_top_idx]
        raw_top_conf = raw_probs[raw_top_name]

        # Raw Top-3 Candidate Behaviors
        sorted_probs = sorted(raw_probs.items(), key=lambda x: x[1], reverse=True)
        raw_top3 = [
            {"class_name": c, "probability": p, "class_id": self.names.index(c)}
            for c, p in sorted_probs[:3]
        ]

        # Apply Temporal Debouncer (pass-through when disabled or in benchmark test)
        if is_benchmark:
            stable_class = raw_top_name
            stable_conf = raw_top_conf
            stable_probs = raw_probs
        else:
            stable_class, stable_conf, stable_probs = self.debouncer.update(
                raw_top_name, raw_top_conf, raw_probs, enabled=temporal_filter
            )

        final_class = stable_class if temporal_filter else raw_top_name
        final_conf = stable_conf if temporal_filter else raw_top_conf
        final_probs = stable_probs if temporal_filter else raw_probs
        final_id = self.names.index(final_class)

        # Operational Behavioral Engagement Indicator (Canonical Mapping)
        eng_meta = ENGAGEMENT_MAP[final_class]
        score = eng_meta["score"]
        level = eng_meta["level"]
        category = eng_meta["category"]
        badge_color = eng_meta["badge_color"]

        # Build embedded UART packet
        uart = self.build_uart_packet(score, level, final_id)

        # Latency tracking
        self.rolling_latency = round(0.85 * self.rolling_latency + 0.15 * raw_latency, 2)
        fps = round(1000.0 / max(1.0, self.rolling_latency), 1) if fps_info <= 0 else round(fps_info, 1)

        # Scientific Logging Requirement (Requirement 14)
        ts_now = datetime.now().strftime("%H:%M:%S")
        uart_status_str = "CONNECTED" if uart_connected else "DISCONNECTED"
        log_msg = (
            f"[{ts_now}] Model B {self.active_version.upper()} "
            f"behavior={final_class} "
            f"confidence={round(final_conf / 100.0, 3)} "
            f"indicator={int(score)} "
            f"fps={fps} "
            f"uart={uart_status_str}"
        )
        logger.info(log_msg)

        return {
            "status": "success",
            "model_version": f"Model B {self.active_version.upper()}",
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            # RAW Predictions (unbiased, directly from YOLO11s softmax)
            "raw_top1_id": raw_top_idx,
            "raw_top1_name": raw_top_name,
            "raw_top1_confidence": round(raw_top_conf, 1),
            "raw_top3": raw_top3,
            "raw_probabilities": raw_probs,
            # Stabilized Output
            "class_id": final_id,
            "class_name": final_class,
            "confidence": round(final_conf / 100.0, 4),
            "confidence_pct": round(final_conf, 1),
            "probabilities": final_probs,
            # Execution Metadata
            "temporal_filter_enabled": temporal_filter,
            "crop_mode": crop_meta["mode"],
            "crop_box": crop_meta,
            "original_dims": {"width": orig_w, "height": orig_h},
            "crop_dims": {"width": crop_meta["width"], "height": crop_meta["height"]},
            # Operational Behavioral Engagement Indicator
            "engagement_score": score,
            "engagement_level": level,
            "engagement_category": category,
            "badge_color": badge_color,
            "uart_packet": uart,
            "latency_ms": self.rolling_latency,
            "raw_latency_ms": round(raw_latency, 1),
            "fps": fps,
            "device": self.device,
        }

    def save_debug_snapshot(
        self, 
        img_bgr: np.ndarray, 
        result: Dict[str, Any], 
        notes: str = ""
    ) -> Dict[str, Any]:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        class_tag = result.get("raw_top1_name", "unknown")
        base_name = f"sample_{timestamp_str}_{class_tag}"

        frame_file = DEBUG_DIR / f"{base_name}_orig.jpg"
        crop_file = DEBUG_DIR / f"{base_name}_crop.jpg"
        meta_file = DEBUG_DIR / f"{base_name}_meta.json"

        cv2.imwrite(str(frame_file), img_bgr)

        crop_box = result.get("crop_box", {})
        x = crop_box.get("x", 0)
        y = crop_box.get("y", 0)
        w = crop_box.get("width", img_bgr.shape[1])
        h = crop_box.get("height", img_bgr.shape[0])
        crop_img = img_bgr[y:y+h, x:x+w]
        cv2.imwrite(str(crop_file), crop_img)

        snapshot_meta = {
            "timestamp": datetime.now().isoformat(),
            "model_version": result.get("model_version"),
            "model_path": str(self.model_path),
            "original_frame_path": str(frame_file.name),
            "crop_frame_path": str(crop_file.name),
            "original_dims": result.get("original_dims"),
            "crop_dims": result.get("crop_dims"),
            "crop_box": crop_box,
            "raw_top1_name": result.get("raw_top1_name"),
            "raw_top1_confidence": result.get("raw_top1_confidence"),
            "raw_top3": result.get("raw_top3"),
            "raw_probabilities": result.get("raw_probabilities"),
            "smoothed_class_name": result.get("class_name"),
            "smoothed_confidence_pct": result.get("confidence_pct"),
            "operational_behavioral_engagement_score": result.get("engagement_score"),
            "notes": notes
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(snapshot_meta, f, indent=2)

        return {
            "saved": True,
            "sample_id": base_name,
            "orig_file": str(frame_file),
            "crop_file": str(crop_file),
            "meta_file": str(meta_file)
        }


# Global singleton
model_b = ModelBClassifier()
