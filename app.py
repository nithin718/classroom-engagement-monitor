"""
AI Classroom Engagement & Behavior Analytics Server
Featuring Model B YOLO11s-cls Single-Student Behavior Classifier + Multi-Student Simulation Engine

Endpoints:
- /                  -> Model B Live Inference Dashboard & Diagnostics
- /model             -> Model B Live Inference Dashboard & Diagnostics
- /portal            -> Teacher Engagement Portal
- /simulation        -> 2D Interactive Classroom Simulation
- /api/model/predict -> Real-time Model B YOLO Inference
- /api/model/info    -> Model B Architecture & Metrics
- /api/model/samples -> Pre-extracted Benchmark Crops
- /api/model/direct-test -> Single Image Direct Model B Test
- /api/model/debug-sample -> Save Debug Snapshot to debug_samples/
- /api/model/crop-compare -> Side-by-side Crop Framing Comparison
- /api/simulation/*  -> Simulation controls & state
"""
from __future__ import annotations

import asyncio
import os
import time
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from simulation_engine import sim_engine, SCENARIOS
from model_b_engine import model_b, MODEL_PATH, CANONICAL_CLASSES

app = FastAPI(title="AI Classroom Engagement & Model B Diagnostics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    sim_engine.start()


# Current system mode: "SIMULATION" or "LIVE"
CURRENT_MODE = "LIVE"

# Shared live state fed by Model B webcam feed
latest_live_prediction = {
    "class_name": "look_forward",
    "confidence_pct": 95.0,
    "raw_class_name": "look_forward",
    "raw_confidence_pct": 95.0,
    "engagement_score": 90.0,
    "engagement_level": "HIGH",
    "uart_packet": {"hex": "0xAA 0x5A 0x02 0x01 0xF6"},
    "updated_at": time.time()
}


# --- Pydantic Request Models ---

class ModeRequest(BaseModel):
    mode: str

class SpeedRequest(BaseModel):
    speed: float

class ScenarioRequest(BaseModel):
    scenario: str

class JumpRequest(BaseModel):
    seconds: float

class PredictRequest(BaseModel):
    image: Optional[str] = None          # Base64 string or data URL
    sample: Optional[str] = None         # Sample filename
    crop_mode: Optional[str] = "mode_a"  # mode_a, mode_b, mode_c
    temporal_filter: Optional[bool] = True
    sync_classroom: Optional[bool] = True
    save_snapshot: Optional[bool] = False
    force_class: Optional[str] = None
    notes: Optional[str] = ""

class DebugSnapshotRequest(BaseModel):
    image: str
    notes: Optional[str] = ""
    crop_mode: Optional[str] = "mode_a"


# --- Model B Dedicated Endpoints ---

class SwitchModelRequest(BaseModel):
    version: str

@app.get("/api/model/info")
def get_model_info():
    return {
        "name": "Model B: Single-Student 8-Class Behavior Classifier",
        "architecture": "YOLO11s Classification",
        "weights": str(model_b.model_path),
        "active_version": getattr(model_b, "active_version", "v2").upper(),
        "available_versions": ["V1", "V2"],
        "input_resolution": [224, 224],
        "classes": CANONICAL_CLASSES,
        "device": model_b.device,
        "is_loaded": model_b.is_loaded,
        "model_b_v2_benchmark": {
            "title": "Model B V2 — Current Webcam Classifier",
            "evaluation_dataset": "Held-out real-webcam test set",
            "test_samples": 232,
            "accuracy": 99.57,
            "macro_precision": 0.9957,
            "macro_recall": 0.9955,
            "macro_f1": 0.9955,
            "training_epochs": 50,
            "dataset_total": 1527,
            "dataset_splits": {"train": 1075, "val": 220, "test": 232},
            "per_class": [
                {"class_name": "handrise", "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "support": 26},
                {"class_name": "look_forward", "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "support": 31},
                {"class_name": "read", "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "support": 27},
                {"class_name": "sleep", "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "support": 30},
                {"class_name": "stand", "precision": 1.0000, "recall": 0.9643, "f1": 0.9818, "support": 28},
                {"class_name": "turn_head", "precision": 0.9655, "recall": 1.0000, "f1": 0.9825, "support": 28},
                {"class_name": "using_device", "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "support": 28},
                {"class_name": "write", "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "support": 34}
            ],
            "confusion_matrix": [
                [26, 0,  0,  0,  0,  0,  0,  0],
                [0,  31, 0,  0,  0,  0,  0,  0],
                [0,  0,  27, 0,  0,  0,  0,  0],
                [0,  0,  0,  30, 0,  0,  0,  0],
                [0,  0,  0,  0,  27, 1,  0,  0],
                [0,  0,  0,  0,  0,  28, 0,  0],
                [0,  0,  0,  0,  0,  0,  28, 0],
                [0,  0,  0,  0,  0,  0,  0,  34]
            ]
        },
        "model_b_v1_benchmark": {
            "title": "Model B V1 — Historical Reference",
            "evaluation_dataset": "Historical V1 test crops",
            "test_samples": 668,
            "accuracy": 94.61,
            "macro_f1": 0.8736
        },
        "model_a_benchmark": {
            "title": "Model A — Multi-Student Behavior Detector",
            "architecture": "YOLO11s",
            "evaluation_dataset": "Held-out multi-person test set",
            "precision": 77.32,
            "recall": 71.91,
            "map50": 76.61,
            "map50_95": 51.13,
            "per_class_ap50": {
                "handrise": 85.95,
                "look_forward": 75.52,
                "read": 84.18,
                "sleep": 92.19,
                "stand": 59.05,
                "turn_head": 52.17,
                "using_device": 78.38,
                "write": 85.42
            },
            "difficult_classes": ["turn_head", "stand"]
        }
    }

@app.post("/api/model/switch")
def switch_model(req: SwitchModelRequest):
    res = model_b.switch_model_version(req.version)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res

@app.get("/api/model/samples")
def get_model_samples():
    manifest_path = Path(__file__).parent / "static" / "samples" / "manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

@app.post("/api/model/predict")
def predict_behavior(req: PredictRequest):
    global latest_live_prediction
    try:
        img_input = None
        if req.sample:
            sample_path = Path(__file__).parent / "static" / "samples" / req.sample
            if not sample_path.exists():
                raise HTTPException(status_code=404, detail="Sample image not found")
            img_input = str(sample_path)
        elif req.image:
            img_input = req.image
        else:
            raise HTTPException(status_code=400, detail="Must provide either 'image' or 'sample'")

        crop_mode = req.crop_mode or "mode_a"
        temporal_filter = True if req.temporal_filter is None else req.temporal_filter

        result = model_b.predict_image(
            img_input, 
            crop_mode=crop_mode, 
            temporal_filter=temporal_filter,
            
        )

        # Save snapshot if requested
        if req.save_snapshot:
            try:
                if isinstance(img_input, str) and img_input.startswith("data:image"):
                    data = base64.b64decode(img_input.split(",", 1)[1])
                    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                elif isinstance(img_input, str) and os.path.exists(img_input):
                    bgr = cv2.imread(img_input)
                else:
                    data = base64.b64decode(img_input)
                    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if bgr is not None:
                    snap_res = model_b.save_debug_snapshot(bgr, result, notes=req.notes or "")
                    result["debug_snapshot"] = snap_res
            except Exception as e:
                print(f"Warning: Failed to save snapshot: {e}")

        # Update live state if synced
        if req.sync_classroom:
            latest_live_prediction = {
                "class_name": result["class_name"],
                "confidence_pct": result["confidence_pct"],
                "raw_class_name": result["raw_top1_name"],
                "raw_confidence_pct": result["raw_top1_confidence"],
                "engagement_score": result["engagement_score"],
                "engagement_level": result["engagement_level"],
                "uart_packet": result["uart_packet"],
                "updated_at": time.time()
            }
            # Also sync into simulated Student #1 if running
            if 1 in sim_engine.students:
                s1 = sim_engine.students[1]
                s1.current_activity = result["class_name"]
                s1.confidence = result["confidence"]
                s1.score = result["engagement_score"]
                s1.level = result["engagement_level"]

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/debug-sample")
def save_debug_sample(req: DebugSnapshotRequest):
    try:
        if req.image.startswith("data:image"):
            data = base64.b64decode(req.image.split(",", 1)[1])
        else:
            data = base64.b64decode(req.image)
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        pred = model_b.predict_image(bgr, crop_mode=req.crop_mode or "mode_a")
        snap_info = model_b.save_debug_snapshot(bgr, pred, notes=req.notes or "")
        return {
            "status": "success",
            "snapshot": snap_info,
            "prediction": pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/crop-compare")
def compare_crops(req: PredictRequest):
    try:
        if not req.image:
            raise HTTPException(status_code=400, detail="Must provide 'image'")
        if req.image.startswith("data:image"):
            data = base64.b64decode(req.image.split(",", 1)[1])
        else:
            data = base64.b64decode(req.image)
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        res_a = model_b.predict_image(bgr, crop_mode="mode_a", temporal_filter=False)
        res_b = model_b.predict_image(bgr, crop_mode="mode_b", temporal_filter=False)
        res_c = model_b.predict_image(bgr, crop_mode="mode_c", temporal_filter=False)

        return {
            "status": "success",
            "crops": {
                "mode_a": res_a,
                "mode_b": res_b,
                "mode_c": res_c
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DirectTestRequest(BaseModel):
    image: str
    crop_mode: Optional[str] = "mode_a"
    temporal_filter: Optional[bool] = False

@app.post("/api/model/direct-test")
def direct_test(req: DirectTestRequest):
    try:
        raw_b64 = req.image.split(",", 1)[1] if "," in req.image else req.image
        data = base64.b64decode(raw_b64)
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        result = model_b.predict_image(
            bgr, 
            crop_mode=req.crop_mode or "mode_a", 
            temporal_filter=req.temporal_filter or False
        )
        return {
            "status": "success",
            "direct_test": {
                "model_path": str(model_b.model_path),
                "raw_top1_name": result["raw_top1_name"],
                "raw_top1_confidence": result["raw_top1_confidence"],
                "raw_top3": result["raw_top3"],
                "raw_probabilities": result["raw_probabilities"],
                "smoothed_class_name": result["class_name"],
                "smoothed_confidence": result["confidence_pct"],
                "crop_dims": result["crop_dims"],
                "crop_mode": result["crop_mode"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/temporal/reset")
def reset_temporal():
    model_b.debouncer.reset()
    return {"status": "reset", "active_class": model_b.debouncer.current_class}


# --- Mode & Simulation API Routes ---

@app.get("/api/status")
def get_status():
    return {
        "mode": CURRENT_MODE,
        "running": sim_engine.running if CURRENT_MODE == "SIMULATION" else True,
        "speed": sim_engine.speed,
        "scenario": sim_engine.scenario_key,
        "model_loaded": model_b.is_loaded,
        "available_scenarios": [
            {"key": k, "title": v["title"], "description": v["description"], "duration": v["duration"]}
            for k, v in SCENARIOS.items()
        ]
    }

@app.post("/api/mode")
def set_mode(req: ModeRequest):
    global CURRENT_MODE
    mode_upper = req.mode.upper()
    if mode_upper not in ("SIMULATION", "LIVE"):
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'SIMULATION' or 'LIVE'.")
    CURRENT_MODE = mode_upper
    if CURRENT_MODE == "SIMULATION":
        sim_engine.start()
    return {"status": "ok", "mode": CURRENT_MODE}

@app.get("/api/simulation/state")
def get_simulation_state():
    return sim_engine.get_state()

@app.get("/api/simulation/students")
def get_simulation_students():
    return {"students": [s for s in sim_engine.get_state()["students"]]}

@app.get("/api/simulation/student/{student_id}")
def get_student_detail(student_id: int):
    if student_id not in sim_engine.students:
        raise HTTPException(status_code=404, detail="Student not found")
    state = sim_engine.get_state()
    for s in state["students"]:
        if s["student_id"] == student_id:
            student_events = [e for e in state["recent_events"] if e["student_id"] == student_id]
            return {"student": s, "events": student_events}
    raise HTTPException(status_code=404, detail="Student not found")

@app.post("/api/simulation/start")
def start_simulation():
    sim_engine.start()
    return {"status": "started", "running": True}

@app.post("/api/simulation/pause")
def pause_simulation():
    sim_engine.pause()
    return {"status": "paused", "running": False}

@app.post("/api/simulation/reset")
def reset_simulation():
    sim_engine.reset()
    return {"status": "reset", "sim_time": 0.0}

@app.post("/api/simulation/speed")
def change_speed(req: SpeedRequest):
    sim_engine.set_speed(req.speed)
    return {"status": "ok", "speed": sim_engine.speed}

@app.post("/api/simulation/scenario")
def change_scenario(req: ScenarioRequest):
    sim_engine.set_scenario(req.scenario)
    return {"status": "ok", "scenario": sim_engine.scenario_key}

@app.post("/api/simulation/jump")
def jump_time(req: JumpRequest):
    sim_engine.jump(req.seconds)
    return {"status": "ok", "sim_time": round(sim_engine.sim_time, 1)}

@app.get("/api/simulation/export/json")
def export_json():
    json_str = sim_engine.export_json()
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=simulation_session_{int(sim_engine.sim_time)}s.json"}
    )

@app.get("/api/simulation/export/csv")
def export_csv():
    csv_str = sim_engine.export_csv()
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=simulation_session_{int(sim_engine.sim_time)}s.csv"}
    )


# --- Backward Compatibility for Existing Teacher Portal / ai_service Routes ---

@app.get("/session/current")
def current_session():
    if CURRENT_MODE == "SIMULATION":
        state = sim_engine.get_state()
        c = state["classroom"]
        class_counts = {
            "look_forward": 0, "read": 0, "write": 0, "handrise": 0,
            "turn_head": 0, "stand": 0, "using_device": 0, "sleep": 0
        }
        for s in state["students"]:
            act = s["current_activity"]
            if act in class_counts:
                class_counts[act] += 1
                
        return {
            "running": sim_engine.running,
            "mode": "SIMULATION",
            "session_id": 999,
            "snapshot": {
                "timestamp": state["sim_time"],
                "score": c["score"],
                "level": c["level"],
                "tracked_student_count": c["total_students"],
                "class_counts": class_counts,
                "uart_packet": c["uart_packet"]
            }
        }
    else:
        act = latest_live_prediction["class_name"]
        class_counts = {
            "look_forward": 0, "read": 0, "write": 0, "handrise": 0,
            "turn_head": 0, "stand": 0, "using_device": 0, "sleep": 0
        }
        if act in class_counts:
            class_counts[act] = 1

        return {
            "running": True,
            "mode": "LIVE",
            "session_id": 101,
            "snapshot": {
                "timestamp": 12.5,
                "score": latest_live_prediction["engagement_score"],
                "level": latest_live_prediction["engagement_level"],
                "tracked_student_count": 1,
                "class_counts": class_counts,
                "uart_packet": latest_live_prediction.get("uart_packet", {}).get("hex", "0xAA 0x5A 0x02 0x01 0xF6")
            }
        }

@app.post("/session/start")
def start_session_compat():
    sim_engine.start()
    return {"session_id": 999, "status": "started", "mode": CURRENT_MODE}

@app.post("/session/stop")
def stop_session_compat():
    sim_engine.pause()
    return {"session_id": 999, "status": "stopped", "mode": CURRENT_MODE}


# --- Model B v2 Dataset Collection & Baseline Evaluation Routes ---
from dataset_manager import dataset_manager, V1_MODEL_PATH

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mount both static and data directories
STATIC_DIR = Path(__file__).parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


class CollectSampleRequest(BaseModel):
    image_base64: str
    label: str
    crop_mode: str = "mode_b"
    session_id: str = "session_1"
    person_id: str = "student_1"
    notes: Optional[str] = ""

class SplitDatasetRequest(BaseModel):
    strategy: str = "session_aware"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

class DeleteSampleRequest(BaseModel):
    sample_id: str


@app.get("/api/dataset/stats")
def get_dataset_stats():
    return dataset_manager.get_stats()

@app.post("/api/dataset/collect")
def collect_sample(payload: CollectSampleRequest):
    img_bgr = None
    try:
        raw = payload.image_base64
        if raw.startswith("data:image"):
            _, encoded = raw.split(",", 1)
        else:
            encoded = raw
        arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding error: {e}")

    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(status_code=400, detail="Invalid or empty image decoded")

    try:
        res = dataset_manager.save_sample(
            img_bgr=img_bgr,
            label=payload.label,
            crop_mode=payload.crop_mode,
            session_id=payload.session_id,
            person_id=payload.person_id,
            notes=payload.notes or ""
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/dataset/delete-last")
def delete_last_sample():
    return dataset_manager.delete_last_sample()

@app.post("/api/dataset/delete-sample")
def delete_sample(payload: DeleteSampleRequest):
    return dataset_manager.delete_sample_by_id(payload.sample_id)

@app.post("/api/dataset/split")
def split_dataset(payload: SplitDatasetRequest):
    return dataset_manager.create_leakage_safe_split(
        strategy=payload.strategy,
        train_ratio=payload.train_ratio,
        val_ratio=payload.val_ratio,
        test_ratio=payload.test_ratio,
        seed=payload.seed
    )

@app.post("/api/dataset/evaluate-v1-baseline")
def evaluate_v1_baseline(split: str = "test"):
    try:
        report = dataset_manager.evaluate_model_on_split(
            weights_path=V1_MODEL_PATH,
            split=split,
            output_prefix="v1_baseline"
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset/package")
def package_dataset_endpoint():
    try:
        from package_dataset import run_pipeline
        res = run_pipeline()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/download-zip")
def download_dataset_zip():
    zip_path = BASE_DIR / "Model_B_Webcam_V2_Dataset.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Dataset ZIP has not been generated yet. Run packaging first.")
    return FileResponse(path=str(zip_path), filename="Model_B_Webcam_V2_Dataset.zip", media_type="application/zip")

@app.get("/api/dataset/audit-image")
def get_audit_image():
    audit_path = BASE_DIR / "model_b_webcam_v2_audit.png"
    if not audit_path.exists():
        raise HTTPException(status_code=404, detail="Audit image not found")
    return FileResponse(path=str(audit_path), media_type="image/png")

@app.get("/rapid-collect", response_class=HTMLResponse)
def serve_rapid_collector():
    collector_path = STATIC_DIR / "rapid_collector.html"
    if collector_path.exists():
        return HTMLResponse(content=collector_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Rapid Collector Loading...</h1>")

@app.get("/collect", response_class=HTMLResponse)
def serve_data_collector():
    collector_path = STATIC_DIR / "collector.html"
    if collector_path.exists():
        return HTMLResponse(content=collector_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dataset Collector Loading...</h1>")

@app.get("/", response_class=HTMLResponse)
@app.get("/model", response_class=HTMLResponse)
def serve_model_dashboard():
    model_path = STATIC_DIR / "model_dashboard.html"
    if model_path.exists():
        return HTMLResponse(content=model_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Model B Dashboard Loading...</h1>")

@app.get("/portal", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_portal():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Teacher Portal Loading...</h1>")

@app.get("/simulation", response_class=HTMLResponse)
def serve_simulation():
    sim_path = STATIC_DIR / "simulation.html"
    if sim_path.exists():
        return HTMLResponse(content=sim_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Simulation Loading...</h1>")


if __name__ == "__main__":
    import uvicorn
    sim_engine.start()
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Model B Diagnostics & AI Classroom Server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
