"""
main.py -- ai_service.dashboard_api equivalent for this simulation build.

Run with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser (desktop, tablet, or phone on
the same network) for the teacher portal.

This app owns ONE SimulationEngine instance (module-level singleton) and
exposes it through REST + WebSocket. A LIVE-mode switch is present and
clearly separated (see /api/mode) but LIVE mode itself -- webcam, YOLO11s,
ByteTrack, the trained best.pt weights -- is NOT implemented here, since
that code lives in your existing project and should be wired in at the
`# --- LIVE MODE HOOK ---` marker below rather than duplicated.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from simulation_engine import SimulationEngine
from scenarios import SCENARIOS

app = FastAPI(title="AI Classroom Engagement -- Simulation & Teacher Portal")

engine = SimulationEngine()

# Global mode flag. LIVE mode is intentionally not implemented in this
# standalone build -- switching to it returns a clear 501 rather than
# silently pretending simulated data is real. See section "LIVE MODE HOOK".
current_mode = {"mode": "SIMULATION"}

_ws_clients: Set[WebSocket] = set()
_broadcast_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Background simulation loop
# ---------------------------------------------------------------------------

async def _simulation_loop():
    """Advances the engine and broadcasts state ~10x/sec while running."""
    while True:
        try:
            engine.tick_realtime()
            if _ws_clients:
                payload = json.dumps(engine.state_dict())
                dead = []
                for ws in list(_ws_clients):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    _ws_clients.discard(ws)
        except Exception as exc:  # never let the loop die
            print(f"[simulation_loop] error: {exc}")
        await asyncio.sleep(0.1)


@app.on_event("startup")
async def on_startup():
    global _broadcast_task
    _broadcast_task = asyncio.create_task(_simulation_loop())


# ---------------------------------------------------------------------------
# Mode switch (section 48)
# ---------------------------------------------------------------------------

class ModeRequest(BaseModel):
    mode: str  # "LIVE" | "SIMULATION"


@app.get("/api/mode")
def get_mode():
    return current_mode


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    mode = req.mode.upper()
    if mode not in ("LIVE", "SIMULATION"):
        raise HTTPException(400, "mode must be LIVE or SIMULATION")
    if mode == "LIVE":
        # --- LIVE MODE HOOK ---
        # Wire your existing webcam + YOLO11s + ActivityDetector + ByteTrack
        # pipeline in here. Deliberately NOT implemented in this standalone
        # build so simulated data can never be mistaken for a real
        # detection. Switching modes must not destroy simulation state, so
        # `engine` is left untouched either way.
        raise HTTPException(
            501,
            "LIVE mode requires the real webcam/YOLO11s/ByteTrack pipeline "
            "from the main project and is not wired into this standalone "
            "simulation build. Integrate at the '# --- LIVE MODE HOOK ---' "
            "marker in main.py.",
        )
    current_mode["mode"] = "SIMULATION"
    return current_mode


# ---------------------------------------------------------------------------
# Simulation state / read endpoints
# ---------------------------------------------------------------------------

@app.get("/api/simulation/state")
def get_state():
    return engine.state_dict()


@app.get("/api/simulation/students")
def get_students():
    return engine.students_dict()


@app.get("/api/simulation/events")
def get_events(limit: int = 100):
    return engine.events_dict(limit=limit)


@app.get("/api/simulation/history")
def get_history():
    return engine.history_dict()


@app.get("/api/simulation/summary")
def get_summary():
    return engine.session_summary()


@app.get("/api/simulation/scenarios")
def get_scenarios():
    return {"scenarios": list(SCENARIOS.keys())}


# ---------------------------------------------------------------------------
# Simulation control endpoints
# ---------------------------------------------------------------------------

@app.post("/api/simulation/start")
def start_simulation():
    engine.play()
    return engine.state_dict()


@app.post("/api/simulation/pause")
def pause_simulation():
    engine.pause()
    return engine.state_dict()


@app.post("/api/simulation/reset")
def reset_simulation():
    engine.reset()
    return engine.state_dict()


class SpeedRequest(BaseModel):
    speed: float


@app.post("/api/simulation/speed")
def set_speed(req: SpeedRequest):
    engine.set_speed(req.speed)
    return engine.state_dict()


class ScenarioRequest(BaseModel):
    scenario: str


@app.post("/api/simulation/scenario")
def set_scenario(req: ScenarioRequest):
    try:
        engine.set_scenario(req.scenario)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return engine.state_dict()


class JumpRequest(BaseModel):
    seconds: float


@app.post("/api/simulation/jump")
def jump(req: JumpRequest):
    engine.jump(req.seconds)
    return engine.state_dict()


# ---------------------------------------------------------------------------
# Export (section 47)
# ---------------------------------------------------------------------------

@app.get("/api/simulation/export.json")
def export_json():
    data = engine.export_session()
    filename = f"simulation_session_{__import__('time').strftime('%Y-%m-%d')}.json"
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/simulation/export.csv")
def export_csv():
    rows = engine.export_csv_rows()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["student_id", "timestamp", "activity", "score", "level"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    buf.seek(0)
    filename = f"simulation_session_{__import__('time').strftime('%Y-%m-%d')}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# WebSocket (real-time push to the teacher portal)
# ---------------------------------------------------------------------------

@app.websocket("/ws/simulation")
async def ws_simulation(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps(engine.state_dict()))
        while True:
            # We don't expect client->server messages, but keep the socket
            # alive and drop it cleanly if the client disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(websocket)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
