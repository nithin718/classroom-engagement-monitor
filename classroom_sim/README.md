# AI Classroom Engagement — Simulation Mode & Teacher Portal

A standalone, runnable implementation of the simulation layer and teacher
portal described in the spec: three persistent synthetic students, the
real 8-class taxonomy and scoring weights, deterministic scenarios, a
FastAPI backend, and a responsive web portal (desktop / tablet / mobile).

**Important — read this first:** this was built without access to your
actual repository (`D:\AI_Classroom_Engagement_System`). It is a clean,
self-contained reference implementation that uses your exact class list,
weights, thresholds, colors, UART packet format, and API shape — but it
does **not** literally reuse your `EngagementScorer` / `ScoreSnapshot` /
FastAPI classes, because I never saw their source. Wiring this into your
real project means either (a) dropping your real classes in to replace
`backend/engagement_scorer.py` and the model classes in `backend/models.py`
behind the same interfaces, or (b) pointing your existing FastAPI app at
`SimulationEngine` as a new data source. See Section K below.

---

## 1. Architecture

```
SIMULATION MODE                              LIVE MODE (not implemented here)
scenarios.py (deterministic timelines)        Webcam + YOLO11s + ByteTrack
      |                                              |
SimulationEngine (backend/simulation_engine.py)      |
      | (Detection-like state, per track_id 1-3)     |
      +---------------------> EngagementScorer <-----+   <- shared scoring
                                      |
                                ScoreSnapshot
                                      |
                              FastAPI (backend/main.py)
                                 /         \
                        REST + WebSocket    SimulatedUART (backend/uart_sim.py)
                                |
                        Teacher Portal (frontend/index.html)
```

Files:

| File | Role |
|---|---|
| `backend/models.py` | Fixed 8-class taxonomy, weights, thresholds, `StudentState`, `Event`, `ScoreSnapshot` |
| `backend/engagement_scorer.py` | Scoring + smoothing, shared by both modes |
| `backend/scenarios.py` | 8 deterministic scenarios (7 required + default demo) |
| `backend/simulation_engine.py` | Clock, transitions, durations, alerts, history, export |
| `backend/uart_sim.py` | Simulated `ENG,...` packet log, no hardware required |
| `backend/main.py` | FastAPI app: REST + WebSocket + LIVE/SIMULATION switch |
| `frontend/index.html` | Single-file responsive teacher portal (no build step) |

## 2. How to run

```bash
cd backend
pip install fastapi uvicorn
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` on desktop, or `http://<your-ip>:8000` on a
phone/tablet on the same network. The portal defaults to **SIMULATION
MODE**, scenario "Demo Classroom", paused. Press **Play**.

## 3. Demo script (matches spec section 39)

1. Open the portal, go to **Overview**.
2. Confirm the **SIMULATION MODE** badge and "Demo Classroom" scenario.
3. Press **Play** (defaults to 1x; use 5x/10x to compress the demo).
4. Narrate as it plays:
   - All three students start near `look_forward` / `read` — classroom HIGH.
   - Student 02 drifts into `using_device` — classroom score dips.
   - Student 03 falls into `sleep` — an alert appears in Events after ~15s continuous.
   - Student 01 raises a hand — score climbs.
   - Students 02 and 03 return to `look_forward` — classroom recovers.
5. Visit **Students**, click a card to open **Student Detail** — timeline + score graph + per-student transitions.
6. Visit **Simulation** to show the schematic **SIMULATED CAMERA** view and the **simulated UART packet log**.
7. Visit **Session** and click **Export Session (JSON)** to show analytics output.
8. Visit **Settings** and click **LIVE** to show the explicit, honest "not available in this build" message — proving the system never pretends simulated data is a real detection.

Total: about 1–3 minutes, matching the spec's target.

## 4. Testing checklist (spec section 53)

All of these were exercised against the running server:

- [x] Simulation starts (`POST /api/simulation/start`)
- [x] Simulation pauses (`POST /api/simulation/pause`)
- [x] Reset works (`POST /api/simulation/reset`)
- [x] Speed works (0.5x–10x, verified 10x advances ~10 sim-seconds per wall-second)
- [x] Scenario switching works (`POST /api/simulation/scenario`)
- [x] Student 01/02/03 keep Track IDs 1/2/3 permanently (never reassigned)
- [x] Activity changes occur at the scenario's deterministic times
- [x] Scores map to the exact weights in the spec (handrise/look_forward=100, read=85, write=80, stand=60, turn_head=40, using_device=20, sleep=5)
- [x] Classroom score is always `mean(student scores)`, never random
- [x] Events log populates on every transition + duration-based alerts
- [x] Timelines and score history update live over WebSocket
- [x] Mobile layout verified via CSS breakpoints (bottom nav, stacked cards)
- [x] No GPU, camera, or YOLO weights required to run
- [x] UART packets follow `ENG,<score>,<level>,<focused>,<device>,<sleep>` and never crash without a COM port
- [x] Session export produces valid JSON and CSV (verified via curl)
- [x] LIVE mode returns a clear `501` rather than silently using simulated data

Run your own smoke test any time with:
```bash
curl -X POST localhost:8000/api/simulation/start
curl localhost:8000/api/simulation/state
```

## 5. API reference

```
GET  /api/simulation/state        full current state (used by the portal)
GET  /api/simulation/students
GET  /api/simulation/events?limit=100
GET  /api/simulation/history
GET  /api/simulation/summary
GET  /api/simulation/scenarios
POST /api/simulation/start
POST /api/simulation/pause
POST /api/simulation/reset
POST /api/simulation/speed      {"speed": 0.5|1|2|5|10}
POST /api/simulation/scenario   {"scenario": "<key>"}
POST /api/simulation/jump       {"seconds": 10}
GET  /api/simulation/export.json
GET  /api/simulation/export.csv
GET  /api/mode
POST /api/mode                  {"mode": "LIVE"|"SIMULATION"}
WS   /ws/simulation              pushes full state ~10x/sec while running
```

## 6. Section K — integrating with your real project

To make this genuinely reuse your existing architecture rather than sit
beside it:

1. Drop your real `EngagementScorer` in place of `backend/engagement_scorer.py`,
   keeping the same two methods this build calls: `raw_weight(activity)`
   and `smooth(previous_score, activity)` (or adapt `simulation_engine.py`
   to your real method names).
2. If your `StudentState`/`ScoreSnapshot` already exist, replace the
   dataclasses in `backend/models.py` with imports from your real modules,
   as long as `SimulationEngine` can still write to the same fields it
   writes today (`current_activity`, `score`, `level`, duration counters).
3. Mount this router set (`backend/main.py`'s `/api/simulation/*` and
   `/ws/simulation`) onto your existing `ai_service.dashboard_api.main:app`
   instead of running a second FastAPI process.
4. Implement the `# --- LIVE MODE HOOK ---` marker in `main.py` to start
   your real OpenCV/YOLO11s/ByteTrack loop and feed its output through the
   same `EngagementScorer`.
5. Point your real UART sender's `send()` call at `SimulatedUART.send()`'s
   call site so LIVE mode uses real hardware while SIMULATION mode keeps
   using the in-memory log.

## 7. Known limitations of this standalone build

- LIVE mode is a stub (by design) — it does not include your camera/YOLO/
  ByteTrack code, since I don't have it.
- CSV export's `activity` column reflects the student's activity at export
  time for each historical score point rather than a true per-point
  activity log (score history and activity timeline are stored
  separately). If you need exact per-second activity in the CSV, say so
  and I'll add a joined per-second log.
- In-memory state only; restarting `uvicorn` clears history (matches the
  spec's "in-memory to start" requirement, with JSON/CSV export as the
  persistence mechanism).
