"""
SimulationEngine: Realistic software simulation for the AI Classroom Engagement Monitoring System.

Simulates 3 distinct students with persistent track IDs, deterministic behavior timelines,
dynamic engagement scoring matching the project's exact 8-class taxonomy, alerts,
event transitions, and simulated UART packet output.
"""
from __future__ import annotations

import time
import json
import csv
import io
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Exact 8-class taxonomy & weights
CLASS_NAMES = [
    "handrise",       # 0
    "look_forward",   # 1
    "read",           # 2
    "sleep",          # 3
    "stand",          # 4
    "turn_head",      # 5
    "using_device",   # 6
    "write",          # 7
]

CLASS_WEIGHTS = {
    "handrise": 100.0,
    "look_forward": 100.0,
    "read": 85.0,
    "write": 80.0,
    "stand": 60.0,
    "turn_head": 40.0,
    "using_device": 20.0,
    "sleep": 5.0,
}

CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

def score_to_level(score: float) -> str:
    if score >= 66.0:
        return "HIGH"
    elif score >= 33.0:
        return "MEDIUM"
    return "LOW"


# Bounding boxes for schematic camera representation: [x1, y1, x2, y2]
STUDENT_BBOXES = {
    1: [60, 140, 220, 380],   # Student 01 (Left Desk)
    2: [260, 130, 420, 370],  # Student 02 (Center Desk)
    3: [460, 145, 620, 385],  # Student 03 (Right Desk)
}

# Pre-defined deterministic scenarios
SCENARIOS = {
    "normal": {
        "title": "Normal Lecture (Standard Active Engagement)",
        "description": "Student 1 highly attentive, Student 2 balanced, Student 3 occasionally distracted.",
        "duration": 180,
        "timelines": {
            1: [
                (0, "look_forward"), (25, "read"), (50, "write"), (75, "handrise"),
                (85, "look_forward"), (115, "read"), (145, "turn_head"), (155, "look_forward")
            ],
            2: [
                (0, "look_forward"), (20, "read"), (45, "write"), (65, "turn_head"),
                (80, "using_device"), (95, "look_forward"), (125, "read"), (155, "write")
            ],
            3: [
                (0, "look_forward"), (15, "turn_head"), (35, "using_device"), (60, "sleep"),
                (85, "stand"), (105, "using_device"), (135, "sleep"), (165, "look_forward")
            ]
        }
    },
    "distraction_event": {
        "title": "Distraction Event & Recovery",
        "description": "A sudden distraction occurs at 25s causing phones and dozing, followed by teacher re-engagement at 75s.",
        "duration": 120,
        "timelines": {
            1: [
                (0, "read"), (30, "turn_head"), (45, "read"), (75, "handrise"), (85, "look_forward")
            ],
            2: [
                (0, "look_forward"), (25, "using_device"), (50, "turn_head"), (75, "look_forward"), (95, "write")
            ],
            3: [
                (0, "turn_head"), (20, "using_device"), (40, "sleep"), (75, "stand"), (85, "look_forward")
            ]
        }
    },
    "question_session": {
        "title": "Teacher Question & Hand-Raising Event",
        "description": "Interactive Q&A period with multiple hand-raises, active reading and note-taking.",
        "duration": 90,
        "timelines": {
            1: [(0, "look_forward"), (15, "handrise"), (25, "look_forward"), (50, "write"), (75, "handrise")],
            2: [(0, "read"), (20, "look_forward"), (35, "handrise"), (45, "write"), (70, "look_forward")],
            3: [(0, "using_device"), (15, "turn_head"), (30, "look_forward"), (55, "read"), (75, "write")]
        }
    },
    "low_engagement": {
        "title": "Low Engagement Fatigue Period",
        "description": "End-of-day lethargy: multiple students dozing, off-task device usage, and turning away.",
        "duration": 120,
        "timelines": {
            1: [(0, "look_forward"), (30, "read"), (60, "turn_head"), (80, "read"), (100, "look_forward")],
            2: [(0, "turn_head"), (25, "using_device"), (55, "sleep"), (85, "using_device")],
            3: [(0, "sleep"), (35, "turn_head"), (50, "sleep"), (80, "stand"), (95, "sleep")]
        }
    }
}


@dataclass
class ActivityTimelineSegment:
    activity: str
    start_time: float
    end_time: float
    duration: float


@dataclass
class StudentProfile:
    student_id: int
    track_id: int
    name: str
    current_activity: str = "look_forward"
    current_score: float = 100.0
    level: str = "HIGH"
    confidence: float = 0.94
    bbox: List[int] = field(default_factory=list)
    
    # Durations in seconds
    focused_time: float = 0.0
    device_time: float = 0.0
    sleep_time: float = 0.0
    stand_time: float = 0.0
    turn_head_time: float = 0.0
    handrise_count: int = 0
    
    # State tracking
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    score_history: List[Dict[str, Any]] = field(default_factory=list)
    activity_start_time: float = 0.0


@dataclass
class SimEvent:
    timestamp: float
    sim_time: float
    student_id: int
    student_name: str
    track_id: int
    from_activity: str
    to_activity: str
    message: str


class SimulationEngine:
    def __init__(self):
        self.running: bool = False
        self.sim_time: float = 0.0
        self.speed: float = 1.0
        self.scenario_key: str = "normal"
        self.last_tick_wall_time: float = time.time()
        
        self.students: Dict[int, StudentProfile] = {}
        self.events: List[SimEvent] = []
        self.alerts: List[Dict[str, Any]] = []
        self.classroom_history: List[Dict[str, Any]] = []
        
        # Classroom aggregates
        self.classroom_score: float = 85.0
        self.classroom_level: str = "HIGH"
        self.focused_count: int = 3
        self.device_count: int = 0
        self.sleep_count: int = 0
        self.stand_count: int = 0
        self.turn_head_count: int = 0
        
        # UART simulation packet
        self.last_uart_packet: str = "ENG,85,HIGH,3,0,0"
        
        self.reset()

    def reset(self):
        """Resets the simulation to t=0 under the active scenario."""
        self.sim_time = 0.0
        self.last_tick_wall_time = time.time()
        self.events.clear()
        self.alerts.clear()
        self.classroom_history.clear()
        
        # Initialize the 3 students with distinct initial states
        self.students = {
            1: StudentProfile(student_id=1, track_id=1, name="Student 01", bbox=STUDENT_BBOXES[1]),
            2: StudentProfile(student_id=2, track_id=2, name="Student 02", bbox=STUDENT_BBOXES[2]),
            3: StudentProfile(student_id=3, track_id=3, name="Student 03", bbox=STUDENT_BBOXES[3]),
        }
        
        # Apply initial activities from scenario
        scenario = SCENARIOS.get(self.scenario_key, SCENARIOS["normal"])
        for sid, student in self.students.items():
            tl = scenario["timelines"].get(sid, [(0, "look_forward")])
            init_act = tl[0][1]
            student.current_activity = init_act
            student.current_score = CLASS_WEIGHTS[init_act]
            student.level = score_to_level(student.current_score)
            student.activity_start_time = 0.0
            student.score_history = [{"time": 0.0, "score": student.current_score}]
            student.timeline = [{"activity": init_act, "start": 0.0, "end": 0.0}]

        self._update_aggregates()
        self._record_classroom_history()

    def set_scenario(self, scenario_key: str):
        if scenario_key in SCENARIOS:
            self.scenario_key = scenario_key
            self.reset()

    def set_speed(self, speed: float):
        if speed in [0.5, 1.0, 2.0, 5.0, 10.0]:
            self.speed = speed

    def start(self):
        self.running = True
        self.last_tick_wall_time = time.time()

    def pause(self):
        self.running = False

    def jump(self, seconds: float):
        """Advance the simulation time by delta seconds."""
        target = self.sim_time + seconds
        step = 1.0
        while self.sim_time < target:
            dt = min(step, target - self.sim_time)
            self._step(dt)

    def tick(self):
        """Advances simulation based on elapsed wall-clock time and speed multiplier."""
        now = time.time()
        wall_dt = now - self.last_tick_wall_time
        self.last_tick_wall_time = now
        
        if not self.running:
            return
            
        dt = wall_dt * self.speed
        # Cap dt to avoid huge jumps on tab freeze
        dt = min(dt, 2.0)
        self._step(dt)

    def _step(self, dt: float):
        self.sim_time += dt
        scenario = SCENARIOS.get(self.scenario_key, SCENARIOS["normal"])
        
        # Loop scenarios once duration reached
        duration = scenario["duration"]
        if self.sim_time > duration:
            self.sim_time = self.sim_time % duration

        # Update each student based on scenario timeline
        for sid, student in self.students.items():
            tl = scenario["timelines"].get(sid, [])
            expected_act = student.current_activity
            
            for t_mark, act in tl:
                if self.sim_time >= t_mark:
                    expected_act = act
                else:
                    break
                    
            # Check for activity transition
            if expected_act != student.current_activity:
                old_act = student.current_activity
                student.current_activity = expected_act
                student.activity_start_time = self.sim_time
                
                # Close previous timeline block
                if student.timeline:
                    student.timeline[-1]["end"] = round(self.sim_time, 1)
                student.timeline.append({"activity": expected_act, "start": round(self.sim_time, 1), "end": round(self.sim_time, 1)})
                
                # Target instantaneous score
                target_score = CLASS_WEIGHTS[expected_act]
                
                # Add transition event
                event = SimEvent(
                    timestamp=time.time(),
                    sim_time=round(self.sim_time, 1),
                    student_id=sid,
                    student_name=student.name,
                    track_id=student.track_id,
                    from_activity=old_act,
                    to_activity=expected_act,
                    message=f"{student.name} transitioned from {old_act} → {expected_act}"
                )
                self.events.insert(0, event)
                if len(self.events) > 50:
                    self.events.pop()
                    
                if expected_act == "handrise":
                    student.handrise_count += 1
            else:
                # Update current timeline end
                if student.timeline:
                    student.timeline[-1]["end"] = round(self.sim_time, 1)

            # Smooth score adjustment (EMA-like transition towards current behavior weight)
            target = CLASS_WEIGHTS[student.current_activity]
            # Smooth towards target with alpha based on dt
            alpha = min(1.0, 0.4 * dt * self.speed)
            student.current_score = round(student.current_score * (1 - alpha) + target * alpha, 1)
            student.level = score_to_level(student.current_score)
            
            # Accumulate behavior durations
            act = student.current_activity
            if act in ("look_forward", "read", "write", "handrise"):
                student.focused_time += dt
            elif act == "using_device":
                student.device_time += dt
            elif act == "sleep":
                student.sleep_time += dt
            elif act == "stand":
                student.stand_time += dt
            elif act == "turn_head":
                student.turn_head_time += dt

            # Record score history every ~2 simulated seconds
            if not student.score_history or (self.sim_time - student.score_history[-1]["time"]) >= 2.0:
                student.score_history.append({"time": round(self.sim_time, 1), "score": student.current_score})
                if len(student.score_history) > 100:
                    student.score_history.pop(0)

        self._update_aggregates()
        self._check_alerts()
        
        # Record classroom score history
        if not self.classroom_history or (self.sim_time - self.classroom_history[-1]["time"]) >= 2.0:
            self._record_classroom_history()

    def _update_aggregates(self):
        scores = [s.current_score for s in self.students.values()]
        self.classroom_score = round(sum(scores) / len(scores), 1) if scores else 50.0
        self.classroom_level = score_to_level(self.classroom_score)
        
        self.focused_count = 0
        self.device_count = 0
        self.sleep_count = 0
        self.stand_count = 0
        self.turn_head_count = 0
        
        for s in self.students.values():
            act = s.current_activity
            if act in ("look_forward", "read", "write", "handrise"):
                self.focused_count += 1
            elif act == "using_device":
                self.device_count += 1
            elif act == "sleep":
                self.sleep_count += 1
            elif act == "stand":
                self.stand_count += 1
            elif act == "turn_head":
                self.turn_head_count += 1
                
        # Format UART packet: ENG,<score>,<level>,<focused>,<device>,<sleep>\n
        self.last_uart_packet = f"ENG,{int(round(self.classroom_score))},{self.classroom_level},{self.focused_count},{self.device_count},{self.sleep_count}"

    def _check_alerts(self):
        """Generates threshold-based alerts (e.g. continuous sleeping, prolonged device use)."""
        for sid, s in self.students.items():
            current_duration = self.sim_time - s.activity_start_time
            if s.current_activity == "sleep" and current_duration >= 15.0:
                self._push_alert("SLEEPING_ALERT", f"{s.name} has been sleeping for {int(current_duration)}s", "warning")
            elif s.current_activity == "using_device" and current_duration >= 20.0:
                self._push_alert("DEVICE_ALERT", f"{s.name} has been on a device for {int(current_duration)}s", "warning")
                
        if self.classroom_score < 33.0:
            self._push_alert("CLASSROOM_LOW", f"Classroom engagement dropped into LOW zone ({self.classroom_score:.1f})", "danger")

    def _push_alert(self, code: str, msg: str, severity: str):
        # Debounce: avoid same alert within 20s
        for a in self.alerts:
            if a["code"] == code and (self.sim_time - a["sim_time"]) < 20.0:
                return
        alert_obj = {
            "id": int(time.time() * 1000),
            "code": code,
            "message": msg,
            "severity": severity,
            "sim_time": round(self.sim_time, 1),
            "timestamp": time.strftime("%H:%M:%S")
        }
        self.alerts.insert(0, alert_obj)
        if len(self.alerts) > 10:
            self.alerts.pop()

    def _record_classroom_history(self):
        self.classroom_history.append({
            "time": round(self.sim_time, 1),
            "score": self.classroom_score,
            "level": self.classroom_level,
            "focused": self.focused_count,
            "device": self.device_count,
            "sleep": self.sleep_count
        })
        if len(self.classroom_history) > 100:
            self.classroom_history.pop(0)

    def get_state(self) -> Dict[str, Any]:
        """Returns the full state payload for dashboard API consumption."""
        return {
            "mode": "SIMULATION",
            "running": self.running,
            "sim_time": round(self.sim_time, 1),
            "speed": self.speed,
            "scenario": self.scenario_key,
            "scenario_title": SCENARIOS.get(self.scenario_key, {}).get("title", ""),
            "classroom": {
                "score": self.classroom_score,
                "level": self.classroom_level,
                "total_students": len(self.students),
                "focused_count": self.focused_count,
                "device_count": self.device_count,
                "sleep_count": self.sleep_count,
                "stand_count": self.stand_count,
                "turn_head_count": self.turn_head_count,
                "uart_packet": self.last_uart_packet
            },
            "students": [asdict(s) for s in self.students.values()],
            "recent_events": [asdict(e) for e in self.events[:15]],
            "alerts": self.alerts[:5],
            "history": self.classroom_history[-30:]
        }

    def export_json(self) -> str:
        data = {
            "session_export": {
                "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenario": self.scenario_key,
                "simulated_duration_seconds": round(self.sim_time, 1),
                "classroom_final_score": self.classroom_score,
                "classroom_final_level": self.classroom_level,
                "students": [asdict(s) for s in self.students.values()],
                "events_log": [asdict(e) for e in self.events],
                "classroom_history": self.classroom_history
            }
        }
        return json.dumps(data, indent=2)

    def export_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["sim_time", "student_id", "name", "track_id", "activity", "score", "level"])
        for s in self.students.values():
            writer.writerow([
                round(self.sim_time, 1), s.student_id, s.name, s.track_id,
                s.current_activity, s.current_score, s.level
            ])
        return output.getvalue()


# Global singleton instance
sim_engine = SimulationEngine()
