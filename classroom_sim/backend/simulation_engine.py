"""
simulation_engine.py

    SIMULATION SOURCE
          |
    Synthetic Student Detection Events   (Detection objects, track_id 1-3)
          |
    Activity/Tracking-compatible representation   (StudentState)
          |
    EngagementScorer (shared with LIVE mode)
          |
    ScoreSnapshot
          |
    FastAPI -> Teacher Portal / UART

SimulationEngine owns:
  - the internal clock (play / pause / reset / speed)
  - the three persistent synthetic student identities (track 1/2/3 ==
    Student 01/02/03 -- never reassigned, standing in for ByteTrack's
    identity persistence)
  - scenario selection
  - event + alert generation
  - classroom + per-student history
  - session export

It never touches the camera, YOLO, or ByteTrack code paths. In the real
project, this module is the thing that gets swapped out for the real
OpenCV/YOLO11s/ActivityDetector/ByteTrack stack when MODE=LIVE.
"""

from __future__ import annotations

import time
import json
from dataclasses import asdict
from typing import Dict, List, Optional

from models import (
    ActivityClass, ACTIVITY_CLASS_IDS, EngagementLevel,
    StudentState, TimelineSegment, ScorePoint, Event,
    ScoreSnapshot, ClassroomHistoryPoint, Detection,
)
from engagement_scorer import EngagementScorer
from scenarios import SCENARIOS, SCENARIO_LABELS, DEFAULT_SCENARIO, activity_at
from uart_sim import SimulatedUART

# Alert thresholds (seconds of *continuous* time in that activity before we alert)
DEVICE_ALERT_THRESHOLD = 20.0
SLEEP_ALERT_THRESHOLD = 15.0
CLASSROOM_LOW_THRESHOLD = 33.0

# Schematic bounding boxes (fixed layout: left / center / right of frame)
BASE_BBOXES = {
    1: [80, 160, 260, 420],
    2: [420, 160, 600, 420],
    3: [760, 160, 940, 420],
}

TRACK_ORDER = [1, 2, 3]


def _fmt_clock(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class SimulationEngine:
    def __init__(self):
        self.scorer = EngagementScorer(smoothing_alpha=0.35)
        self.uart = SimulatedUART()

        self.running: bool = False
        self.speed: float = 1.0
        self.scenario: str = DEFAULT_SCENARIO
        self.simulation_time: float = 0.0
        self._last_wall_time: Optional[float] = None

        self.students: Dict[int, StudentState] = {}
        self.recent_events: List[Event] = []
        self.classroom_history: List[ClassroomHistoryPoint] = []
        self.classroom_score: float = 100.0
        self.classroom_level: str = EngagementLevel.HIGH.value

        # tracks continuous-duration alert state so we fire once, not every tick
        self._device_alert_fired: Dict[int, bool] = {}
        self._sleep_alert_fired: Dict[int, bool] = {}
        self._classroom_low_alert_fired: bool = False

        self._session_start_wall = time.time()
        self._max_events = 500
        self._max_history_points = 3600  # ~1 hr at 1s resolution

        self._init_students()

    # ------------------------------------------------------------------
    # Setup / reset
    # ------------------------------------------------------------------

    def _init_students(self):
        names = {1: "Student 01", 2: "Student 02", 3: "Student 03"}
        self.students = {}
        for track_id in TRACK_ORDER:
            activity = activity_at(self.scenario, track_id, 0.0)
            weight = self.scorer.raw_weight(activity)
            st = StudentState(
                student_id=f"student_{track_id:02d}",
                track_id=track_id,
                display_name=names[track_id],
                current_activity=activity,
                activity_class_id=ACTIVITY_CLASS_IDS[activity],
                confidence=0.9,
                bbox=list(BASE_BBOXES[track_id]),
                raw_score=weight,
                score=weight,
                level=self.scorer.level(weight),
                status="Active",
                activity_since=0.0,
            )
            st.activity_timeline.append(TimelineSegment(activity=activity, start=0.0, end=None))
            st.score_history.append(ScorePoint(t=0.0, score=weight))
            self.students[track_id] = st

        self._device_alert_fired = {t: False for t in TRACK_ORDER}
        self._sleep_alert_fired = {t: False for t in TRACK_ORDER}
        self._classroom_low_alert_fired = False
        self.recent_events = []
        self.classroom_history = []
        self._recompute_classroom(0.0)
        self._log_system_event("Simulation initialized.", sim_time=0.0)

    def reset(self):
        self.running = False
        self.simulation_time = 0.0
        self._last_wall_time = None
        self._session_start_wall = time.time()
        self._init_students()

    def set_scenario(self, key: str):
        if key not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{key}'")
        self.scenario = key
        self.reset()

    def set_speed(self, speed: float):
        allowed = (0.5, 1.0, 2.0, 5.0, 10.0)
        if speed not in allowed:
            # snap to nearest allowed value rather than hard failing
            speed = min(allowed, key=lambda x: abs(x - speed))
        self.speed = speed

    def play(self):
        self.running = True
        self._last_wall_time = time.time()

    def pause(self):
        self.running = False
        self._last_wall_time = None

    def jump(self, seconds: float):
        self._advance(seconds)

    # ------------------------------------------------------------------
    # Tick / advance
    # ------------------------------------------------------------------

    def tick_realtime(self):
        """Called ~10x/sec by the background loop. Converts wall-clock
        elapsed time into simulated seconds using the current speed."""
        if not self.running:
            return
        now = time.time()
        if self._last_wall_time is None:
            self._last_wall_time = now
            return
        elapsed_wall = now - self._last_wall_time
        self._last_wall_time = now
        self._advance(elapsed_wall * self.speed)

    def _advance(self, delta_seconds: float):
        if delta_seconds <= 0:
            return
        # Sub-step in <=1s increments so durations/events/alerts stay accurate
        # even at 10x speed.
        remaining = delta_seconds
        step = 1.0
        while remaining > 0:
            dt = min(step, remaining)
            self._advance_step(dt)
            remaining -= dt

    def _advance_step(self, dt: float):
        prev_time = self.simulation_time
        self.simulation_time += dt

        for track_id in TRACK_ORDER:
            st = self.students[track_id]
            target_activity = activity_at(self.scenario, track_id, self.simulation_time)

            if target_activity != st.current_activity:
                self._transition(st, target_activity)

            # accumulate duration for whichever activity was active during this dt
            self._accumulate_duration(st, st.current_activity, dt)

            # smooth score toward current activity's weight
            st.raw_score = self.scorer.raw_weight(st.current_activity)
            st.score = self.scorer.smooth(st.score, st.current_activity)
            st.level = self.scorer.level(st.score)

            self._maybe_alert_duration(st)

        # record score history + timeline close-out at ~1s resolution
        for track_id in TRACK_ORDER:
            st = self.students[track_id]
            st.score_history.append(ScorePoint(t=round(self.simulation_time, 1), score=st.score))
            if len(st.score_history) > self._max_history_points:
                st.score_history.pop(0)

        self._recompute_classroom(self.simulation_time)
        self.classroom_history.append(
            ClassroomHistoryPoint(t=round(self.simulation_time, 1),
                                   score=self.classroom_score,
                                   level=self.classroom_level)
        )
        if len(self.classroom_history) > self._max_history_points:
            self.classroom_history.pop(0)

        self._maybe_classroom_alert()

        # publish simulated UART packet on the same cadence the real system would
        snapshot = self.build_snapshot()
        self.uart.send(snapshot.uart_packet(), self.simulation_time)

    def _transition(self, st: StudentState, new_activity: str):
        old_activity = st.current_activity
        # close out previous timeline segment
        if st.activity_timeline:
            st.activity_timeline[-1].end = round(self.simulation_time, 1)
        st.activity_timeline.append(
            TimelineSegment(activity=new_activity, start=round(self.simulation_time, 1), end=None)
        )
        st.current_activity = new_activity
        st.activity_class_id = ACTIVITY_CLASS_IDS[new_activity]
        st.activity_since = self.simulation_time
        st.confidence = 0.85 + (hash((st.track_id, new_activity, int(self.simulation_time))) % 10) / 100.0

        if new_activity == ActivityClass.HANDRISE:
            st.handrise_count += 1

        # reset continuous-duration alert flags on transition away from that activity
        if old_activity == ActivityClass.USING_DEVICE:
            self._device_alert_fired[st.track_id] = False
        if old_activity == ActivityClass.SLEEP:
            self._sleep_alert_fired[st.track_id] = False

        self._log_event(
            student=st, kind="transition",
            from_activity=old_activity, to_activity=new_activity,
            message=f"{st.display_name}: {old_activity} \u2192 {new_activity}",
            severity="info",
        )

    def _accumulate_duration(self, st: StudentState, activity: str, dt: float):
        field_map = {
            ActivityClass.LOOK_FORWARD: "look_forward_duration",
            ActivityClass.READ: "read_duration",
            ActivityClass.WRITE: "write_duration",
            ActivityClass.SLEEP: "sleep_duration",
            ActivityClass.STAND: "stand_duration",
            ActivityClass.TURN_HEAD: "turn_head_duration",
            ActivityClass.USING_DEVICE: "using_device_duration",
            ActivityClass.HANDRISE: "handrise_duration",
        }
        attr = field_map.get(activity)
        if attr:
            setattr(st, attr, getattr(st, attr) + dt)

    def _maybe_alert_duration(self, st: StudentState):
        continuous = self.simulation_time - st.activity_since
        if st.current_activity == ActivityClass.USING_DEVICE:
            if continuous >= DEVICE_ALERT_THRESHOLD and not self._device_alert_fired[st.track_id]:
                self._device_alert_fired[st.track_id] = True
                self._log_event(
                    student=st, kind="alert",
                    message=f"{st.display_name} has been using a device for "
                            f"{int(continuous)} seconds.",
                    severity="warning",
                )
        if st.current_activity == ActivityClass.SLEEP:
            if continuous >= SLEEP_ALERT_THRESHOLD and not self._sleep_alert_fired[st.track_id]:
                self._sleep_alert_fired[st.track_id] = True
                self._log_event(
                    student=st, kind="alert",
                    message=f"{st.display_name} has been in sleep state for "
                            f"{int(continuous)} seconds.",
                    severity="warning",
                )

    def _maybe_classroom_alert(self):
        if self.classroom_score < CLASSROOM_LOW_THRESHOLD and not self._classroom_low_alert_fired:
            self._classroom_low_alert_fired = True
            self._log_event(
                student=None, kind="alert",
                message=f"Classroom engagement dropped below 33 "
                        f"(currently {round(self.classroom_score)}).",
                severity="critical",
            )
        elif self.classroom_score >= CLASSROOM_LOW_THRESHOLD:
            self._classroom_low_alert_fired = False

    def _recompute_classroom(self, sim_time: float):
        scores = [s.score for s in self.students.values()]
        self.classroom_score = self.scorer.classroom_aggregate(scores)
        self.classroom_level = self.scorer.level(self.classroom_score)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _log_event(self, student: Optional[StudentState], kind: str, message: str,
                    from_activity: Optional[str] = None, to_activity: Optional[str] = None,
                    severity: str = "info"):
        ev = Event(
            timestamp=round(self.simulation_time, 1),
            wall_clock=_fmt_clock(self.simulation_time),
            student_id=student.student_id if student else None,
            track_id=student.track_id if student else None,
            display_name=student.display_name if student else None,
            kind=kind,
            from_activity=from_activity,
            to_activity=to_activity,
            message=message,
            severity=severity,
        )
        self.recent_events.append(ev)
        if len(self.recent_events) > self._max_events:
            self.recent_events.pop(0)

    def _log_system_event(self, message: str, sim_time: float):
        self._log_event(student=None, kind="system", message=message, severity="info")

    # ------------------------------------------------------------------
    # Snapshot / API surface
    # ------------------------------------------------------------------

    def build_snapshot(self) -> ScoreSnapshot:
        students = list(self.students.values())
        focused = sum(1 for s in students if s.current_activity in
                       (ActivityClass.LOOK_FORWARD, ActivityClass.READ,
                        ActivityClass.WRITE, ActivityClass.HANDRISE))
        device = sum(1 for s in students if s.current_activity == ActivityClass.USING_DEVICE)
        sleep = sum(1 for s in students if s.current_activity == ActivityClass.SLEEP)
        stand = sum(1 for s in students if s.current_activity == ActivityClass.STAND)
        turn_head = sum(1 for s in students if s.current_activity == ActivityClass.TURN_HEAD)
        return ScoreSnapshot(
            simulation_time=round(self.simulation_time, 1),
            classroom_score=round(self.classroom_score, 1),
            classroom_level=self.classroom_level,
            students=students,
            focused_count=focused,
            device_count=device,
            sleep_count=sleep,
            stand_count=stand,
            turn_head_count=turn_head,
        )

    def state_dict(self) -> dict:
        snap = self.build_snapshot()
        return {
            "mode": "SIMULATION",
            "scenario": self.scenario,
            "scenario_label": SCENARIO_LABELS.get(self.scenario, self.scenario),
            "available_scenarios": [
                {"key": k, "label": v} for k, v in SCENARIO_LABELS.items()
            ],
            "running": self.running,
            "speed": self.speed,
            "simulation_time": snap.simulation_time,
            "simulation_clock": _fmt_clock(self.simulation_time),
            "session_wall_seconds": round(time.time() - self._session_start_wall, 1),
            "classroom_score": snap.classroom_score,
            "classroom_level": snap.classroom_level,
            "student_count": len(snap.students),
            "focused_count": snap.focused_count,
            "device_count": snap.device_count,
            "sleep_count": snap.sleep_count,
            "stand_count": snap.stand_count,
            "turn_head_count": snap.turn_head_count,
            "students": [s.to_dict() for s in snap.students],
            "recent_events": [e.to_dict() for e in self.recent_events[-50:]],
            "classroom_history": [asdict(p) for p in self.classroom_history[-600:]],
            "uart_last_packet": self.uart.last_packet,
            "uart_log": self.uart.recent(20),
        }

    def students_dict(self) -> dict:
        return {"students": [s.to_dict() for s in self.students.values()]}

    def events_dict(self, limit: int = 100) -> dict:
        return {"events": [e.to_dict() for e in self.recent_events[-limit:]]}

    def history_dict(self) -> dict:
        return {
            "classroom_history": [asdict(p) for p in self.classroom_history],
            "students": {
                str(tid): {
                    "score_history": [asdict(p) for p in s.score_history],
                    "activity_timeline": [
                        {"activity": seg.activity, "start": seg.start, "end": seg.end}
                        for seg in s.activity_timeline
                    ],
                }
                for tid, s in self.students.items()
            },
        }

    # ------------------------------------------------------------------
    # Session summary + export
    # ------------------------------------------------------------------

    def session_summary(self) -> dict:
        students = list(self.students.values())
        if self.classroom_history:
            avg_score = sum(p.score for p in self.classroom_history) / len(self.classroom_history)
        else:
            avg_score = self.classroom_score

        highest = max(students, key=lambda s: s.score) if students else None
        lowest = min(students, key=lambda s: s.score) if students else None

        total_handraises = sum(s.handrise_count for s in students)
        total_device_events = sum(
            1 for e in self.recent_events
            if e.kind == "transition" and e.to_activity == ActivityClass.USING_DEVICE
        )
        total_sleep_events = sum(
            1 for e in self.recent_events
            if e.kind == "transition" and e.to_activity == ActivityClass.SLEEP
        )
        total_turn_head_events = sum(
            1 for e in self.recent_events
            if e.kind == "transition" and e.to_activity == ActivityClass.TURN_HEAD
        )
        total_standing_events = sum(
            1 for e in self.recent_events
            if e.kind == "transition" and e.to_activity == ActivityClass.STAND
        )

        return {
            "session_duration": round(self.simulation_time, 1),
            "session_clock": _fmt_clock(self.simulation_time),
            "average_classroom_score": round(avg_score, 1),
            "final_classroom_score": round(self.classroom_score, 1),
            "final_classroom_level": self.classroom_level,
            "highest_engagement_student": highest.display_name if highest else None,
            "highest_engagement_score": round(highest.score, 1) if highest else None,
            "lowest_engagement_student": lowest.display_name if lowest else None,
            "lowest_engagement_score": round(lowest.score, 1) if lowest else None,
            "total_handraises": total_handraises,
            "total_device_events": total_device_events,
            "total_sleep_events": total_sleep_events,
            "total_turn_head_events": total_turn_head_events,
            "total_standing_events": total_standing_events,
            "total_reading_time": round(sum(s.read_duration for s in students), 1),
            "total_writing_time": round(sum(s.write_duration for s in students), 1),
            "total_look_forward_time": round(sum(s.look_forward_duration for s in students), 1),
            "per_student_average_score": {
                s.display_name: round(
                    sum(p.score for p in s.score_history) / len(s.score_history), 1
                ) if s.score_history else round(s.score, 1)
                for s in students
            },
            "engagement_distribution": {
                "HIGH": sum(1 for s in students if s.level == EngagementLevel.HIGH.value),
                "MEDIUM": sum(1 for s in students if s.level == EngagementLevel.MEDIUM.value),
                "LOW": sum(1 for s in students if s.level == EngagementLevel.LOW.value),
            },
            "alerts": [e.to_dict() for e in self.recent_events if e.kind == "alert"],
        }

    def export_session(self) -> dict:
        return {
            "session_metadata": {
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "scenario": self.scenario,
                "scenario_label": SCENARIO_LABELS.get(self.scenario, self.scenario),
                "simulation_time": round(self.simulation_time, 1),
                "speed": self.speed,
            },
            "student_states": [s.to_dict(history_limit=100000) for s in self.students.values()],
            "events": [e.to_dict() for e in self.recent_events],
            "classroom_history": [asdict(p) for p in self.classroom_history],
            "session_summary": self.session_summary(),
        }

    def export_csv_rows(self) -> List[dict]:
        """student_id, timestamp, activity, score, level -- one row per
        student per recorded score-history point."""
        rows = []
        for s in self.students.values():
            for point in s.score_history:
                rows.append({
                    "student_id": s.student_id,
                    "timestamp": point.t,
                    "activity": s.current_activity,  # best-effort; see README note
                    "score": round(point.score, 1),
                    "level": self.scorer.level(point.score),
                })
        return rows
