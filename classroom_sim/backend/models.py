"""
models.py
Core data models shared by the simulation engine, the engagement scorer,
and the FastAPI layer. These mirror the semantics of the real pipeline:

    Webcam -> OpenCV -> YOLO11s -> ActivityDetector -> ByteTrack
           -> EngagementScorer -> ScoreSnapshot -> FastAPI -> Teacher Portal -> UART

In simulation mode, SimulationEngine plays the role of
(OpenCV + YOLO11s + ActivityDetector + ByteTrack) and produces the same
downstream objects (StudentState / ScoreSnapshot / Event) that LIVE mode
would produce, so the scoring, API, and UART layers do not need to know or
care where the data came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Fixed behavior taxonomy. DO NOT change these IDs or names -- they must
# match the trained model's class order exactly.
# ---------------------------------------------------------------------------

class ActivityClass(str, Enum):
    HANDRISE = "handrise"
    LOOK_FORWARD = "look_forward"
    READ = "read"
    SLEEP = "sleep"
    STAND = "stand"
    TURN_HEAD = "turn_head"
    USING_DEVICE = "using_device"
    WRITE = "write"


ACTIVITY_CLASS_IDS: Dict[str, int] = {
    ActivityClass.HANDRISE: 0,
    ActivityClass.LOOK_FORWARD: 1,
    ActivityClass.READ: 2,
    ActivityClass.SLEEP: 3,
    ActivityClass.STAND: 4,
    ActivityClass.TURN_HEAD: 5,
    ActivityClass.USING_DEVICE: 6,
    ActivityClass.WRITE: 7,
}

# Engagement weight per activity (0-100). Used by EngagementScorer.
ACTIVITY_WEIGHTS: Dict[str, int] = {
    ActivityClass.HANDRISE: 100,
    ActivityClass.LOOK_FORWARD: 100,
    ActivityClass.READ: 85,
    ActivityClass.WRITE: 80,
    ActivityClass.STAND: 60,
    ActivityClass.TURN_HEAD: 40,
    ActivityClass.USING_DEVICE: 20,
    ActivityClass.SLEEP: 5,
}


class EngagementLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


def level_for_score(score: float) -> EngagementLevel:
    if score >= 66:
        return EngagementLevel.HIGH
    if score >= 33:
        return EngagementLevel.MEDIUM
    return EngagementLevel.LOW


# Semantic colors for the portal. HIGH is intentionally neutral/white,
# never red. Red is reserved for warnings/alerts only.
LEVEL_COLORS: Dict[str, str] = {
    EngagementLevel.HIGH: "#f4f5f7",   # white / neutral
    EngagementLevel.MEDIUM: "#f5c518",  # yellow
    EngagementLevel.LOW: "#3b82f6",     # blue
}


# ---------------------------------------------------------------------------
# Synthetic detection representation (what SimulationEngine emits per tick,
# standing in for a YOLO11s + ByteTrack detection in LIVE mode).
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in a normalized 0-1000 x 0-600 schematic frame

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-student state
# ---------------------------------------------------------------------------

@dataclass
class ScorePoint:
    t: float          # simulation_time (seconds)
    score: float


@dataclass
class TimelineSegment:
    activity: str
    start: float
    end: Optional[float]  # None while still ongoing


@dataclass
class StudentState:
    student_id: str          # "student_01"
    track_id: int             # 1, 2, 3 (persistent -- ByteTrack concept)
    display_name: str         # "Student 01"

    current_activity: str = ActivityClass.LOOK_FORWARD
    activity_class_id: int = ACTIVITY_CLASS_IDS[ActivityClass.LOOK_FORWARD]
    confidence: float = 0.9
    bbox: List[float] = field(default_factory=lambda: [0, 0, 0, 0])

    raw_score: float = 100.0       # instantaneous weight of current activity
    score: float = 100.0           # smoothed engagement score (EngagementScorer output)
    level: str = EngagementLevel.HIGH
    status: str = "Active"         # Active / Idle (presence indicator)

    activity_since: float = 0.0    # simulation_time the current activity started

    # Duration counters (seconds), section 25
    look_forward_duration: float = 0.0
    read_duration: float = 0.0
    write_duration: float = 0.0
    sleep_duration: float = 0.0
    stand_duration: float = 0.0
    turn_head_duration: float = 0.0
    using_device_duration: float = 0.0
    handrise_duration: float = 0.0
    handrise_count: int = 0

    activity_timeline: List[TimelineSegment] = field(default_factory=list)
    score_history: List[ScorePoint] = field(default_factory=list)

    def focused_time(self) -> float:
        """Focused = look_forward + read + write + handrise."""
        return (
            self.look_forward_duration
            + self.read_duration
            + self.write_duration
            + self.handrise_duration
        )

    def to_dict(self, history_limit: int = 240) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "track_id": self.track_id,
            "display_name": self.display_name,
            "current_activity": self.current_activity,
            "activity_class_id": self.activity_class_id,
            "confidence": round(self.confidence, 2),
            "bbox": self.bbox,
            "score": round(self.score, 1),
            "raw_score": round(self.raw_score, 1),
            "level": self.level,
            "status": self.status,
            "focused_time": round(self.focused_time(), 1),
            "device_time": round(self.using_device_duration, 1),
            "sleep_time": round(self.sleep_duration, 1),
            "stand_time": round(self.stand_duration, 1),
            "turn_head_time": round(self.turn_head_duration, 1),
            "read_time": round(self.read_duration, 1),
            "write_time": round(self.write_duration, 1),
            "look_forward_time": round(self.look_forward_duration, 1),
            "handrise_count": self.handrise_count,
            "activity_timeline": [
                {"activity": s.activity, "start": s.start, "end": s.end}
                for s in self.activity_timeline[-history_limit:]
            ],
            "score_history": [
                {"t": p.t, "score": round(p.score, 1)}
                for p in self.score_history[-history_limit:]
            ],
        }


# ---------------------------------------------------------------------------
# Events / alerts
# ---------------------------------------------------------------------------

@dataclass
class Event:
    timestamp: float          # simulation_time
    wall_clock: str           # HH:MM:SS for display
    student_id: Optional[str]
    track_id: Optional[int]
    display_name: Optional[str]
    kind: str                  # "transition" | "alert" | "system"
    from_activity: Optional[str] = None
    to_activity: Optional[str] = None
    message: str = ""
    severity: str = "info"     # info | warning | critical

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Classroom-level snapshot (mirrors the real ScoreSnapshot object)
# ---------------------------------------------------------------------------

@dataclass
class ClassroomHistoryPoint:
    t: float
    score: float
    level: str


@dataclass
class ScoreSnapshot:
    """Represents one published tick of classroom state -- the object that
    would normally be handed to the FastAPI dashboard layer and to the
    UART sender in the real system."""
    simulation_time: float
    classroom_score: float
    classroom_level: str
    students: List[StudentState]
    focused_count: int
    device_count: int
    sleep_count: int
    stand_count: int
    turn_head_count: int

    def uart_packet(self) -> str:
        # ENG,<score>,<level>,<focused_count>,<device_count>,<sleep_count>\n
        return (
            f"ENG,{round(self.classroom_score)},{self.classroom_level},"
            f"{self.focused_count},{self.device_count},{self.sleep_count}"
        )
