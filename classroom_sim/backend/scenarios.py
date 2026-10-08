"""
scenarios.py

Deterministic scenario definitions. Each scenario maps a track_id (1, 2, 3)
to an ordered list of (activity, duration_seconds) segments. Segments are
expanded into absolute-time event lists at import time so playback is
fully repeatable and debuggable (never randomized by default).

Each scenario loops once its longest student timeline is exhausted, so a
demo can be left running indefinitely.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Segment = Tuple[str, float]          # (activity, duration_seconds)
TimedEvent = Dict[str, float]          # {"time": t, "activity": a}


def _expand(segments: List[Segment]) -> Tuple[List[TimedEvent], float]:
    events: List[TimedEvent] = []
    t = 0.0
    for activity, duration in segments:
        events.append({"time": t, "activity": activity})
        t += duration
    return events, t


def _build(profile: Dict[int, List[Segment]]) -> Dict[str, object]:
    tracks: Dict[int, List[TimedEvent]] = {}
    max_len = 0.0
    for track_id, segments in profile.items():
        events, length = _expand(segments)
        tracks[track_id] = events
        max_len = max(max_len, length)
    return {"tracks": tracks, "loop_length": max_len}


# ---------------------------------------------------------------------------
# Scenario 0 (default): "Demo Classroom" -- matches the exact demo script
# in the spec (section 39) plus the three baseline profiles (section 8).
# Designed to be self-explanatory within ~90 seconds.
# ---------------------------------------------------------------------------

DEMO_CLASSROOM = _build({
    1: [  # Student 01 -- high engagement, stable
        ("look_forward", 12), ("read", 10), ("write", 6), ("handrise", 3),
        ("look_forward", 9), ("read", 8), ("turn_head", 4), ("look_forward", 8),
        ("write", 6), ("read", 8), ("look_forward", 6),
    ],
    2: [  # Student 02 -- mixed, dips into device use mid-session then recovers
        ("look_forward", 10), ("read", 8), ("write", 7), ("turn_head", 6),
        ("using_device", 10), ("using_device", 8),   # distraction period
        ("look_forward", 8), ("using_device", 6), ("read", 6),
        ("look_forward", 11),
    ],
    3: [  # Student 03 -- low engagement, dips into sleep, then recovers at the end
        ("look_forward", 8), ("turn_head", 7), ("using_device", 8),
        ("sleep", 16),                                  # triggers "sleep" alert
        ("stand", 6), ("using_device", 7),
        ("look_forward", 8), ("look_forward", 10),      # recovery
    ],
})


NORMAL_CLASSROOM = _build({
    1: [("look_forward", 14), ("read", 12), ("write", 8), ("look_forward", 10),
        ("read", 10), ("turn_head", 5), ("look_forward", 11)],
    2: [("look_forward", 12), ("read", 10), ("write", 10), ("turn_head", 6),
        ("look_forward", 10), ("read", 12)],
    3: [("look_forward", 10), ("read", 9), ("turn_head", 8), ("look_forward", 9),
        ("write", 8), ("look_forward", 16)],
})

MIXED_ENGAGEMENT = _build({
    1: [("look_forward", 10), ("read", 10), ("write", 8), ("handrise", 3),
        ("look_forward", 12), ("read", 9), ("turn_head", 4), ("look_forward", 8)],
    2: [("look_forward", 10), ("read", 8), ("write", 7), ("turn_head", 8),
        ("using_device", 9), ("look_forward", 7), ("using_device", 6), ("read", 9)],
    3: [("look_forward", 8), ("turn_head", 9), ("using_device", 13),
        ("sleep", 12), ("stand", 6), ("using_device", 8), ("look_forward", 8)],
})

DISTRACTION_EVENT = _build({
    1: [("read", 20), ("read", 20), ("look_forward", 15), ("read", 15)],
    2: [("look_forward", 10), ("using_device", 25), ("using_device", 15),
        ("look_forward", 20)],
    3: [("turn_head", 8), ("sleep", 30), ("sleep", 12), ("look_forward", 20)],
})

TEACHER_QUESTION = _build({
    1: [("look_forward", 8), ("handrise", 6), ("look_forward", 10),
        ("write", 10), ("look_forward", 10)],
    2: [("look_forward", 8), ("turn_head", 5), ("handrise", 4),
        ("look_forward", 12), ("read", 10), ("look_forward", 8)],
    3: [("turn_head", 10), ("using_device", 6), ("look_forward", 4),
        ("handrise", 3), ("look_forward", 10), ("turn_head", 8)],
})

LOW_ENGAGEMENT_PERIOD = _build({
    1: [("read", 15), ("turn_head", 10), ("using_device", 10), ("look_forward", 10)],
    2: [("using_device", 18), ("sleep", 12), ("turn_head", 10), ("using_device", 10)],
    3: [("sleep", 20), ("sleep", 18), ("stand", 8), ("using_device", 12)],
})

DEVICE_DISTRACTION_PERIOD = _build({
    1: [("read", 12), ("using_device", 10), ("read", 12), ("look_forward", 10)],
    2: [("using_device", 15), ("using_device", 15), ("look_forward", 8), ("using_device", 12)],
    3: [("using_device", 15), ("using_device", 15), ("using_device", 10), ("turn_head", 8)],
})

RECOVERY_PERIOD = _build({
    1: [("turn_head", 8), ("look_forward", 12), ("read", 14), ("write", 10)],
    2: [("using_device", 10), ("turn_head", 6), ("look_forward", 12), ("read", 14)],
    3: [("sleep", 12), ("stand", 6), ("turn_head", 8), ("look_forward", 16)],
})


SCENARIOS: Dict[str, Dict[str, object]] = {
    "demo_classroom": DEMO_CLASSROOM,
    "normal_classroom": NORMAL_CLASSROOM,
    "mixed_engagement": MIXED_ENGAGEMENT,
    "distraction_event": DISTRACTION_EVENT,
    "teacher_question": TEACHER_QUESTION,
    "low_engagement_period": LOW_ENGAGEMENT_PERIOD,
    "device_distraction_period": DEVICE_DISTRACTION_PERIOD,
    "recovery_period": RECOVERY_PERIOD,
}

SCENARIO_LABELS: Dict[str, str] = {
    "demo_classroom": "Demo Classroom",
    "normal_classroom": "1. Normal Classroom",
    "mixed_engagement": "2. Mixed Engagement",
    "distraction_event": "3. Distraction Event",
    "teacher_question": "4. Teacher Question / Hand-Raising",
    "low_engagement_period": "5. Low-Engagement Period",
    "device_distraction_period": "6. Device-Distraction Period",
    "recovery_period": "7. Recovery / Re-Engagement",
}

DEFAULT_SCENARIO = "demo_classroom"


def activity_at(scenario_key: str, track_id: int, sim_time: float) -> str:
    """Return the activity a given track should be performing at sim_time,
    looping the scenario once it finishes."""
    scenario = SCENARIOS[scenario_key]
    events: List[TimedEvent] = scenario["tracks"][track_id]
    loop_length = scenario["loop_length"] or 1.0
    t = sim_time % loop_length

    current = events[0]["activity"]
    for ev in events:
        if ev["time"] <= t:
            current = ev["activity"]
        else:
            break
    return current
