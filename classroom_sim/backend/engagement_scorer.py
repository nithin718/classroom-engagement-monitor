"""
engagement_scorer.py

The single source of truth for turning an "activity" (from YOLO in LIVE
mode, or from SimulationEngine in SIMULATION mode) into an engagement
score and level. Both modes call the SAME EngagementScorer instance per
student -- this is the "convergence point" described in the architecture
(the two input sources should meet as early as practical so the rest of
the pipeline is shared).

Includes temporal smoothing (exponential moving average) so scores do not
jump instantly between activities -- matching the requirement that
simulation should resemble the real system's aggregation rather than
snapping between raw numbers every tick.
"""

from __future__ import annotations

from models import ACTIVITY_WEIGHTS, EngagementLevel, level_for_score


class EngagementScorer:
    """
    Stateless with respect to which student it's scoring -- callers pass in
    the student's previous smoothed score and get back the new one. This
    mirrors a scorer that could be shared across many tracked students.
    """

    def __init__(self, smoothing_alpha: float = 0.35):
        # smoothing_alpha: weight given to the NEW raw score each update.
        # Lower = smoother/slower to react, higher = snappier.
        self.smoothing_alpha = smoothing_alpha

    def raw_weight(self, activity: str) -> float:
        return float(ACTIVITY_WEIGHTS.get(activity, 50))

    def smooth(self, previous_score: float, activity: str) -> float:
        raw = self.raw_weight(activity)
        alpha = self.smoothing_alpha
        return (alpha * raw) + ((1 - alpha) * previous_score)

    def level(self, score: float) -> str:
        return level_for_score(score).value

    def classroom_aggregate(self, student_scores):
        """Classroom score = arithmetic mean of current student scores.
        Never a random or hard-coded number -- always derived."""
        scores = list(student_scores)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
