"""TUG classification, flag detection, and plain-language interpretation.

Pure functions only — no I/O, no MediaPipe, no MongoDB. The engine
calls these after metrics are computed to build the final TUGResult.

Cutoffs are EXACTLY as specified in the MotionLens spec for the TUG
test — do not introduce additional thresholds here without a spec
update.
"""
from __future__ import annotations

from typing import Optional

from backend.engines.tug_models import TUGClassification, TUGFlag


# ─── Total-time classification ───────────────────────────────────
# Cutoffs (Podsiadlo & Richardson 1991 + Bohannon 2006 meta-analysis):
#   < 10.0 s  → Normal mobility
#   10.0-13.5 s → Mild fall risk
#   13.5-20.0 s → Elevated fall risk
#   > 20.0 s  → Significant fall risk / impaired mobility
def classify_total_time(seconds: float) -> TUGClassification:
    if seconds < 10.0:
        return "normal"
    if seconds < 13.5:
        return "mild_fall_risk"
    if seconds <= 20.0:
        return "elevated_fall_risk"
    return "significant_fall_risk"


CLASSIFICATION_LABEL: dict[TUGClassification, str] = {
    "normal":                "Normal mobility",
    "mild_fall_risk":        "Mild fall risk",
    "elevated_fall_risk":    "Elevated fall risk",
    "significant_fall_risk": "Significant fall risk / impaired mobility",
}


# ─── Independent sub-phase flags ────────────────────────────────
TURN_TIME_FLAG_SEC = 4.0      # turn longer than this = balance impairment
TURN_STEPS_FLAG_COUNT = 5      # more than this many shuffle steps = instability
TEST_TOO_FAST_THRESHOLD_SEC = 5.0  # below this is likely an artifact

def compute_flags(
    turn_duration_sec: float,
    turn_step_count: Optional[int],
    total_time_sec: float,
    phase_truncated_any: bool,
    turn_undetected: bool,
    no_strikes_anywhere: bool,
) -> list[TUGFlag]:
    flags: list[TUGFlag] = []

    if turn_duration_sec > TURN_TIME_FLAG_SEC:
        flags.append(TUGFlag(
            code="turn_time_excessive",
            severity="concern",
            message=(
                f"Balance impairment flag — turn took {turn_duration_sec:.1f} s "
                f"(threshold: {TURN_TIME_FLAG_SEC:.1f} s)."
            ),
        ))

    if turn_step_count is not None and turn_step_count > TURN_STEPS_FLAG_COUNT:
        flags.append(TUGFlag(
            code="turn_steps_excessive",
            severity="concern",
            message=(
                f"Turning instability flag — patient took {turn_step_count} "
                f"shuffle steps during turn (threshold: {TURN_STEPS_FLAG_COUNT})."
            ),
        ))

    if phase_truncated_any:
        flags.append(TUGFlag(
            code="phase_truncated",
            severity="warning",
            message=(
                "One or more phases could not be cleanly bounded — patient "
                "may have walked out of frame or the video ended before "
                "the test was complete. Metrics may be partial."
            ),
        ))

    if turn_undetected:
        flags.append(TUGFlag(
            code="turn_undetected",
            severity="warning",
            message=(
                "Turn phase could not be auto-detected from shoulder rotation. "
                "Walk-out and walk-back durations were split evenly across "
                "the remaining frames — interpret cautiously."
            ),
        ))

    if no_strikes_anywhere:
        flags.append(TUGFlag(
            code="no_strikes_detected",
            severity="warning",
            message=(
                "No foot strikes were detected — step counts, step lengths "
                "and cadence are not reported."
            ),
        ))

    if total_time_sec < TEST_TOO_FAST_THRESHOLD_SEC:
        flags.append(TUGFlag(
            code="test_too_fast",
            severity="info",
            message=(
                f"Total time of {total_time_sec:.1f} s is unusually fast — "
                "review video to confirm the patient performed the full "
                "test (stand, walk 3 m, turn, walk back, sit)."
            ),
        ))

    return flags


# ─── Age-matched norms ──────────────────────────────────────────
# General published norms — patient's TUG time should sit below
# the age-matched ceiling for a "pass" against the population norm.
AGE_NORMS: list[tuple[int, int, float, str]] = [
    # (age_min, age_max, threshold_sec, label)
    (0,   59,  10.0, "under 60"),
    (60,  69,  12.0, "60-69"),
    (70,  79,  13.0, "70-79"),
    (80,  200, 14.0, "80+"),
]


def age_matched_norm(age: Optional[int]) -> Optional[tuple[float, str]]:
    if age is None:
        return None
    for age_min, age_max, threshold, label in AGE_NORMS:
        if age_min <= age <= age_max:
            return threshold, label
    return None


# ─── Plain-language interpretation paragraph ────────────────────
def build_interpretation(
    total_time_sec: float,
    classification: TUGClassification,
    turn_duration_sec: float,
    turn_step_count: Optional[int],
    walk_out_speed_mps: Optional[float],
    walk_back_speed_mps: Optional[float],
    flags: list[TUGFlag],
    patient_age: Optional[int],
) -> str:
    lines: list[str] = []

    # Headline sentence
    label = CLASSIFICATION_LABEL[classification].lower()
    lines.append(
        f"Patient completed the TUG test in {total_time_sec:.1f} seconds — "
        f"{label}."
    )

    # Walking speed summary
    speeds = [s for s in (walk_out_speed_mps, walk_back_speed_mps) if s is not None]
    if speeds:
        mean_speed = sum(speeds) / len(speeds)
        if mean_speed >= 1.0:
            lines.append(
                f"Mean walking speed across the two passes was "
                f"{mean_speed:.2f} m/s, within community-ambulator range."
            )
        elif mean_speed >= 0.6:
            lines.append(
                f"Mean walking speed was {mean_speed:.2f} m/s, suggesting "
                "limited community ambulation — clinical correlation advised."
            )
        else:
            lines.append(
                f"Mean walking speed was {mean_speed:.2f} m/s, below the "
                "household-ambulator threshold (~0.6 m/s)."
            )

    # Turn-specific commentary (only when concerning)
    turn_flag = next((f for f in flags if f.code == "turn_time_excessive"), None)
    turn_steps_flag = next((f for f in flags if f.code == "turn_steps_excessive"), None)
    if turn_flag or turn_steps_flag:
        bits: list[str] = []
        if turn_flag:
            bits.append(f"the {turn_duration_sec:.1f}-second turn duration")
        if turn_steps_flag and turn_step_count is not None:
            bits.append(f"{turn_step_count} shuffle steps during the turn")
        lines.append(
            "Turn analysis flagged " + " and ".join(bits) +
            " — these are independent indicators of balance impairment regardless of total time."
        )

    # Age-matched norm comparison
    norm = age_matched_norm(patient_age)
    if norm is not None:
        threshold, label = norm
        if total_time_sec <= threshold:
            lines.append(
                f"Result is within the age-matched norm "
                f"(age {label}: ≤ {threshold:.1f} s)."
            )
        else:
            lines.append(
                f"Result exceeds the age-matched norm "
                f"(age {label}: ≤ {threshold:.1f} s) by "
                f"{total_time_sec - threshold:.1f} s."
            )
    else:
        lines.append(
            "Age-matched norm comparison was not run — patient age not "
            "available on the profile."
        )

    # Closing recommendation
    if classification != "normal":
        lines.append("Clinical correlation recommended.")

    return " ".join(lines)
