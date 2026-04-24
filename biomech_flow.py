"""
biomech_flow.py
Biomechanical Analysis flow for GaitVision (shoulder + neck range-of-motion).

Sub-flow tracked by st.session_state["biomech_step"]:
    patient   → body_part → movement → mode → capture → report

This module is reached only when st.session_state["app_mode"] == "biomech".
The render entry-point is `render_biomech_flow()`, called from app.py's
top-level dispatcher.

Status:
  Commit 4 (this commit) — capture-mode chooser + video-upload analyzer.
  Live camera ("Coming soon" tag) and the report screen are still
  placeholder; report lands in commit 6.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from datetime import date

import streamlit as st

from shoulder_engine import (
    SHOULDER_NORMAL_RANGES,
    compute_shoulder_angle,
)
from neck_engine import (
    NECK_NORMAL_RANGES,
    compute_neck_angle,
)


# ──────────────────────────────────────────────
# Static config
# ──────────────────────────────────────────────
GENDER_OPTIONS = ["", "Female", "Male", "Other", "Prefer not to say"]
DEFAULT_HEIGHT_CM = 170

# Movement metadata — title + 1-line layperson description, displayed
# on the movement-selection cards. Order matches the spec.
SHOULDER_MOVEMENTS = [
    ("flexion",
     "Flexion",
     "Raise the arm forward and upward.",
     "180°"),
    ("extension",
     "Extension",
     "Reach the arm backward.",
     "45–60°"),
    ("abduction",
     "Abduction",
     "Raise the arm out to the side.",
     "180°"),
    ("adduction",
     "Adduction",
     "Bring the arm toward the body's midline.",
     "30–50°"),
    ("external_rotation",
     "External Rotation",
     "With elbow at 90°, rotate hand outward.",
     "90°"),
    ("internal_rotation",
     "Internal Rotation",
     "With elbow at 90°, rotate hand toward stomach.",
     "70–90°"),
]

NECK_MOVEMENTS = [
    ("flexion",
     "Flexion / Extension",
     "Chin to chest (flex), then look up (extend).",
     "Flex 45–80° · Ext 50–70°"),
    ("lateral_flexion",
     "Lateral Flexion",
     "Tilt the ear toward the shoulder, each side.",
     "20–45° each side"),
    ("rotation",
     "Rotation",
     "Turn the head to look left and right.",
     "70–90° each side"),
]


# Per-movement recording instructions. Surfaced on the upload-capture
# screen as a numbered checklist. Source: AAOS / APTA standard ROM
# positioning guidance.
MOVEMENT_INSTRUCTIONS = {
    ("shoulder", "flexion"): [
        "Stand 6 feet from the camera, facing sideways (true side view).",
        "Keep the arm being assessed straight, hanging at your side.",
        "Slowly raise the arm forward and upward as high as you can.",
        "Hold at the peak position for 2 seconds.",
        "Slowly lower the arm back to the starting position.",
    ],
    ("shoulder", "extension"): [
        "Stand 6 feet from the camera, facing sideways (true side view).",
        "Keep the arm straight at your side as the starting position.",
        "Slowly reach the arm backward, behind your body, as far as comfortable.",
        "Hold at the peak position for 2 seconds.",
        "Return slowly to the starting position.",
    ],
    ("shoulder", "abduction"): [
        "Stand 6 feet from the camera, facing the camera (front view).",
        "Keep the arm straight at your side, palm facing your body.",
        "Slowly raise the arm out to the side, palm down, as high as you can.",
        "Hold at the peak position for 2 seconds.",
        "Slowly lower back to the starting position.",
    ],
    ("shoulder", "adduction"): [
        "Stand 6 feet from the camera, facing the camera (front view).",
        "Start with the arm raised slightly out to the side.",
        "Slowly bring the arm across the front of the body toward the opposite shoulder.",
        "Hold at the peak position for 2 seconds.",
        "Return slowly to the starting position.",
    ],
    ("shoulder", "external_rotation"): [
        "Stand 6 feet from the camera, facing the camera (front view).",
        "Bend the elbow to 90°, with the upper arm at your side, "
        "forearm pointing forward.",
        "Without moving the elbow, slowly rotate the forearm outward "
        "(away from your body).",
        "Hold at the peak position for 2 seconds.",
        "Return slowly to the starting position.",
    ],
    ("shoulder", "internal_rotation"): [
        "Stand 6 feet from the camera, facing the camera (front view).",
        "Bend the elbow to 90°, with the upper arm at your side, "
        "forearm pointing forward.",
        "Without moving the elbow, slowly rotate the forearm inward "
        "(across your stomach).",
        "Hold at the peak position for 2 seconds.",
        "Return slowly to the starting position.",
    ],

    ("neck", "flexion"): [
        "Sit upright facing sideways to the camera (true side view).",
        "Keep your shoulders relaxed and still.",
        "Slowly bring your chin toward your chest as far as comfortable.",
        "Hold for 2 seconds, then return to neutral upright posture.",
        "Then slowly tilt your head backward (look up at the ceiling) "
        "as far as comfortable, hold 2 seconds, return to neutral.",
    ],
    ("neck", "lateral_flexion"): [
        "Sit upright facing the camera (front view).",
        "Keep your shoulders level and relaxed — do not raise the shoulder "
        "toward the ear.",
        "Slowly tilt your head so your ear moves toward your shoulder "
        "(do not turn the head — keep nose facing forward).",
        "Hold for 2 seconds, return to upright, then repeat to the other side.",
    ],
    ("neck", "rotation"): [
        "Sit upright facing the camera (front view).",
        "Keep your shoulders level and still.",
        "Slowly turn your head to one side as far as comfortable, keeping the chin level.",
        "Hold for 2 seconds, return to centre, then repeat to the other side.",
    ],
}


# ──────────────────────────────────────────────
# Frame-by-frame analyzer (video upload mode)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_pose_estimator():
    """Cached MediaPipe Pose instance for biomech (uses the older
    `mp.solutions.pose.Pose` API — simpler than the task API and self-
    contained, no .task model file needed). Cached across reruns."""
    import mediapipe as mp
    return mp.solutions.pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _run_biomech_upload_analysis(video_path: str,
                                  body_part: str,
                                  movement: str,
                                  side: str) -> dict:
    """Process the uploaded video frame by frame, computing the chosen
    movement's angle each frame and tracking the peak magnitude.

    Returns a dict with keys:
      peak_angle        — signed peak angle (degrees) or None
      peak_magnitude    — abs(peak_angle); 0.0 when no valid frames
      valid_frames      — frames where the angle could be computed
      total_frames      — total frames processed
      fps               — video fps
    """
    import cv2

    pose = _get_pose_estimator()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    peak_magnitude = 0.0
    peak_angle: float | None = None
    valid_frames = 0
    total_frames = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        total_frames += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        if result.pose_landmarks is None:
            continue

        landmarks = result.pose_landmarks.landmark
        if body_part == "shoulder":
            angle = compute_shoulder_angle(landmarks, side.lower(), movement)
        else:
            angle = compute_neck_angle(landmarks, movement)
        if angle is None:
            continue

        valid_frames += 1
        if abs(angle) > peak_magnitude:
            peak_magnitude = abs(angle)
            peak_angle = angle

    cap.release()

    return {
        "peak_angle":     peak_angle,
        "peak_magnitude": peak_magnitude,
        "valid_frames":   valid_frames,
        "total_frames":   total_frames,
        "fps":            float(fps),
    }


def _is_rotation_movement(body_part: str, movement: str | None) -> bool:
    if body_part == "shoulder":
        return movement in ("external_rotation", "internal_rotation")
    if body_part == "neck":
        return movement == "rotation"
    return False


# ══════════════════════════════════════════════════════════════════
# LIVE CAMERA — WebRTC video processor (runs on a worker thread)
# ══════════════════════════════════════════════════════════════════
class BiomechVideoProcessor:
    """streamlit-webrtc VideoProcessor: per-frame Pose extraction +
    angle computation + skeleton overlay drawing.

    CONTINUOUS-CAPTURE MODE — there is no Start/Stop or 10-second
    window. From the moment the camera connects the processor tracks
    the peak angle on every valid frame; the main thread reads via
    `get_state()`. To wipe the running peak (e.g. user wants to redo
    the take), call `reset_peak()`. Configuration of body_part /
    movement / side comes via `configure()`.
    """

    def __init__(self):
        import cv2
        import mediapipe as mp

        self._lock = threading.Lock()
        self._cv2  = cv2
        self._pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._draw_utils       = mp.solutions.drawing_utils
        self._drawing_spec     = mp.solutions.drawing_utils.DrawingSpec
        self._pose_connections = mp.solutions.pose.POSE_CONNECTIONS

        # Configurable from main thread
        self.body_part: str | None = None
        self.movement:  str | None = None
        self.side:      str        = "right"

        # Continuous tracking state
        self.peak_angle:     float | None = None
        self.peak_magnitude: float        = 0.0
        self.current_angle:  float | None = None
        self.posture_status: str          = "idle"
        self.valid_frames    = 0    # frames where an angle was computed
        self.total_frames    = 0    # frames seen since last reset

    # ---- Main-thread API --------------------------------------------
    def configure(self, body_part: str, movement: str, side: str) -> None:
        """Re-configure the active movement. If the body_part / movement
        changed since last call, the running peak is reset (otherwise we
        would mix angles across movements during chained capture)."""
        with self._lock:
            changed = (
                body_part != self.body_part
                or movement != self.movement
                or side.lower() != self.side
            )
            self.body_part = body_part
            self.movement  = movement
            self.side      = side.lower()
            if changed:
                self.peak_angle     = None
                self.peak_magnitude = 0.0
                self.valid_frames   = 0
                self.total_frames   = 0

    def reset_peak(self) -> None:
        """Wipe the running peak so the next take starts fresh."""
        with self._lock:
            self.peak_angle     = None
            self.peak_magnitude = 0.0
            self.valid_frames   = 0
            self.total_frames   = 0

    def get_state(self) -> dict:
        with self._lock:
            return {
                "current_angle":   self.current_angle,
                "peak_angle":      self.peak_angle,
                "peak_magnitude":  self.peak_magnitude,
                "valid_frames":    self.valid_frames,
                "total_frames":    self.total_frames,
                "posture_status":  self.posture_status,
            }

    # ---- Worker-thread frame callback --------------------------------
    def recv(self, frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        with self._lock:
            body_part = self.body_part
            movement  = self.movement
            side      = self.side
            self.total_frames += 1

            if result.pose_landmarks is None:
                self.posture_status = "no_landmarks"
                self.current_angle  = None
            else:
                # Skeleton overlay on the outgoing frame
                self._draw_utils.draw_landmarks(
                    img, result.pose_landmarks, self._pose_connections,
                    landmark_drawing_spec=self._drawing_spec(
                        color=(255, 255, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=self._drawing_spec(
                        color=(180, 180, 180), thickness=1),
                )

                angle = None
                if body_part == "shoulder" and movement:
                    angle = compute_shoulder_angle(
                        result.pose_landmarks.landmark, side, movement)
                elif body_part == "neck" and movement:
                    angle = compute_neck_angle(
                        result.pose_landmarks.landmark, movement)

                if angle is None:
                    self.posture_status = "low_visibility"
                    self.current_angle  = None
                else:
                    self.posture_status = "good"
                    self.current_angle  = angle
                    self.valid_frames  += 1
                    if abs(angle) > self.peak_magnitude:
                        self.peak_magnitude = abs(angle)
                        self.peak_angle     = angle

            # On-frame text overlays — continuous, no recording state
            if self.current_angle is not None:
                txt = f"{self.current_angle:+.1f} deg"
                colour = (0, 255, 255)  # cyan
            else:
                txt = "-- no angle --"
                colour = (100, 100, 255)
            self._cv2.putText(img, txt, (10, 50),
                              self._cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                              colour, 2)
            if self.peak_magnitude > 0:
                self._cv2.putText(img,
                                  f"peak: {self.peak_magnitude:.1f} deg",
                                  (10, 90),
                                  self._cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                  (50, 200, 50), 2)
            self._cv2.putText(img,
                              f"frames: {self.valid_frames}/{self.total_frames}",
                              (10, 125),
                              self._cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                              (200, 200, 200), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ──────────────────────────────────────────────
# Multi-movement chain helpers (used by Live capture)
# ──────────────────────────────────────────────
def _ensure_chain() -> None:
    """Initialise biomech_chain on first entry to live capture, based
    on whether the user picked Full Assessment or a single movement."""
    chain = st.session_state.get("biomech_chain")
    if chain:
        return
    body_part = st.session_state.get("biomech_body_part")
    full      = st.session_state.get("biomech_full_assessment", False)
    if full:
        movements_table = (SHOULDER_MOVEMENTS if body_part == "shoulder"
                           else NECK_MOVEMENTS)
        chain = [m[0] for m in movements_table]
    else:
        chosen = st.session_state.get("biomech_movement")
        chain = [chosen] if chosen else []
    st.session_state["biomech_chain"]     = chain
    st.session_state["biomech_chain_idx"] = 0


def _current_chain_movement() -> str | None:
    chain = st.session_state.get("biomech_chain") or []
    idx   = st.session_state.get("biomech_chain_idx", 0)
    if idx >= len(chain):
        return None
    return chain[idx]


def _advance_chain_or_finish() -> None:
    """Advance to the next movement in the chain. If the chain is
    exhausted, jump to the report screen."""
    st.session_state["biomech_chain_idx"] += 1
    if st.session_state["biomech_chain_idx"] >= len(st.session_state["biomech_chain"]):
        st.session_state["biomech_chain"]     = []
        st.session_state["biomech_chain_idx"] = 0
        _set_step("report")
    else:
        st.rerun()


def _save_peak_to_recordings(state: dict, body_part: str,
                              movement: str, side: str) -> None:
    """Persist a captured peak from the live processor into
    biomech_recordings, then trigger chain advance."""
    if state["peak_magnitude"] <= 0:
        # Nothing captured — treat as skip
        _advance_chain_or_finish()
        return
    recordings = st.session_state["biomech_recordings"]
    side_key = side if body_part == "shoulder" else "—"
    recordings.setdefault(movement, {})[side_key] = {
        "peak_angle":     state["peak_angle"],
        "peak_magnitude": state["peak_magnitude"],
        "valid_frames":   state["valid_frames"],
        "total_frames":   state["total_frames"],
        "fps":            0.0,  # not measured for live mode
        "body_part":      body_part,
        "side":           side_key,
    }
    st.session_state["biomech_recordings"] = recordings
    _advance_chain_or_finish()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _set_step(step: str) -> None:
    st.session_state["biomech_step"] = step
    st.rerun()


def _back_to_main() -> None:
    """Used by inline 'Back' buttons inside the flow when the user wants
    to step back from a sub-step rather than exit to main menu."""
    pass  # placeholder — sub-step Back buttons use _set_step directly


# ──────────────────────────────────────────────
# Step renderers
# ──────────────────────────────────────────────
def _render_patient() -> None:
    """Patient details form — same field set as the gait wizard."""
    p = st.session_state["biomech_patient"]

    st.markdown(
        '<div class="wizard-title">Patient Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name *",
                             value=p.get("name", ""),
                             placeholder="Jane Doe",
                             key="bio_name")
        age = st.number_input("Age *",
                              min_value=1, max_value=120, step=1,
                              value=int(p.get("age") or 30),
                              key="bio_age")
        height_cm = st.number_input("Height (cm) *",
                                    min_value=50, max_value=250, step=1,
                                    value=int(p.get("height_cm") or DEFAULT_HEIGHT_CM),
                                    key="bio_height")
        assess_date = st.date_input("Assessment Date",
                                    value=p.get("assessment_date") or date.today(),
                                    key="bio_date")
    with col2:
        patient_id = st.text_input("Patient ID (optional)",
                                   value=p.get("patient_id", ""),
                                   placeholder="auto-generated if blank",
                                   key="bio_pid")
        gender_idx = (
            GENDER_OPTIONS.index(p["gender"])
            if p.get("gender") in GENDER_OPTIONS
            else 0
        )
        gender = st.selectbox("Gender *",
                              GENDER_OPTIONS,
                              index=gender_idx,
                              key="bio_gender")
        weight_kg = st.number_input("Weight (kg)",
                                    min_value=10.0, max_value=300.0, step=0.1,
                                    value=float(p.get("weight_kg") or 70.0),
                                    key="bio_weight")
        clinician = st.text_input("Referring Clinician",
                                  value=p.get("clinician", ""),
                                  key="bio_clin")

    notes = st.text_area("Clinical Notes (optional)",
                         value=p.get("notes", ""),
                         height=100,
                         key="bio_notes")

    required_ok = (
        bool(name.strip())
        and bool(gender)
        and 1 <= int(age) <= 120
        and 50 <= int(height_cm) <= 250
    )

    _, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Next →", disabled=not required_ok, type="primary",
                     use_container_width=True, key="bio_next_patient"):
            new_id = patient_id.strip() or (
                f"BIO-{date.today().strftime('%Y%m%d')}-"
                f"{uuid.uuid4().hex[:6].upper()}"
            )
            st.session_state["biomech_patient"] = {
                "name": name.strip(),
                "patient_id": new_id,
                "age": int(age),
                "gender": gender,
                "height_cm": int(height_cm),
                "weight_kg": float(weight_kg),
                "assessment_date": assess_date,
                "clinician": clinician.strip(),
                "notes": notes.strip(),
            }
            _set_step("body_part")


def _render_body_part() -> None:
    """Two big cards: Shoulder | Neck."""
    st.markdown(
        '<div class="wizard-title">Choose a body part to assess</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="text-align:center; color:#94A3B8; font-size:14px; '
        'margin-bottom:24px;">'
        'Each body part has its own set of standard ROM movements.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(
            '<div class="mode-card">'
            '<div>'
            '<div class="mode-card-icon">🦾</div>'
            '<div class="mode-card-title">Shoulder</div>'
            '<div class="mode-card-desc">'
            '6 standard ROM movements: flexion, extension, abduction, '
            'adduction, external rotation, internal rotation.'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Select →", key="bio_pick_shoulder",
                     type="primary", use_container_width=True):
            st.session_state["biomech_body_part"] = "shoulder"
            st.session_state["biomech_movement"] = None
            _set_step("movement")

    with col_b:
        st.markdown(
            '<div class="mode-card">'
            '<div>'
            '<div class="mode-card-icon">🙆</div>'
            '<div class="mode-card-title">Neck</div>'
            '<div class="mode-card-desc">'
            '4 cervical ROM movements: flexion, extension, lateral '
            'flexion, rotation.'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Select →", key="bio_pick_neck",
                     type="primary", use_container_width=True):
            st.session_state["biomech_body_part"] = "neck"
            st.session_state["biomech_movement"] = None
            _set_step("movement")

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back", key="bio_back_body",
                     use_container_width=True):
            _set_step("patient")


def _render_movement() -> None:
    """Movement chooser. First card is 'Full Assessment' (chains all
    movements). Remaining cards are individual movements with their
    layperson description + clinical normal range."""
    body_part = st.session_state.get("biomech_body_part")
    if body_part not in ("shoulder", "neck"):
        st.warning("No body part selected. Going back.")
        _set_step("body_part")
        return

    movements = SHOULDER_MOVEMENTS if body_part == "shoulder" else NECK_MOVEMENTS
    body_label = body_part.capitalize()

    st.markdown(
        f'<div class="wizard-title">Choose a {body_label.lower()} '
        f'movement to assess</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="text-align:center; color:#94A3B8; font-size:14px; '
        'margin-bottom:24px;">'
        f'Pick a single movement, or run the full {body_label} '
        'battery to assess all of them in sequence.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Full-assessment card — full width across two columns.
    st.markdown(
        '<div class="mode-card" style="margin-bottom:18px;">'
        '<div>'
        '<div class="mode-card-icon">🎯</div>'
        f'<div class="mode-card-title">Full {body_label} Assessment</div>'
        '<div class="mode-card-desc">'
        f'Run all {len(movements)} movements in sequence and produce a '
        f'combined report. Best for an initial baseline or follow-up.'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button(f"Start Full {body_label} Assessment →",
                 key="bio_pick_full", type="primary",
                 use_container_width=True):
        st.session_state["biomech_movement"]        = None  # full battery
        st.session_state["biomech_full_assessment"] = True
        _set_step("mode")

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:14px; font-weight:500; color:#CBD5E1; '
        'margin-bottom:8px;">— OR pick a single movement —</div>',
        unsafe_allow_html=True,
    )

    # Per-movement cards in 2-column grid
    for i in range(0, len(movements), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j >= len(movements):
                continue
            key, title, desc, normal_str = movements[i + j]
            with col:
                st.markdown(
                    f'<div class="app-card" style="text-align:center; '
                    f'min-height:160px; padding:20px;">'
                    f'<div style="font-size:16px; font-weight:600; '
                    f'color:#F1F5F9; margin-bottom:6px;">{title}</div>'
                    f'<div style="color:#CBD5E1; font-size:13px; '
                    f'line-height:1.45; margin-bottom:8px;">{desc}</div>'
                    f'<div style="color:#94A3B8; font-size:12px;">'
                    f'Normal: {normal_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Assess {title}",
                             key=f"bio_pick_movement_{key}",
                             use_container_width=True):
                    st.session_state["biomech_movement"]        = key
                    st.session_state["biomech_full_assessment"] = False
                    _set_step("mode")

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back", key="bio_back_movement",
                     use_container_width=True):
            _set_step("body_part")


def _movement_label(body_part: str, movement: str) -> str:
    """Display name for a movement key, e.g. 'external_rotation' →
    'External Rotation'. Falls back to title-case if unknown."""
    table = SHOULDER_MOVEMENTS if body_part == "shoulder" else NECK_MOVEMENTS
    for key, title, _, _ in table:
        if key == movement:
            return title
    return movement.replace("_", " ").title()


def _render_mode() -> None:
    """Capture Mode chooser. Two cards: Live Camera (real-time WebRTC
    via streamlit-webrtc) | Video Upload (frame-by-frame analyzer).

    For Full Assessment, only Live Camera is offered — Upload requires
    one video per movement which doesn't fit the chained UX. (Users can
    still run individual movements via Upload after Full → Single.)

    Side selector (Right | Left) appears for shoulder movements; neck
    has no per-side laterality at this level.
    """
    body_part = st.session_state.get("biomech_body_part")
    movement  = st.session_state.get("biomech_movement")
    full      = st.session_state.get("biomech_full_assessment", False)

    if body_part not in ("shoulder", "neck"):
        st.warning("No body part selected. Going back.")
        _set_step("body_part")
        return

    st.markdown(
        '<div class="wizard-title">Capture Mode</div>',
        unsafe_allow_html=True,
    )

    if full:
        title_str = f"Full {body_part.capitalize()} Assessment"
    else:
        title_str = (f"{body_part.capitalize()} — "
                     f"{_movement_label(body_part, movement)}")
    st.markdown(
        f'<div style="text-align:center; color:#94A3B8; font-size:14px; '
        f'margin-bottom:18px;">'
        f'Selected: <b style="color:#F1F5F9;">{title_str}</b></div>',
        unsafe_allow_html=True,
    )

    # Side selector (shoulder only)
    if body_part == "shoulder":
        st.markdown(
            '<div style="font-size:14px; font-weight:500; color:#CBD5E1; '
            'margin-bottom:6px;">Side to assess</div>',
            unsafe_allow_html=True,
        )
        current_side = st.session_state.get("biomech_side", "Right")
        side = st.radio(
            "Side", ["Right", "Left"],
            index=0 if current_side != "Left" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="bio_side_radio",
        )
        st.session_state["biomech_side"] = side
        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(
            '<div class="mode-card">'
            '<div>'
            '<div class="mode-card-icon">📹</div>'
            '<div class="mode-card-title">Live Camera</div>'
            '<div class="mode-card-desc">'
            'Continuous real-time tracking with on-screen pose overlay '
            'and live angle / peak readout. Click Show Analysis when '
            'you reach the peak. Required for Full Assessment.'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Start Live Capture →", key="bio_pick_live",
                     type="primary", use_container_width=True):
            st.session_state["biomech_capture_mode"] = "live"
            # Reset chain so _ensure_chain rebuilds it for this run
            st.session_state["biomech_chain"]     = []
            st.session_state["biomech_chain_idx"] = 0
            _set_step("capture")

    with col_b:
        upload_disabled = full
        card_style    = ' style="opacity:0.55;"' if upload_disabled else ""
        disabled_note = (
            '<div style="color:#94A3B8; font-size:12px; '
            'font-style:italic;">Single movement only</div>'
            if upload_disabled else ""
        )
        st.markdown(
            f'<div class="mode-card"{card_style}>'
            '<div>'
            '<div class="mode-card-icon">📁</div>'
            '<div class="mode-card-title">Video Upload</div>'
            '<div class="mode-card-desc">'
            'Upload a recorded video of the movement. Frame-by-frame '
            'analysis extracts the peak angle.'
            '</div>'
            '</div>'
            f'{disabled_note}'
            '</div>',
            unsafe_allow_html=True,
        )
        if upload_disabled:
            st.button("Upload (single only)", disabled=True,
                      use_container_width=True,
                      key="bio_pick_upload_disabled")
        else:
            if st.button("Upload Video →", key="bio_pick_upload",
                         use_container_width=True):
                st.session_state["biomech_capture_mode"] = "upload"
                _set_step("capture")

    if full:
        st.caption(
            "💡 Video Upload only handles one movement at a time. "
            "For multi-movement Full Assessment, use Live Camera."
        )

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back", key="bio_back_mode",
                     use_container_width=True):
            _set_step("movement")


def _render_capture() -> None:
    """Dispatch to the upload or live capture screen."""
    capture_mode = st.session_state.get("biomech_capture_mode")
    if capture_mode == "live":
        _render_capture_live()
        return
    if capture_mode == "upload":
        _render_capture_upload()
        return
    st.warning("No capture mode selected. Going back.")
    if st.button("← Back to Mode", key="bio_back_capture_no_mode"):
        _set_step("mode")


def _render_capture_upload() -> None:
    """Upload + analyze (video file → frame-by-frame → peak)."""
    body_part = st.session_state.get("biomech_body_part")
    movement  = st.session_state.get("biomech_movement")
    side      = st.session_state.get("biomech_side", "Right")

    movement_title = _movement_label(body_part, movement)
    side_str = f" — {side} side" if body_part == "shoulder" else ""
    st.markdown(
        f'<div class="wizard-title">Upload Video — '
        f'{body_part.capitalize()} {movement_title}{side_str}</div>',
        unsafe_allow_html=True,
    )

    # Movement-specific instructions
    instructions = MOVEMENT_INSTRUCTIONS.get((body_part, movement), [])
    if instructions:
        items_html = "".join(
            f'<div style="margin:4px 0;">'
            f'<b style="color:#3B82F6;">{i + 1}.</b> {step}</div>'
            for i, step in enumerate(instructions)
        )
        st.markdown(
            f'<div class="wizard-info-strip">'
            f'<div style="font-weight:600; color:#F1F5F9; margin-bottom:8px;">'
            f'Recording instructions</div>{items_html}</div>',
            unsafe_allow_html=True,
        )

    # Rotation-specific disclaimer
    if _is_rotation_movement(body_part, movement):
        st.warning(
            "⚠️ Rotation measurements from 2D video are approximate. "
            "For precise clinical measurements, use a goniometer."
        )

    # File uploader
    uploaded = st.file_uploader(
        "Upload video (MP4 / MOV / AVI)",
        type=["mp4", "mov", "avi", "mkv"],
        key="bio_uploader",
    )
    if uploaded is not None:
        size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        st.markdown(
            f'<div class="wizard-file-info">'
            f'📁 <b>{uploaded.name}</b>  ·  {size_mb:.2f} MB'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.session_state["biomech_video_file"] = uploaded
        with st.expander("Preview", expanded=False):
            st.video(uploaded)

    file_ready = st.session_state.get("biomech_video_file") is not None
    if st.button("Analyze →",
                 type="primary",
                 disabled=not file_ready,
                 use_container_width=True,
                 key="bio_analyze"):
        if _run_capture_and_store():
            _set_step("report")

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back", key="bio_back_capture",
                     use_container_width=True):
            _set_step("mode")


def _run_capture_and_store() -> bool:
    """Save uploaded video to a temp file, analyze, store the result
    in `biomech_recordings`. Returns True on success."""
    video     = st.session_state.get("biomech_video_file")
    body_part = st.session_state.get("biomech_body_part")
    movement  = st.session_state.get("biomech_movement")
    side      = st.session_state.get("biomech_side", "Right")
    if video is None or movement is None or body_part is None:
        st.error("Missing video, movement, or body part — cannot analyze.")
        return False

    file_bytes = video.getvalue()
    file_name  = video.name or "video.mp4"
    suffix = os.path.splitext(file_name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with st.status("Analyzing video …", expanded=True) as status:
            st.write("Loading pose model …")
            _ = _get_pose_estimator()  # warm cache
            st.write("Extracting poses + computing angles …")
            result = _run_biomech_upload_analysis(
                tmp_path, body_part, movement, side,
            )
            if result["valid_frames"] == 0:
                status.update(label="Analysis failed.", state="error")
                st.error(
                    "No frames had high-confidence pose landmarks. "
                    "Re-record with better lighting or position."
                )
                return False

            # Store keyed by movement → side → result. Side is uniform
            # 'neck' for cervical movements (no laterality at this level).
            recordings = st.session_state["biomech_recordings"]
            side_key = side if body_part == "shoulder" else "—"
            recordings.setdefault(movement, {})[side_key] = {
                **result,
                "body_part": body_part,
                "side":      side_key,
            }
            st.session_state["biomech_recordings"] = recordings

            status.update(
                label=(
                    f"Done. Peak {body_part} {_movement_label(body_part, movement)}"
                    f"{' (' + side + ')' if body_part == 'shoulder' else ''}: "
                    f"{result['peak_magnitude']:.1f}°  "
                    f"({result['valid_frames']} valid / "
                    f"{result['total_frames']} total frames)"
                ),
                state="complete",
            )
        return True
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ──────────────────────────────────────────────
# Report screen + helpers
# ──────────────────────────────────────────────
_STATUS_GOOD    = ("Good",    "#10B981")
_STATUS_FAIR    = ("Fair",    "#F59E0B")
_STATUS_LIMITED = ("Limited", "#EF4444")


def _normal_for(body_part: str, movement: str) -> dict | None:
    table = (SHOULDER_NORMAL_RANGES if body_part == "shoulder"
             else NECK_NORMAL_RANGES)
    return table.get(movement)


def _classify(measured: float, target: float) -> tuple[str, str, float]:
    """Return (status_label, color_hex, pct_of_normal)."""
    if target <= 0:
        return ("—", "#94A3B8", 0.0)
    pct = (measured / target) * 100.0
    if pct >= 90.0:
        label, color = _STATUS_GOOD
    elif pct >= 75.0:
        label, color = _STATUS_FAIR
    else:
        label, color = _STATUS_LIMITED
    return (label, color, pct)


def _flatten_recordings() -> list[dict]:
    """Turn the nested biomech_recordings dict into a flat list of rows
    suitable for the table + bar chart."""
    rows = []
    for movement, by_side in (st.session_state.get("biomech_recordings") or {}).items():
        for side, data in by_side.items():
            body_part = data.get("body_part", "shoulder")
            normal = _normal_for(body_part, movement) or {}
            target = float(normal.get("target", 0.0))
            rng = normal.get("range", (0.0, 0.0))
            measured = float(data.get("peak_magnitude", 0.0))
            label, color, pct = _classify(measured, target)
            rows.append({
                "body_part": body_part,
                "movement":  movement,
                "title":     _movement_label(body_part, movement),
                "side":      side,
                "measured":  measured,
                "target":    target,
                "range":     rng,
                "pct":       pct,
                "status":    label,
                "color":     color,
                "valid_frames": data.get("valid_frames", 0),
                "total_frames": data.get("total_frames", 0),
            })
    return rows


_SHOULDER_EDU = (
    "The shoulder is a ball-and-socket joint with the greatest range of "
    "motion of any joint in the body. Reduced range of motion in shoulder "
    "movements can indicate rotator cuff issues, adhesive capsulitis "
    "('frozen shoulder'), impingement syndromes, or post-injury stiffness. "
    "Bilateral asymmetry — significantly different ROM on left vs right — "
    "is often more clinically informative than absolute values, since "
    "individual baselines vary widely."
)

_NECK_EDU = (
    "The cervical spine allows the head to move in all directions. "
    "Reduced cervical range of motion is commonly associated with muscle "
    "strain, cervical spondylosis, disc herniation, or whiplash injury. "
    "Approximately 50% of cervical rotation occurs at the atlanto-axial "
    "(C1–C2) joint; reductions in rotation specifically may point to "
    "upper-cervical involvement, while flexion/extension deficits more "
    "often reflect lower-cervical pathology."
)


def _build_interpretation(rows: list[dict]) -> str:
    """Auto-generate a short summary sentence per row."""
    if not rows:
        return ""
    sentences = []
    for r in rows:
        side_str = f" ({r['side']})" if r["side"] not in ("—", "") else ""
        rng = r["range"]
        rng_str = (f"{rng[0]:.0f}°–{rng[1]:.0f}°"
                   if rng[0] != rng[1]
                   else f"{rng[0]:.0f}°")
        sentences.append(
            f"{r['title']}{side_str} measured {r['measured']:.1f}°, which is "
            f"{r['pct']:.0f}% of the {rng_str} normal range — {r['status'].lower()}."
        )
    return " ".join(sentences)


def _build_report_pdf(patient: dict, rows: list[dict],
                       body_part: str) -> bytes:
    """Single-page A4 PDF with patient header + results table + bar chart.
    Uses matplotlib PdfPages so no extra dependency."""
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Single landscape page with three regions: header, table, chart.
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 2.2, 3.5],
                              hspace=0.4)

        # ── Header ────────────────────────────────────────────────
        ax_h = fig.add_subplot(gs[0]); ax_h.axis("off")
        ax_h.text(0.5, 0.92, "Biomechanical Assessment Report",
                  ha="center", fontsize=22, fontweight="bold",
                  color="#1A1A1A")
        ax_h.text(0.5, 0.78, f"Body part assessed: {body_part.capitalize()}",
                  ha="center", fontsize=13, color="#6B7280")
        info_pairs = [
            ("Name",     patient.get("name", "—")),
            ("ID",       patient.get("patient_id", "—")),
            ("Age",      str(patient.get("age", "—"))),
            ("Gender",   patient.get("gender", "—")),
            ("Date",     str(patient.get("assessment_date", "—"))),
            ("Clinician", patient.get("clinician", "") or "—"),
        ]
        x_step = 1.0 / (len(info_pairs))
        for i, (label, value) in enumerate(info_pairs):
            x = x_step / 2 + i * x_step
            ax_h.text(x, 0.50, label, ha="center", fontsize=9,
                      color="#6B7280")
            ax_h.text(x, 0.32, str(value), ha="center", fontsize=11,
                      fontweight="bold", color="#1A1A1A")

        # ── Results table ─────────────────────────────────────────
        ax_t = fig.add_subplot(gs[1]); ax_t.axis("off")
        if rows:
            cell_text = []
            cell_colors = []
            header = ["Movement", "Side", "Measured",
                      "Normal", "% of Normal", "Status"]
            for r in rows:
                rng = r["range"]
                rng_str = (f"{rng[0]:.0f}°–{rng[1]:.0f}°"
                           if rng[0] != rng[1] else f"{rng[0]:.0f}°")
                row_cells = [
                    r["title"],
                    r["side"],
                    f"{r['measured']:.1f}°",
                    rng_str,
                    f"{r['pct']:.0f}%",
                    r["status"],
                ]
                cell_text.append(row_cells)
                cell_colors.append(["white"] * 5 + [r["color"]])
            table = ax_t.table(
                cellText=cell_text,
                colLabels=header,
                cellLoc="center", loc="center",
                colColours=["#F3F4F6"] * len(header),
                cellColours=cell_colors,
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.4)
            # Color the status text white for contrast
            for i in range(1, len(cell_text) + 1):
                table[i, 5].get_text().set_color("white")
                table[i, 5].get_text().set_fontweight("bold")
        else:
            ax_t.text(0.5, 0.5, "No measurements recorded.",
                      ha="center", fontsize=12, color="#6B7280")

        # ── Bar chart ─────────────────────────────────────────────
        ax_c = fig.add_subplot(gs[2])
        if rows:
            labels = [
                f"{r['title']}" + (f" ({r['side']})" if r["side"] not in ("—", "") else "")
                for r in rows
            ]
            measured = [r["measured"] for r in rows]
            target   = [r["target"]   for r in rows]
            colors   = [r["color"]    for r in rows]
            y_pos = list(range(len(rows)))
            bar_height = 0.36
            ax_c.barh([y - bar_height / 2 for y in y_pos], measured,
                      height=bar_height, color=colors, label="Measured")
            ax_c.barh([y + bar_height / 2 for y in y_pos], target,
                      height=bar_height, color="#94A3B8",
                      alpha=0.55, label="Target (normal)")
            ax_c.set_yticks(y_pos)
            ax_c.set_yticklabels(labels, fontsize=9)
            ax_c.invert_yaxis()
            ax_c.set_xlabel("Angle (°)", color="#374151", fontsize=10)
            ax_c.tick_params(colors="#374151")
            for sp in ax_c.spines.values():
                sp.set_color("#E5E7EB")
            ax_c.spines["top"].set_visible(False)
            ax_c.spines["right"].set_visible(False)
            ax_c.grid(True, axis="x", color="#E5E7EB", linewidth=0.6)
            ax_c.legend(loc="lower right", fontsize=9, frameon=False)
        else:
            ax_c.axis("off")

        fig.suptitle("",  y=0.99)  # placeholder; header carries the title
        # Disclaimer footer
        fig.text(0.5, 0.02,
                 "Descriptive output from a 2D pose-estimation pipeline. "
                 "Rotation measurements are inherently approximate. "
                 "Clinical interpretation belongs to a qualified professional.",
                 ha="center", fontsize=8, color="#6B7280", style="italic")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return buf.getvalue()


def _render_report() -> None:
    """Patient header + results table + bar chart + interpretation +
    educational block + Download PDF + action buttons + disclaimer."""
    body_part = st.session_state.get("biomech_body_part") or "—"
    patient   = st.session_state.get("biomech_patient", {}) or {}
    rows      = _flatten_recordings()

    st.markdown(
        '<div class="wizard-title">Assessment Report</div>',
        unsafe_allow_html=True,
    )

    # Header strip
    header = (
        f"<b style='color:#F1F5F9;'>{patient.get('name', '—')}</b>"
        f"  ·  ID: {patient.get('patient_id', '—')}"
        f"  ·  Date: {patient.get('assessment_date', '—')}"
        f"  ·  Body part: <b style='color:#F1F5F9;'>{body_part.capitalize()}</b>"
    )
    st.markdown(
        f'<div class="wizard-info-strip">{header}</div>',
        unsafe_allow_html=True,
    )

    if not rows:
        st.warning(
            "No measurements recorded yet. Go back and run at least one "
            "movement assessment."
        )
        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("← Back", key="bio_report_back_no_data"):
                _set_step("mode")
        return

    # ── Results table (rendered as Streamlit table with color spans) ──
    st.markdown(
        '<div class="wizard-section-heading">Results</div>',
        unsafe_allow_html=True,
    )
    table_rows_html = []
    for r in rows:
        rng = r["range"]
        rng_str = (f"{rng[0]:.0f}°–{rng[1]:.0f}°"
                   if rng[0] != rng[1] else f"{rng[0]:.0f}°")
        table_rows_html.append(
            f"<tr>"
            f"<td>{r['title']}</td>"
            f"<td style='text-align:center;'>{r['side']}</td>"
            f"<td style='text-align:right;'>{r['measured']:.1f}°</td>"
            f"<td style='text-align:right;'>{rng_str}</td>"
            f"<td style='text-align:right;'>{r['pct']:.0f}%</td>"
            f"<td style='text-align:center;'>"
            f"<span style='background:{r['color']}; color:white; "
            f"padding:3px 10px; border-radius:6px; font-size:12px; "
            f"font-weight:600;'>{r['status']}</span></td>"
            f"</tr>"
        )
    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table style="width:100%; border-collapse:collapse; '
        'color:#CBD5E1; font-size:14px;">'
        '<thead><tr style="border-bottom:1px solid #334155;">'
        '<th style="text-align:left; padding:8px;">Movement</th>'
        '<th style="text-align:center; padding:8px;">Side</th>'
        '<th style="text-align:right; padding:8px;">Measured</th>'
        '<th style="text-align:right; padding:8px;">Normal</th>'
        '<th style="text-align:right; padding:8px;">% of Normal</th>'
        '<th style="text-align:center; padding:8px;">Status</th>'
        '</tr></thead>'
        '<tbody>' + "".join(
            row.replace("<td>", "<td style='padding:8px;'>")
               .replace("<td style='text-align:center;'>",
                        "<td style='padding:8px; text-align:center;'>")
               .replace("<td style='text-align:right;'>",
                        "<td style='padding:8px; text-align:right;'>")
            for row in table_rows_html
        ) + '</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # ── Bar chart (plotly horizontal) ─────────────────────────────────
    st.markdown(
        '<div class="wizard-section-heading" style="margin-top:24px;">'
        'Measured vs Normal</div>',
        unsafe_allow_html=True,
    )
    import plotly.graph_objects as go
    labels = [
        r["title"] + (f" ({r['side']})" if r["side"] not in ("—", "") else "")
        for r in rows
    ]
    measured = [r["measured"] for r in rows]
    target   = [r["target"]   for r in rows]
    colors   = [r["color"]    for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=measured, orientation="h", name="Measured",
        marker=dict(color=colors),
        text=[f"{m:.1f}°" for m in measured], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Measured: %{x:.1f}°<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=target, orientation="h", name="Target (normal)",
        marker=dict(color="#94A3B8", opacity=0.55),
        hovertemplate="<b>%{y}</b><br>Target: %{x:.1f}°<extra></extra>",
    ))
    max_x = max(max(measured + target), 1.0) * 1.15
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#1E293B", paper_bgcolor="#1E293B",
        font=dict(color="#CBD5E1"),
        xaxis=dict(title="Angle (°)", range=[0, max_x],
                   gridcolor="#334155", showline=True, linecolor="#475569"),
        yaxis=dict(title="", autorange="reversed",
                   gridcolor="#334155", showline=True, linecolor="#475569"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color="#CBD5E1")),
        margin=dict(l=120, r=30, t=20, b=70),
        height=max(280, 60 * len(rows) + 100),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})

    # ── Clinical interpretation ──────────────────────────────────────
    st.markdown(
        '<div class="wizard-section-heading" style="margin-top:24px;">'
        'Clinical interpretation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="obs-card" style="margin-top:8px;">'
        f'{_build_interpretation(rows)}</div>',
        unsafe_allow_html=True,
    )

    # ── Educational block about this body part ──────────────────────
    edu = _SHOULDER_EDU if body_part == "shoulder" else _NECK_EDU
    st.markdown(
        '<div class="wizard-section-heading" style="margin-top:24px;">'
        f'About {body_part} biomechanics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sug-card" style="margin-top:8px;">{edu}</div>',
        unsafe_allow_html=True,
    )

    # Rotation disclaimer if any rotation row is present
    has_rotation = any(_is_rotation_movement(r["body_part"], r["movement"])
                       for r in rows)
    if has_rotation:
        st.warning(
            "⚠️ Rotation measurements from 2D video are approximate. "
            "For precise clinical measurements, use a goniometer."
        )

    # ── Action buttons ───────────────────────────────────────────────
    # Note: do NOT do `from app import ...` here. Streamlit runs the main
    # script as `__main__`; importing it under the name `app` would cause
    # the module body (including st.set_page_config) to execute a second
    # time and crash. State resets are performed inline via session_state.
    st.markdown("---")
    col_pdf, col_main = st.columns(2)
    try:
        pdf_bytes = _build_report_pdf(patient, rows, body_part)
        with col_pdf:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=(
                    f"BiomechReport_{patient.get('patient_id', 'report')}"
                    f"_{patient.get('assessment_date', 'date')}.pdf"
                ),
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="bio_dl_pdf",
            )
    except Exception as exc:
        with col_pdf:
            st.error(f"PDF build failed: {exc}")
    with col_main:
        if st.button("← Back to Main Menu", key="bio_main_menu",
                     use_container_width=True):
            # Reset biomech state inline + clear app_mode. Equivalent to
            # _reset_biomech_state(keep_patient=False) + _go_to_main_menu()
            # but without re-importing app.py.
            for k, v in {
                "biomech_step":            "patient",
                "biomech_patient":         {},
                "biomech_body_part":       None,
                "biomech_movement":        None,
                "biomech_full_assessment": False,
                "biomech_side":            "Right",
                "biomech_capture_mode":    None,
                "biomech_video_file":      None,
                "biomech_recordings":      {},
                "app_mode":                None,
            }.items():
                st.session_state[k] = v
            st.rerun()

    # ── Footer disclaimer ────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer" style="margin-top:24px;">'
        'These measurements are derived from 2D pose estimation and are '
        'approximate. Rotation measurements are especially sensitive to '
        'camera angle and should be considered indicative only. Clinical '
        'interpretation requires a qualified professional\'s assessment.'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# LIVE CAMERA CAPTURE SCREEN  (split-screen: instructions | webcam)
# ══════════════════════════════════════════════════════════════════
def _render_capture_live() -> None:
    """Two-column live capture screen — CONTINUOUS-CAPTURE mode.

    Right: streamlit-webrtc webcam stream with skeleton overlay +
           on-frame current angle, peak, and frame counter.
    Left:  movement instructions, posture-status badge, live readout,
           buttons: Show Analysis · Reset Peak · Skip · Cancel.

    There is no Start/Stop — the moment the camera connects, every
    frame's angle is computed and the running peak is tracked. The
    user clicks **Show Analysis →** when satisfied with the take to
    save the peak and advance (next movement in the chain, or report
    on the last). **Reset Peak** wipes the running peak so the user
    can redo the movement without losing the camera connection."""
    from streamlit_webrtc import webrtc_streamer, WebRtcMode

    body_part = st.session_state.get("biomech_body_part")
    side      = st.session_state.get("biomech_side", "Right")

    if body_part not in ("shoulder", "neck"):
        st.warning("No body part selected. Going back.")
        _set_step("body_part")
        return

    _ensure_chain()
    movement = _current_chain_movement()
    if movement is None:
        _set_step("report")
        return

    chain_pos = st.session_state["biomech_chain_idx"] + 1
    chain_len = len(st.session_state["biomech_chain"])

    side_str = f" — {side} side" if body_part == "shoulder" else ""
    progress_str = (f' <span style="color:#94A3B8; font-size:0.8em;">'
                    f'(Movement {chain_pos} of {chain_len})</span>'
                    if chain_len > 1 else "")
    st.markdown(
        f'<div class="wizard-title">Live Capture — '
        f'{body_part.capitalize()} {_movement_label(body_part, movement)}'
        f'{side_str}{progress_str}</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1, 1], gap="medium")

    # ─── Right column: WebRTC streamer ─────────────────────────────
    with col_r:
        # Per-movement key forces a fresh streamer (and processor) for
        # each chained movement so peak counters can't leak across.
        ctx = webrtc_streamer(
            key=f"biomech_live_{movement}_{chain_pos}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=BiomechVideoProcessor,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                ],
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if not ctx.state.playing:
            st.info(
                "Click **START** above to enable the webcam. If the "
                "camera doesn't connect within ~10 seconds, your network "
                "may block WebRTC — switch to Video Upload mode."
            )

    # Configure the processor each rerun (safe — lock-guarded). Reads
    # latest state for the live readout below.
    if ctx.video_processor is not None:
        ctx.video_processor.configure(body_part, movement, side)
        state = ctx.video_processor.get_state()
    else:
        state = None

    # ─── Left column: instructions + status + buttons ───────────────
    with col_l:
        # Instructions
        instructions = MOVEMENT_INSTRUCTIONS.get((body_part, movement), [])
        if instructions:
            items_html = "".join(
                f'<div style="margin:4px 0;">'
                f'<b style="color:#3B82F6;">{i + 1}.</b> {step}</div>'
                for i, step in enumerate(instructions)
            )
            st.markdown(
                f'<div class="wizard-info-strip">'
                f'<div style="font-weight:600; color:#F1F5F9; margin-bottom:8px;">'
                f'Movement instructions</div>{items_html}</div>',
                unsafe_allow_html=True,
            )

        # Rotation disclaimer
        if _is_rotation_movement(body_part, movement):
            st.warning(
                "⚠️ Rotation measurements from 2D video are approximate. "
                "For precise clinical measurements, use a goniometer."
            )

        # Live readout (only when the camera is active)
        if state is not None:
            st.markdown(
                '<div class="wizard-section-heading">Live status</div>',
                unsafe_allow_html=True,
            )

            posture = state["posture_status"]
            badges = {
                "good":           ("#10B981", "✓ Subject visible"),
                "low_visibility": ("#F59E0B", "⚠ Required landmarks below visibility threshold"),
                "no_landmarks":   ("#EF4444", "✗ No subject detected — check position / lighting"),
                "idle":           ("#64748B", "— waiting for camera"),
            }
            colour, text = badges.get(posture, badges["idle"])
            st.markdown(
                f'<div style="display:inline-block; background:{colour}; '
                f'color:white; padding:5px 12px; border-radius:6px; '
                f'font-size:13px; font-weight:500;">{text}</div>',
                unsafe_allow_html=True,
            )

            # Live angle + peak + frame count readouts
            cur = state["current_angle"]
            peak = state["peak_magnitude"]
            cur_str  = f"{cur:+.1f}°" if cur is not None else "—"
            peak_str = f"{peak:.1f}°" if peak > 0 else "—"
            st.markdown(
                f'<div style="display:flex; gap:24px; margin-top:14px; '
                f'flex-wrap:wrap;">'
                f'  <div>'
                f'    <div style="font-size:12px; color:#94A3B8; '
                f'      text-transform:uppercase; letter-spacing:0.5px;">'
                f'      Current angle</div>'
                f'    <div style="font-size:30px; font-weight:600; '
                f'      color:#F1F5F9;">{cur_str}</div>'
                f'  </div>'
                f'  <div>'
                f'    <div style="font-size:12px; color:#94A3B8; '
                f'      text-transform:uppercase; letter-spacing:0.5px;">'
                f'      Peak (so far)</div>'
                f'    <div style="font-size:30px; font-weight:600; '
                f'      color:#3B82F6;">{peak_str}</div>'
                f'  </div>'
                f'  <div>'
                f'    <div style="font-size:12px; color:#94A3B8; '
                f'      text-transform:uppercase; letter-spacing:0.5px;">'
                f'      Frames</div>'
                f'    <div style="font-size:24px; font-weight:600; '
                f'      color:#CBD5E1;">'
                f'      {state["valid_frames"]}'
                f'      <span style="font-size:14px; color:#64748B;">'
                f'      / {state["total_frames"]}</span></div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div style="margin-top:12px; padding:10px 14px; '
                'background:#0F172A; border:1px solid #334155; '
                'border-radius:6px; color:#94A3B8; font-size:13px;">'
                'Capture is continuous — perform the movement, then '
                'click <b style="color:#F1F5F9;">Show Analysis</b> when '
                'you reach the peak.'
                '</div>',
                unsafe_allow_html=True,
            )

        # Action buttons
        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

        if ctx.video_processor is None:
            st.info("Start the webcam (in the right panel) to begin tracking.")
        else:
            has_peak = state["peak_magnitude"] > 0
            show_label = ("Show Analysis →"
                          if chain_pos < chain_len
                          else "Show Analysis & Finish →")

            col_show, col_reset, col_skip = st.columns(3)
            with col_show:
                if st.button(show_label,
                             type="primary",
                             use_container_width=True,
                             disabled=not has_peak,
                             key=f"bio_live_show_{movement}_{chain_pos}"):
                    _save_peak_to_recordings(state, body_part, movement, side)
                    return
            with col_reset:
                if st.button("↻ Reset Peak",
                             use_container_width=True,
                             disabled=not has_peak,
                             key=f"bio_live_reset_{movement}_{chain_pos}"):
                    ctx.video_processor.reset_peak()
                    st.rerun()
            with col_skip:
                if st.button("Skip ⤏",
                             use_container_width=True,
                             key=f"bio_live_skip_{movement}_{chain_pos}"):
                    ctx.video_processor.reset_peak()
                    _advance_chain_or_finish()
                    return

        # Auto-poll the UI while the camera is active so the live
        # angle / peak / frame-count readouts tick. 0.7s cadence keeps
        # rerun load reasonable on Streamlit Cloud.
        if ctx.video_processor is not None:
            time.sleep(0.7)
            st.rerun()

    # Bottom Cancel link — wipes the chain and returns to Mode chooser.
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Cancel & Back",
                     key=f"bio_live_back_{movement}_{chain_pos}",
                     use_container_width=True):
            st.session_state["biomech_chain"]     = []
            st.session_state["biomech_chain_idx"] = 0
            _set_step("mode")


# ──────────────────────────────────────────────
# Top-level renderer
# ──────────────────────────────────────────────
_RENDERERS = {
    "patient":   _render_patient,
    "body_part": _render_body_part,
    "movement":  _render_movement,
    "mode":      _render_mode,
    "capture":   _render_capture,
    "report":    _render_report,
}


def render_biomech_flow() -> None:
    """Dispatch on biomech_step. Falls back to patient screen if the
    state contains an unrecognised step value."""
    step = st.session_state.get("biomech_step", "patient")
    renderer = _RENDERERS.get(step)
    if renderer is None:
        st.error(f"Unknown biomech step: {step!r}. Resetting.")
        st.session_state["biomech_step"] = "patient"
        st.rerun()
    else:
        renderer()
