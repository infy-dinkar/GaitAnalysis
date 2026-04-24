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
    """Capture Mode chooser. Two cards (Live Camera disabled, Video
    Upload enabled), preceded by a Side toggle when the movement has
    laterality (all shoulder movements). Full-assessment is gracefully
    blocked — it requires live-camera multi-movement chaining which is
    deferred."""
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
        st.warning(
            f"**Full {body_part.capitalize()} Assessment** chains multiple "
            "movements in sequence. This requires live camera mode, which is "
            "coming in a follow-up release. For now, please pick a single "
            "movement and run it via video upload."
        )
        col_back, _ = st.columns([1, 5])
        with col_back:
            if st.button("← Pick a single movement",
                         key="bio_back_mode_full",
                         use_container_width=True):
                st.session_state["biomech_full_assessment"] = False
                _set_step("movement")
        return

    # Single-movement path
    st.markdown(
        f'<div style="text-align:center; color:#94A3B8; font-size:14px; '
        f'margin-bottom:18px;">'
        f'Movement: <b style="color:#F1F5F9;">'
        f'{body_part.capitalize()} — {_movement_label(body_part, movement)}'
        f'</b></div>',
        unsafe_allow_html=True,
    )

    # Side selector (shoulder only — neck has no laterality at this level).
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
            '<div class="mode-card" style="opacity:0.55;">'
            '<div>'
            '<div class="mode-card-icon">📹</div>'
            '<div class="mode-card-title">Live Camera</div>'
            '<div class="mode-card-desc">'
            'Real-time guided assessment with on-screen pose overlay '
            'and live angle readout.'
            '</div>'
            '</div>'
            '<div style="color:#94A3B8; font-size:12px; '
            'font-style:italic;">Coming soon</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.button("Coming soon", key="bio_pick_live",
                  disabled=True, use_container_width=True)
    with col_b:
        st.markdown(
            '<div class="mode-card">'
            '<div>'
            '<div class="mode-card-icon">📁</div>'
            '<div class="mode-card-title">Video Upload</div>'
            '<div class="mode-card-desc">'
            'Upload a recorded video of the movement. Frame-by-frame '
            'analysis extracts the peak angle.'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Upload Video →", key="bio_pick_upload",
                     type="primary", use_container_width=True):
            st.session_state["biomech_capture_mode"] = "upload"
            _set_step("capture")

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back", key="bio_back_mode",
                     use_container_width=True):
            _set_step("movement")


def _render_capture() -> None:
    """Upload + analyze. Live-camera mode is gated behind a 'coming
    soon' notice."""
    capture_mode = st.session_state.get("biomech_capture_mode")
    body_part    = st.session_state.get("biomech_body_part")
    movement     = st.session_state.get("biomech_movement")
    side         = st.session_state.get("biomech_side", "Right")

    if capture_mode != "upload":
        st.warning(
            "Live camera mode lands in a later release. Please go back "
            "and pick Video Upload."
        )
        if st.button("← Back to Mode", key="bio_back_capture_mode_only"):
            _set_step("mode")
        return

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


def _render_report_placeholder() -> None:
    st.warning("Report screen not implemented yet — commit 6.")
    if st.button("← Back to Body Part", key="bio_back_report"):
        _set_step("body_part")


# ──────────────────────────────────────────────
# Top-level renderer
# ──────────────────────────────────────────────
_RENDERERS = {
    "patient":   _render_patient,
    "body_part": _render_body_part,
    "movement":  _render_movement,
    "mode":      _render_mode,
    "capture":   _render_capture,
    "report":    _render_report_placeholder,
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
