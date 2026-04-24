"""
biomech_flow.py
Biomechanical Analysis flow for GaitVision (shoulder + neck range-of-motion).

Sub-flow tracked by st.session_state["biomech_step"]:
    patient   → body_part → movement → mode → capture → report

This module is reached only when st.session_state["app_mode"] == "biomech".
The render entry-point is `render_biomech_flow()`, called from app.py's
top-level dispatcher.

Status:
  Commit 3 (this commit) — patient form + body-part chooser + movement
                           chooser implemented. Capture (mode + screen)
                           and report still placeholder.
"""
from __future__ import annotations

import uuid
from datetime import date

import streamlit as st

from shoulder_engine import SHOULDER_NORMAL_RANGES
from neck_engine     import NECK_NORMAL_RANGES


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


def _render_mode_placeholder() -> None:
    """Placeholder — Capture Mode chooser lands in commit 4 (with the
    upload flow) and commit 5 (live camera, deferred)."""
    body_part = st.session_state.get("biomech_body_part") or "—"
    full = st.session_state.get("biomech_full_assessment", False)
    movement = st.session_state.get("biomech_movement")
    chosen = (f"Full {body_part.capitalize()} Assessment"
              if full else f"{body_part.capitalize()} {movement}")
    st.markdown(
        '<div class="wizard-title">Capture Mode</div>',
        unsafe_allow_html=True,
    )
    st.info(
        f"Selected: **{chosen}**\n\n"
        "Capture mode + analysis screens land in commit 4 of this "
        "feature (video upload). Live camera lands in commit 5 (deferred)."
    )
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back", key="bio_back_mode",
                     use_container_width=True):
            _set_step("movement")


def _render_capture_placeholder() -> None:
    st.warning("Capture screen not implemented yet — commit 4 / 5.")
    if st.button("← Back", key="bio_back_capture"):
        _set_step("mode")


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
    "mode":      _render_mode_placeholder,
    "capture":   _render_capture_placeholder,
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
