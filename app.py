"""
app.py
Streamlit entry point for the Video-Based Gait Analysis System.
"""

import os
import urllib.request
import tempfile
import uuid
from datetime import date
import numpy as np
import streamlit as st
import mediapipe as mp

from gait_engine import extract_poses, build_time_series, compute_all_features, interpret
from gait_plots import (
    build_all_figures,
    normal_hip_reference,
    normal_knee_reference,
    normal_ankle_reference,
)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GaitVision",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default adult height — used until the user changes it in the height input on the main page.
DEFAULT_HEIGHT_CM = 170

# ──────────────────────────────────────────────
# CUSTOM CSS — single dark slate / blue medical-tool theme, app-wide.
# Replaces the prior light theme with a deep-slate + bright-blue palette
# per the manager-brief design spec.
# ──────────────────────────────────────────────
st.markdown("""
<style>
  .stApp {
    background:
      radial-gradient(circle at 8% 0%,    rgba(59,130,246,0.12), transparent 45%),
      radial-gradient(circle at 92% 4%,   rgba(6,182,212,0.08),  transparent 42%),
      radial-gradient(circle at 88% 92%,  rgba(168,85,247,0.10), transparent 45%),
      radial-gradient(circle at 4% 96%,   rgba(236,72,153,0.06), transparent 40%),
      #0F172A;
  }
  .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
  }
  html, body, [class*="st-"] {
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: #CBD5E1;
  }
  h1, h2, h3, h4, h5, h6 {
    color: #F1F5F9 !important;
    font-weight: 500;
  }
  p, span, div { color: inherit; }

  /* ===== Sidebar ===== */
  [data-testid="stSidebar"] {
    background: #1E293B;
    border-right: 1px solid #334155;
  }
  [data-testid="stSidebar"] > div { padding-top: 2rem; }
  [data-testid="stSidebar"] * { color: #CBD5E1; }
  [data-testid="stSidebar"] h1 {
    color: #3B82F6;
    font-size: 22px; font-weight: 700;
    letter-spacing: 0.5px;
    margin: 12px 0 4px 0;
  }
  [data-testid="stSidebar"] .sidebar-tagline {
    font-size: 12px; color: #94A3B8;
    margin-bottom: 24px; letter-spacing: 0.4px;
  }

  /* ===== Sidebar stepper ===== */
  .stepper {
    display: flex; flex-direction: column;
    gap: 12px; margin-top: 8px;
  }
  .stepper-item {
    display: flex; align-items: center;
    gap: 12px; padding: 8px 4px;
    transition: transform 0.2s ease;
  }
  .stepper-circle {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 600; flex-shrink: 0;
    transition: all 0.25s ease;
  }
  .stepper-name { font-size: 14px; line-height: 1.2; }
  .stepper-active   .stepper-circle {
    background: linear-gradient(135deg, #3B82F6, #2563EB);
    color: #FFFFFF;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.18),
                0 0 18px rgba(59,130,246,0.45);
  }
  .stepper-active   .stepper-name   { color: #F1F5F9; font-weight: 600; }
  .stepper-done     .stepper-circle {
    background: linear-gradient(135deg, #10B981, #059669);
    color: #FFFFFF;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.15);
  }
  .stepper-done     .stepper-name   { color: #F1F5F9; font-weight: 400; }
  .stepper-upcoming .stepper-circle {
    background: transparent;
    border: 1.5px solid #334155;
    color: #64748B;
  }
  .stepper-upcoming .stepper-name   { color: #64748B; }

  /* Per-step color accents on the active circle (overrides the base
     active gradient + glow above when the matching nth-child is active). */
  .stepper-item:nth-child(2).stepper-active .stepper-circle {
    background: linear-gradient(135deg, #06B6D4, #0891B2);
    box-shadow: 0 0 0 4px rgba(6,182,212,0.18),
                0 0 18px rgba(6,182,212,0.45);
  }
  .stepper-item:nth-child(3).stepper-active .stepper-circle {
    background: linear-gradient(135deg, #A855F7, #7E22CE);
    box-shadow: 0 0 0 4px rgba(168,85,247,0.18),
                0 0 18px rgba(168,85,247,0.45);
  }
  .stepper-item:nth-child(4).stepper-active .stepper-circle {
    background: linear-gradient(135deg, #EC4899, #BE185D);
    box-shadow: 0 0 0 4px rgba(236,72,153,0.18),
                0 0 18px rgba(236,72,153,0.45);
  }

  /* ===== Form inputs ===== */
  .stTextInput input, .stNumberInput input,
  .stDateInput input, .stTextArea textarea {
    background: #0F172A !important;
    color: #F1F5F9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
  }
  .stTextInput input:focus, .stNumberInput input:focus,
  .stDateInput input:focus, .stTextArea textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.30) !important;
    outline: none !important;
  }
  .stTextInput input::placeholder,
  .stTextArea textarea::placeholder {
    color: #64748B !important;
  }
  .stSelectbox > div > div {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
  }
  .stSelectbox > div > div > div { color: #F1F5F9 !important; }

  /* CRITICAL: number input +/- buttons MUST NOT render black */
  .stNumberInput button {
    background: #334155 !important;
    color: #CBD5E1 !important;
    border: 1px solid #475569 !important;
    border-radius: 6px !important;
  }
  .stNumberInput button:hover { background: #475569 !important; }
  .stNumberInput button:disabled {
    background: #1E293B !important;
    color: #475569 !important;
  }

  /* CRITICAL: selectbox dropdown popover must be dark */
  [data-baseweb="popover"] {
    background: #1E293B !important;
  }
  [data-baseweb="menu"] {
    background: #1E293B !important;
  }
  [data-baseweb="menu"] li {
    color: #CBD5E1 !important;
    background: #1E293B !important;
  }
  [data-baseweb="menu"] li:hover {
    background: #334155 !important;
  }

  /* ===== Field labels ===== */
  label,
  .stTextInput label, .stNumberInput label, .stSelectbox label,
  .stDateInput label, .stTextArea label, .stFileUploader label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
  }

  /* ===== Wizard step components ===== */
  .wizard-title {
    font-size: 22px; font-weight: 600;
    color: #F1F5F9;
    margin-bottom: 20px; text-align: center;
  }
  .wizard-info-strip {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px; color: #CBD5E1;
    margin: 12px 0 20px 0;
  }
  .wizard-file-info {
    background: #0F172A;
    border: 1px solid #334155;
    border-left: 3px solid #3B82F6;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px; color: #CBD5E1;
    margin: 8px 0 12px 0;
  }
  .wizard-section-heading {
    font-size: 15px; font-weight: 600;
    color: #F1F5F9;
    margin: 18px 0 8px 0;
  }
  .wizard-check-row {
    color: #CBD5E1;
    padding: 6px 0;
    font-size: 14px; line-height: 1.5;
  }
  .wizard-check-row::before {
    content: "✓ ";
    color: #10B981;
    font-weight: bold; margin-right: 4px;
  }

  /* ===== Generic app card ===== */
  .app-card {
    background: linear-gradient(180deg, #1E293B, #1A2336);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35),
                0 0 0 1px rgba(59,130,246,0.06);
    margin-bottom: 20px;
    color: #CBD5E1;
  }

  /* ===== Section title (legacy Step 4 dashboard) ===== */
  .section-title {
    position: relative;
    text-align: center;
    font-size: 18px; font-weight: 500;
    color: #F1F5F9;
    padding: 14px;
    background: linear-gradient(180deg, #1E293B, #1A2336);
    border: 1px solid #334155;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.30);
    margin: 24px 0 16px 0;
    overflow: hidden;
  }
  .section-title::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #3B82F6 0%, #A855F7 50%, #EC4899 100%);
  }

  /* ===== Hero (legacy Step 4 dashboard) ===== */
  .hero-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #3B82F6 0%, #A855F7 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent !important;
    line-height: 1.2; margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
  }
  .hero-sub {
    font-size: 1.0rem; color: #94A3B8;
    margin-bottom: 1.5rem;
  }
  .wizard-title {
    background: linear-gradient(135deg, #F1F5F9 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* ===== Legacy metric cards (Step 4 dashboard) ===== */
  .metric-card {
    background: linear-gradient(180deg, #1E293B, #1A2336);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    height: 110px;
    display: flex; flex-direction: column; justify-content: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3B82F6, #A855F7, #EC4899, transparent);
    opacity: 0;
    transition: opacity 0.25s ease;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(59,130,246,0.25),
                0 0 0 1px rgba(59,130,246,0.30);
    border-color: rgba(59,130,246,0.30);
  }
  .metric-card:hover::before { opacity: 1; }
  .metric-card .metric-label {
    font-size: 0.72rem; font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 6px;
    background: transparent; padding: 0; display: block;
  }
  .metric-card .metric-value {
    font-size: 1.85rem; font-weight: 700;
    background: linear-gradient(135deg, #3B82F6, #A855F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent;
    line-height: 1;
    padding: 0; display: block;
  }
  .metric-card .metric-unit {
    font-size: 0.75rem; color: #94A3B8; margin-top: 4px;
  }

  /* ===== Insight cards ===== */
  .obs-card {
    background: linear-gradient(180deg, #1E293B, #1A2336);
    border: 1px solid #334155;
    border-left: 4px solid #10B981;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.92rem; line-height: 1.5;
    color: #CBD5E1;
    box-shadow: 0 2px 6px rgba(0,0,0,0.20),
                inset 4px 0 0 0 rgba(16,185,129,0.18);
  }
  .sug-card {
    background: linear-gradient(180deg, #1E293B, #1A2336);
    border: 1px solid #334155;
    border-left: 4px solid #F59E0B;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.92rem; line-height: 1.5;
    color: #CBD5E1;
    box-shadow: 0 2px 6px rgba(0,0,0,0.20),
                inset 4px 0 0 0 rgba(245,158,11,0.18);
  }

  /* ===== Tabs ===== */
  .stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid #334155;
    background: transparent;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748B;
    font-size: 13px; font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 12px 4px;
    border-bottom: 3px solid transparent;
    transition: color 0.2s ease, border-color 0.2s ease;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #CBD5E1;
  }
  .stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom-color: #3B82F6 !important;
    box-shadow: 0 4px 8px -4px rgba(59,130,246,0.50);
  }

  /* ===== Buttons ===== */
  .stButton button {
    border-radius: 8px;
    font-weight: 500;
    padding: 10px 20px;
    transition: all 0.2s ease;
  }
  .stButton button[kind="primary"] {
    background: linear-gradient(135deg, #3B82F6, #2563EB);
    color: #FFFFFF;
    border: none;
    box-shadow: 0 1px 3px rgba(0,0,0,0.20);
  }
  .stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    box-shadow: 0 4px 14px rgba(59,130,246,0.40),
                0 0 0 1px rgba(59,130,246,0.20);
    transform: translateY(-1px);
  }
  .stButton button[kind="secondary"] {
    background: transparent;
    color: #CBD5E1;
    border: 1px solid #334155;
  }
  .stButton button[kind="secondary"]:hover {
    background: #334155;
    border-color: #475569;
  }
  .stButton button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
  .stDownloadButton button {
    background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.20) !important;
    transition: all 0.2s ease !important;
  }
  .stDownloadButton button:hover {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.40) !important;
    transform: translateY(-1px);
  }

  /* ===== File uploader ===== */
  [data-testid="stFileUploader"] section {
    background: #0F172A !important;
    border: 1px dashed #334155 !important;
    border-radius: 12px !important;
  }
  [data-testid="stFileUploader"] section button {
    background: #334155 !important;
    color: #CBD5E1 !important;
    border: 1px solid #475569 !important;
  }

  /* ===== Plotly chart container (kept ready for any future plotly use) ===== */
  [data-testid="stPlotlyChart"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.30);
  }

  /* ===== Expanders ===== */
  .streamlit-expanderHeader {
    background: #0F172A !important;
    color: #F1F5F9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
  }
  [data-testid="stExpander"] details {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
  }
  [data-testid="stExpander"] summary {
    color: #F1F5F9 !important;
  }

  /* ===== Status / Alert boxes ===== */
  [data-testid="stAlert"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #CBD5E1 !important;
  }
  [data-testid="stStatus"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #CBD5E1 !important;
  }

  /* ===== Streamlit chrome hiding ===== */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
def _init_session_state():
    """Initialise wizard state on first render. Idempotent across reruns.

    Keys:
      step          — int, 1..4. Active wizard step.
      patient       — dict, holds form values from Step 1
                      (name, patient_id, age, gender, height_cm,
                      weight_kg, assessment_date, clinician, notes).
      video_file    — Streamlit UploadedFile | None. Set in Step 3.
      features      — dict | None. Output of compute_all_features.
      insights      — dict | None. Output of interpret().
      fps           — float | None. Captured at extract_poses time.
      total_frames  — int | None. Captured at extract_poses time.
    """
    defaults = {
        "step": 1,
        "patient": {},
        "video_file": None,
        "features": None,
        "insights": None,
        "fps": None,
        "total_frames": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ──────────────────────────────────────────────
# CACHED MODEL LOADER
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pose_model_options():
    model_path = "pose_landmarker_lite.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _metric_card(label: str, value: str, unit: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>
    """


def _render_metric_grid(m: dict) -> None:
    """Render the 8-card 4×2 metric grid for one metric set (Total or Clean)."""
    cadence    = m["cadence"]
    sym        = m["symmetry"]
    knee_peak  = m["knee_angles"].get("overall_peak", 0.0)
    stride_cv  = m["stride_cv"]                              # already a percentage
    step_count = m["step_data"]["total_steps"]
    step_len   = m["step_length"]["mean"]
    step_unit  = m["step_length"].get("unit", "m")
    torso      = m["torso_lean"]["mean"]
    mean_st    = m["step_timing"]["mean_step_time"]

    row1 = st.columns(4)
    row2 = st.columns(4)
    cards = [
        ("Step Count",     str(step_count),                         "strikes"),
        ("Cadence",        f"{cadence}",                            "steps / min"),
        ("Symmetry",       f"{sym:.1%}",                            "L/R rhythm"),
        ("Knee Peak",      f"{knee_peak:.1f}°",                     "swing flexion"),
        ("Stride CV",      f"{stride_cv:.2f}%",                     "lower = better"),
        ("Step Length",    f"{step_len:.3f}",                       step_unit),
        ("Torso Lean",     f"{torso:.1f}°",                         "+ fwd / - back"),
        ("Step Time",      f"{mean_st:.3f}s" if mean_st > 0 else "N/A", "avg interval"),
    ]
    for i, (label, val, unit) in enumerate(cards):
        col = row1[i] if i < 4 else row2[i - 4]
        with col:
            st.markdown(_metric_card(label, val, unit), unsafe_allow_html=True)


def _render_metrics(features: dict) -> None:
    """
    Render TWO metric sets side-by-side:
      • Total  — every frame in the video (informational; includes turns/accel/decel).
      • Clean  — steady-state frames only (drives ALL observations & suggestions).
    """
    total = features.get("total_metrics", features)
    clean = features.get("clean_metrics", features)
    direction   = features.get("direction", "Unknown")
    num_passes  = features.get("num_passes", 0)
    total_dur   = features.get("duration_sec", 0)
    clean_dur   = features.get("steady_state_duration_s", 0)

    # ── Total ────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Total Metrics '
        '<span style="font-size:0.75rem;color:#6E7681;font-weight:400;">'
        '— entire video (informational; includes turns / accel / decel)'
        '</span></div>',
        unsafe_allow_html=True,
    )
    _render_metric_grid(total)
    st.markdown(
        f"<p style='color:#6E7681;font-size:0.78rem;margin-top:0.4rem;'>"
        f"Window: {total_dur:.1f}s · all frames included</p>",
        unsafe_allow_html=True,
    )

    # ── Clean ────────────────────────────────────────────
    st.markdown(
        '<div class="section-title" style="margin-top:1.5rem">'
        '✨ Clean Metrics '
        '<span style="font-size:0.75rem;color:#3FB950;font-weight:400;">'
        '— steady-state only · used for all observations & suggestions'
        '</span></div>',
        unsafe_allow_html=True,
    )
    _render_metric_grid(clean)
    st.markdown(
        f"<p style='color:#6E7681;font-size:0.78rem;margin-top:0.4rem;'>"
        f"Window: {clean_dur:.1f}s steady-state across "
        f"<b style='color:#C9D1D9'>{num_passes}</b> validated pass"
        f"{'es' if num_passes != 1 else ''} "
        f"({(clean_dur / total_dur * 100) if total_dur > 0 else 0:.0f}% of video)</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<p style='color:#6E7681;font-size:0.82rem;margin-top:0.8rem;'>"
        f" Walking direction: <b style='color:#C9D1D9'>{direction}</b></p>",
        unsafe_allow_html=True,
    )


_REFERENCE_FUNCS = {
    "HIP":   normal_hip_reference,
    "KNEE":  normal_knee_reference,
    "ANKLE": normal_ankle_reference,
}


def _phase_label(pct: int) -> str:
    """Map a 0-100 cycle percent to its standard gait-phase name."""
    if pct < 15:  return "early stance"
    if pct < 35:  return "mid-stance"
    if pct < 50:  return "late stance"
    if pct < 65:  return "pre-swing"
    if pct < 80:  return "mid-swing"
    return "late swing"


def _plateau_center(curve: np.ndarray, threshold_frac: float = 0.05) -> int:
    """Return the cycle-percent at the center of a curve's peak plateau.

    Defined as the circular mean of all sample indices where the curve
    sits within `threshold_frac × ROM` of its maximum. More stable than
    `np.nanargmax` for broad peaks where a single noisy sample can shift
    the argmax by 10%+ of the cycle. The circular mean correctly handles
    plateaus that wrap across the 0/100 boundary (typical for hip).

    Falls back to nanargmax if the curve is empty or flat.
    """
    valid = ~np.isnan(curve)
    if not valid.any():
        return 0
    max_val = float(np.nanmax(curve))
    min_val = float(np.nanmin(curve))
    rng = max_val - min_val
    if rng <= 0:
        return int(np.nanargmax(curve))

    threshold = max_val - threshold_frac * rng
    plateau_idx = np.where((curve >= threshold) & valid)[0]
    if len(plateau_idx) == 0:
        return int(np.nanargmax(curve))

    # Circular mean over [0, 100]: index 100 == index 0 (same cycle event,
    # both at heel strike), so map index → angle as i * 2π / 100.
    angles = plateau_idx * 2.0 * np.pi / 100.0
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return int(round(mean_angle * 100.0 / (2.0 * np.pi))) % 100


def _render_cycle_observations(features: dict) -> None:
    """Auto-generated kinematic observations rendered below the cycle figure.

    For each joint with valid mean curves on both legs, computes:
      - per-leg ROM (max - min of the mean curve)
      - per-leg peak timing (plateau center, with phase label)
      - per-leg amplitude ratio vs the reference ROM (max - min of the
        reference mean curve from gait_plots.normal_*_reference())

    Renders one consolidated 'Clinical observations' section with a
    sub-header per joint. Bullets present both legs together so
    asymmetry is visible without a separate row.
    """
    gc = features.get("gait_cycle_curves") or {}
    if not gc:
        return

    blocks = []
    for joint_label, key in (("HIP", "hip"), ("KNEE", "knee"), ("ANKLE", "ankle")):
        joint = gc.get(key, {}) or {}
        L = joint.get("left",  {}) or {}
        R = joint.get("right", {}) or {}
        mL = L.get("mean")
        mR = R.get("mean")
        if mL is None or mR is None:
            continue

        ref_mean, _ = _REFERENCE_FUNCS[joint_label]()
        ref_rom  = float(np.max(ref_mean) - np.min(ref_mean))
        ref_peak = _plateau_center(ref_mean)

        L_rom   = float(np.nanmax(mL) - np.nanmin(mL))
        R_rom   = float(np.nanmax(mR) - np.nanmin(mR))
        L_peak  = _plateau_center(mL)
        R_peak  = _plateau_center(mR)
        L_ratio = (L_rom / ref_rom * 100.0) if ref_rom > 0 else float("nan")
        R_ratio = (R_rom / ref_rom * 100.0) if ref_rom > 0 else float("nan")

        blocks.append(
            f"**{joint_label}**\n\n"
            f"- ROM: L {L_rom:.0f}°, R {R_rom:.0f}° "
            f"(reference {ref_rom:.0f}°; L {L_ratio:.0f}%, R {R_ratio:.0f}% of normal)\n"
            f"- Peak timing: L at {L_peak}% ({_phase_label(L_peak)}), "
            f"R at {R_peak}% ({_phase_label(R_peak)}) "
            f"— reference peak at {ref_peak}%"
        )

    if not blocks:
        return

    st.markdown("---")
    st.markdown("**Clinical observations**")
    st.markdown("\n\n".join(blocks))
    st.caption(
        "Auto-generated kinematic observations. See expanders below for "
        "interpretation guidance."
    )


def _render_cycle_explanations() -> None:
    """Caption + three per-joint expandable sections explaining the cycle chart.
    Renders below the gait-cycle figure. Dual-audience copy (clinician + layperson)."""
    st.caption(
        "The grey band shows the typical healthy-walker range AS MEASURED BY "
        "THIS PIPELINE — not the 3D motion-capture clinical norm. Your "
        "measurements are the blue (left leg) and red (right leg) lines."
    )

    with st.expander("Understanding the HIP graph"):
        st.markdown("""
**What you're looking at:** How your thigh swings forward and backward during one complete step, starting the moment your heel hits the ground.

**The grey band = healthy-walker range, measured by this 2D-video pipeline.** A healthy adult's hip typically shows about 28° of measurable motion in this view — swung forward (around +21°) when the heel lands, swinging backward (around −7°) as the body rolls over the planted foot, then forward again to set up the next step. The true 3D range is ~40°; the smaller measured value reflects projection loss inherent to single-camera 2D pose estimation.

**The blue and red lines = your left and right hip.** Your lines should stay roughly within the grey band and follow its shape — a smooth wave pattern.

**Interpreting your curve:**
- **Range of motion narrower than ~17°** indicates the hip is moving through a smaller envelope than the typical healthy-walker range for this pipeline.
- **Curve minimum stays above 0°** indicates the hip does not pass into extension during stance — a kinematic pattern associated with reduced effective stride length.
- **Vertical separation between blue and red** indicates left and right hips are operating in different parts of the range.
- **Horizontal shift relative to the reference band** indicates the timing of peak flexion/extension does not align with the typical 0%/50% pattern.

*These observations are descriptive only. Clinical interpretation should be made by a qualified professional considering the full clinical picture, medical history, and other assessments.*
""")

    with st.expander("Understanding the KNEE graph"):
        st.markdown("""
**What you're looking at:** How much your knee bends during one complete step. 0° means a perfectly straight leg; higher numbers mean more bending.

**The grey band = healthy-walker range, measured by this 2D-video pipeline.** A healthy knee shows two distinct peaks per step:

1. **First peak (~9° measured here, ~18° in 3D mocap) right after heel strike:** the knee bends slightly to absorb landing impact, like a shock absorber.
2. **Second, larger peak (~33° measured here, ~65° in 3D mocap) during mid-swing:** the knee bends to lift your foot off the ground so your toes don't drag as the leg swings forward.

The measured peaks are smaller than the true 3D values because 2D side-view captures only the projected component of knee motion — knee is the most underestimated joint in single-camera pose estimation.

**The blue and red lines = your left and right knee.** You should see two clear humps on each line, matching the shape of the grey band.

**Interpreting your curve:**
- **Absent or reduced loading peak (the small bump at 0–20%)** indicates limited knee flexion during weight acceptance — sometimes described in the literature as a "stiff-knee" loading pattern.
- **Reduced swing peak (the larger hump near ~70%, below ~25°)** indicates the knee is bending less than typical to lift the foot during the swing phase.
- **Asymmetry between blue and red** indicates left and right knees are following different flexion profiles across the cycle.
- **A wider shaded band around a curve** indicates greater stride-to-stride variability in that knee's motion.

*These observations are descriptive only. Clinical interpretation should be made by a qualified professional considering the full clinical picture, medical history, and other assessments.*
""")

    with st.expander("Understanding the ANKLE graph"):
        st.markdown("""
**What you're looking at:** How your ankle rotates up and down during one complete step.
- **Positive values (dorsiflexion):** your toes are pulled toward your shin — this happens when your foot is flat on the ground and your body rolls forward over it.
- **Negative values (plantarflexion):** your toes are pointed down — this happens when you push off the ground to propel yourself forward.

**The grey band = healthy-walker range, measured by this 2D-video pipeline.** A healthy ankle typically shows (in this view):
- A small negative dip (~−3°) right after heel strike as the forefoot slaps down.
- A gradual rise to about +7° as you roll forward over the planted foot.
- A sharp plunge to about −12° at push-off (around 60% of the step) — this is where most of your walking power comes from.
- A return to near 0° during swing so your toes clear the ground.

True 3D push-off plantarflexion is closer to −18°; the measured value is smaller because of 2D projection loss.

**The blue and red lines = your left and right ankle.**

**Interpreting your curve:**
- **Reduced plantarflexion at push-off (minimum stays above ~−7°)** indicates the ankle is generating less downward pointing during the propulsion phase than the typical healthy-walker range for this pipeline.
- **Ankle does not return to neutral during swing (stays negative past ~70%)** indicates limited dorsiflexion recovery as the foot is brought forward.
- **Compressed overall range** indicates a smaller-than-typical sweep between dorsiflexion and plantarflexion across the cycle.
- **Asymmetry between blue and red** indicates left and right ankles are following different profiles across the cycle.

*These observations are descriptive only. Clinical interpretation should be made by a qualified professional considering the full clinical picture, medical history, and other assessments.*
""")


def _render_graphs(features: dict) -> None:
    """Render plot tabs (6 existing + 1 new gait-cycle tab)."""
    st.markdown('<div class="section-title"> Analysis Graphs</div>', unsafe_allow_html=True)

    figs = build_all_figures(features)

    tab_labels = [
        "Knee Angles",
        "Heel Position",
        "Step Length",
        "Timing",
        "Torso Lean",
        "Ankle Trajectory",
        "Gait Cycle",
    ]
    tabs = st.tabs(tab_labels)

    tab_keys = ["knee", "heel", "step", "timing", "torso", "ankle", "cycle"]

    # Single-line info captions for the six pre-existing tabs.
    # The gait-cycle tab shows the matplotlib figure + cycle-quality
    # captions only; clinical observations and the HIP/KNEE/ANKLE
    # expanders are rendered later by _render_step_4_legacy_dashboard
    # so both chart sections stay back-to-back.
    descriptions = {
        "knee": "**Understanding this graph:** This graph tracks the flexion (bending) angle of your knees over time. 0° represents full extension (a straight leg), while higher values indicate knee bending. The grey shaded band represents the typical range of motion for normal walking as captured by this 2D-video pipeline (0° to ~35°; the true 3D range is 0° to ~65°, but single-camera 2D pose estimation underestimates knee flexion). If your leg cannot reach ~0°, it may indicate a limp or extension deficit.",
        "heel": "**Understanding this graph:** This plots the raw horizontal forward progression of your heels. The grey dotted line represents an idealized smoothed envelope of your overall motion to help highlight any sudden jitter, dragging, or instability.",
        "step": "**Understanding this graph:** Step length is normalized by dividing the physical ground distance of a step by your estimated body height (shoulder to heel mapping). This scaling prevents taller subjects from artificially looking like they overstride. A normal healthy step length is roughly 41.5% of height. The grey band (0.35 – 0.55) represents a standard healthy ratio. Values dropping below indicate short, shuffling steps; values above suggest over-striding.",
        "timing": "**Understanding this graph:** This chart displays the elapsed time between consecutive steps. Consistent timing is a key indicator of gait balance. The grey band highlights an acceptable variation margin (±10%) around your personal average. Spikes outside this box suggest irregular pacing or limping.",
        "torso": "**Understanding this graph:** The torso lean angle is measured against a perfect vertical axis (0°). The grey shaded zone (±7°) highlights an acceptable physiological comfort zone for upright walking. Consistent numbers outside this layer highlight a tendency to lean heavily forward (+ values) or backward (- values).",
        "ankle": "**Understanding this graph:** This shows the ankles crossing each other as you walk. (Note: position is normalized to screen width from 0 to 1). The dotted grey curve is an idealized harmonic sine wave model, representing the expected smooth, pendulum-like motion of a perfect walking swing phase. Sharp deviations imply jerky movements.",
    }

    for tab, key in zip(tabs, tab_keys):
        with tab:
            if key == "cycle":
                # K<3-on-either-knee-leg fallback per spec
                gc = features.get("gait_cycle_curves") or {}
                knee_data = gc.get("knee", {})
                KL = knee_data.get("left",  {}).get("K", 0)
                KR = knee_data.get("right", {}).get("K", 0)
                fig = figs.get("cycle")
                if fig is not None and min(KL, KR) >= 3:
                    st.pyplot(fig, use_container_width=True)
                    # Heel-strike amplitude-filter rejection counts
                    sr = features.get("strike_rejection") or {}
                    L_meta = sr.get("left",  {}) or {}
                    R_meta = sr.get("right", {}) or {}
                    L_acc, L_rej = L_meta.get("accepted", 0), L_meta.get("rejected", 0)
                    R_acc, R_rej = R_meta.get("accepted", 0), R_meta.get("rejected", 0)
                    if (L_acc + L_rej + R_acc + R_rej) > 0:
                        st.caption(
                            f" Cycle detection: {R_acc} right strides accepted, "
                            f"{R_rej} rejected due to low heel-clearance amplitude "
                            f"(likely foot-drag events). "
                            f"{L_acc} left strides accepted, {L_rej} rejected."
                        )
                    # Cycle duration-filter rejection counts
                    cdf = features.get("cycle_duration_filter") or {}
                    L_cdf = cdf.get("left",  {}) or {}
                    R_cdf = cdf.get("right", {}) or {}
                    L_kept  = L_cdf.get("kept", 0)
                    L_long  = L_cdf.get("rejected_long", 0)
                    L_short = L_cdf.get("rejected_short", 0)
                    R_kept  = R_cdf.get("kept", 0)
                    R_long  = R_cdf.get("rejected_long", 0)
                    R_short = R_cdf.get("rejected_short", 0)
                    if (L_kept + L_long + L_short + R_kept + R_long + R_short) > 0:
                        st.caption(
                            f" Cycle quality: {L_kept} left cycles kept "
                            f"({L_long} rejected as too long, {L_short} as too short); "
                            f"{R_kept} right cycles kept "
                            f"({R_long} rejected as too long, {R_short} as too short)."
                        )
                    # NOTE: clinical-observations panel and HIP/KNEE/ANKLE
                    # expanders used to render here. They were moved out so
                    # both chart sections (this matplotlib figure + the
                    # interactive plotly tabs below) appear back-to-back
                    # before any interpretation text. They now render in
                    # _render_step_4_legacy_dashboard after the interactive
                    # tabs section.
                else:
                    st.info(
                        "Not enough valid gait cycles detected to generate "
                        "cycle-normalized curves. Need at least 3 clean strides per leg."
                    )
            else:
                fig = figs.get(key)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                    st.info(descriptions[key], icon="ℹ️")


def _render_insights(insights: dict) -> None:
    """Render observations and suggestions side by side."""
    st.markdown('<div class="section-title"> Insights & Suggestions</div>', unsafe_allow_html=True)

    col_obs, col_sug = st.columns(2)

    with col_obs:
        st.markdown("**Observations**")
        for obs in insights["observations"]:
            st.markdown(f'<div class="obs-card">{obs}</div>', unsafe_allow_html=True)

    with col_sug:
        st.markdown("**Suggestions**")
        for sug in insights["suggestions"]:
            st.markdown(f'<div class="sug-card">{sug}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# WIZARD — sidebar stepper + four step renderers + legacy-step-4 wrap
# ══════════════════════════════════════════════════════════════════
WIZARD_STEPS = [
    (1, "Patient Details"),
    (2, "Instructions"),
    (3, "Upload & Analyze"),
    (4, "Results"),
]


def _render_sidebar_stepper() -> None:
    """Vertical step indicator in the sidebar; reflects st.session_state['step']."""
    with st.sidebar:
        st.markdown("# GaitVision")
        st.markdown(
            '<div class="sidebar-tagline">CLINICAL GAIT ANALYSIS</div>',
            unsafe_allow_html=True,
        )
        current = st.session_state["step"]
        items = []
        for num, name in WIZARD_STEPS:
            if num < current:
                cls, marker = "stepper-done", "✓"
            elif num == current:
                cls, marker = "stepper-active", str(num)
            else:
                cls, marker = "stepper-upcoming", str(num)
            items.append(
                f'<div class="stepper-item {cls}">'
                f'<div class="stepper-circle">{marker}</div>'
                f'<div class="stepper-name">{name}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="stepper">{"".join(items)}</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# STEP 1 — PATIENT DETAILS
# ──────────────────────────────────────────────
GENDER_OPTIONS = ["", "Female", "Male", "Other", "Prefer not to say"]


def _render_step_1() -> None:
    """Patient information form. Disabled-only validation: Next greys out
    until all required fields (Name, Age, Gender, Height) are valid."""
    p = st.session_state["patient"]

    st.markdown(
        '<div class="wizard-title">Patient Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input(
            "Full Name *",
            value=p.get("name", ""),
            placeholder="Jane Doe",
        )
        age = st.number_input(
            "Age *",
            min_value=1, max_value=120, step=1,
            value=int(p.get("age") or 30),
        )
        height_cm = st.number_input(
            "Height (cm) *",
            min_value=50, max_value=250, step=1,
            value=int(p.get("height_cm") or DEFAULT_HEIGHT_CM),
            help="Used for pixel-to-meter calibration of step length.",
        )
        assess_date = st.date_input(
            "Assessment Date",
            value=p.get("assessment_date") or date.today(),
        )
    with col2:
        patient_id = st.text_input(
            "Patient ID (optional)",
            value=p.get("patient_id", ""),
            placeholder="auto-generated if blank",
        )
        gender_default_idx = (
            GENDER_OPTIONS.index(p["gender"])
            if p.get("gender") in GENDER_OPTIONS
            else 0
        )
        gender = st.selectbox(
            "Gender *",
            GENDER_OPTIONS,
            index=gender_default_idx,
        )
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=10.0, max_value=300.0, step=0.1,
            value=float(p.get("weight_kg") or 70.0),
        )
        clinician = st.text_input(
            "Referring Clinician",
            value=p.get("clinician", ""),
        )

    notes = st.text_area(
        "Clinical Notes (optional)",
        value=p.get("notes", ""),
        height=100,
    )

    # Disabled-only validation
    required_ok = (
        bool(name.strip())
        and bool(gender)
        and 1 <= int(age) <= 120
        and 50 <= int(height_cm) <= 250
    )

    _, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Next →", disabled=not required_ok, type="primary",
                     use_container_width=True):
            new_id = patient_id.strip() or (
                f"GV-{date.today().strftime('%Y%m%d')}-"
                f"{uuid.uuid4().hex[:6].upper()}"
            )
            st.session_state["patient"] = {
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
            st.session_state["step"] = 2
            st.rerun()


# ──────────────────────────────────────────────
# STEP 2 — INSTRUCTIONS
# ──────────────────────────────────────────────
INSTRUCTION_BLOCKS = [
    ("Recording setup", [
        "Camera perpendicular to walking direction (true side view)",
        "Walking path at least 4 to 5 meters, straight, unobstructed",
        "Full body visible for entire walk",
        "Good even lighting — no backlighting or harsh shadows",
        "Subject wears fitted clothing (loose clothing hides joints)",
    ]),
    ("Video requirements", [
        "30 fps or higher",
        "720p minimum resolution",
        "3 to 5 back-and-forth passes recommended for stride averaging",
    ]),
    ("For accurate analysis", [
        "Keep camera steady — use a tripod if possible",
        "Subject walks at their natural self-selected pace",
        "No carrying objects or using a phone during recording",
    ]),
]


def _render_step_2() -> None:
    """Read-only checklist screen. Back / Next navigation only."""
    st.markdown(
        '<div class="wizard-title">Before You Record</div>',
        unsafe_allow_html=True,
    )

    for heading, items in INSTRUCTION_BLOCKS:
        st.markdown(
            f'<div class="wizard-section-heading">{heading}</div>',
            unsafe_allow_html=True,
        )
        for item in items:
            st.markdown(
                f'<div class="wizard-check-row">{item}</div>',
                unsafe_allow_html=True,
            )

    col_back, _, col_next = st.columns([1, 4, 1])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 1
            st.rerun()
    with col_next:
        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state["step"] = 3
            st.rerun()


# ──────────────────────────────────────────────
# STEP 3 — UPLOAD & ANALYZE
# ──────────────────────────────────────────────
def _run_analysis() -> bool:
    """Run the full pipeline on the uploaded file and store results in
    session state. Returns True on success, False on failure (the calling
    site uses this to decide whether to advance to Step 4)."""
    video = st.session_state["video_file"]
    if video is None:
        st.error("No video file in session. Re-upload and try again.")
        return False
    file_bytes = video.getvalue()
    file_name  = video.name or "video.mp4"
    user_height_cm = st.session_state["patient"].get("height_cm", DEFAULT_HEIGHT_CM)

    suffix = os.path.splitext(file_name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with st.status("Analyzing gait …", expanded=True) as status:
            st.write("Loading pose model …")
            pose_options = load_pose_model_options()

            st.write("Extracting poses from video …")
            raw, fps, total_frames = extract_poses(tmp_path, pose_options)

            st.write("Detecting walking passes …")
            ts = build_time_series(raw)

            st.write(
                "Filtering turning and acceleration phases · "
                "detecting heel-strike events · "
                "computing joint angles …"
            )

            st.write("Building gait-cycle curves and metrics …")
            features = compute_all_features(
                ts, fps, total_frames, user_height_cm=user_height_cm,
            )
            features["_ts"] = ts

            st.write("Generating insights …")
            insights = interpret(features)

            st.session_state["features"]    = features
            st.session_state["insights"]    = insights
            st.session_state["fps"]         = fps
            st.session_state["total_frames"] = total_frames

            status.update(label="Analysis complete.", state="complete")
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


def _render_step_3() -> None:
    """File upload, preview, and analyze trigger."""
    p = st.session_state["patient"]

    st.markdown(
        '<div class="wizard-title">Upload Walking Video</div>',
        unsafe_allow_html=True,
    )

    info_strip = (
        f"Subject: {p.get('name', '—')}  ·  "
        f"Age: {p.get('age', '—')}  ·  "
        f"Height: {p.get('height_cm', '—')} cm"
    )
    st.markdown(
        f'<div class="wizard-info-strip">{info_strip}</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drag and drop a walking video, or click to browse",
        type=["mp4", "mov", "avi", "mkv"],
    )

    if uploaded is not None:
        size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        st.markdown(
            f'<div class="wizard-file-info">'
            f'📁 <b>{uploaded.name}</b>  ·  {size_mb:.2f} MB'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.session_state["video_file"] = uploaded

        with st.expander("Preview", expanded=False):
            st.video(uploaded)

    file_ready = st.session_state.get("video_file") is not None
    if st.button(
        "Analyze Gait",
        disabled=not file_ready,
        type="primary",
        use_container_width=True,
    ):
        if _run_analysis():
            st.session_state["step"] = 4
            st.rerun()

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state["step"] = 2
            st.rerun()


# ──────────────────────────────────────────────
# PDF REPORT BUILDER  (used by the Download PDF button in Step 4)
# ──────────────────────────────────────────────
def _build_pdf_bytes(features: dict, patient: dict) -> bytes:
    """Return a 2-page PDF: cover page with patient details + the
    gait-cycle figure on the next page. Uses matplotlib's PdfPages so
    no extra dependency is required.

    The cover page is rendered on a white background so it prints
    cleanly; the gait-cycle figure inherits its on-screen styling.
    """
    from io import BytesIO
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # ── Page 1: cover ──────────────────────────────────────────
        cover, ax = plt.subplots(figsize=(8.5, 11), facecolor="white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.text(0.5, 0.94, "Gait Analysis Report",
                ha="center", fontsize=26, fontweight="bold", color="#1A1A1A")
        ax.text(0.5, 0.89, "Generated by GaitVision",
                ha="center", fontsize=12, color="#6B7280")

        rows = [
            ("Name",            patient.get("name", "—")),
            ("Patient ID",      patient.get("patient_id", "—")),
            ("Age",             str(patient.get("age", "—"))),
            ("Gender",          patient.get("gender", "—")),
            ("Height",          f"{patient.get('height_cm', '—')} cm"),
            ("Weight",          f"{patient.get('weight_kg', '—')} kg"),
            ("Assessment Date", str(patient.get("assessment_date", "—"))),
            ("Clinician",       patient.get("clinician", "—") or "—"),
        ]
        y = 0.78
        for label, value in rows:
            ax.text(0.18, y, f"{label}:",
                    fontsize=12, fontweight="bold", color="#374151")
            ax.text(0.42, y, str(value),
                    fontsize=12, color="#1A1A1A")
            y -= 0.045

        notes = (patient.get("notes", "") or "").strip()
        if notes:
            ax.text(0.18, y - 0.04, "Clinical Notes",
                    fontsize=12, fontweight="bold", color="#374151")
            wrapped = notes if len(notes) <= 600 else notes[:600] + "…"
            ax.text(0.18, y - 0.08, wrapped,
                    fontsize=10, color="#1A1A1A", wrap=True,
                    verticalalignment="top")

        ax.text(0.5, 0.05,
                "Descriptive output from a 2D pose-estimation pipeline. "
                "Clinical interpretation belongs to a qualified professional.",
                ha="center", fontsize=8, color="#6B7280", style="italic",
                wrap=True)

        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        # ── Page 2: gait-cycle figure (only if K is sufficient) ────
        gc = features.get("gait_cycle_curves") or {}
        knee = gc.get("knee", {})
        KL = knee.get("left",  {}).get("K", 0)
        KR = knee.get("right", {}).get("K", 0)
        if min(KL, KR) >= 3:
            from gait_plots import build_all_figures
            figs = build_all_figures(features)
            cycle_fig = figs.get("cycle")
            if cycle_fig is not None:
                pdf.savefig(cycle_fig, bbox_inches="tight",
                            facecolor=cycle_fig.get_facecolor())
                plt.close(cycle_fig)

    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════
# INTERACTIVE PER-JOINT TIME-SERIES (plotly, dark-theme styled)
# ══════════════════════════════════════════════════════════════════
def build_joint_timeseries(time_s, left, right, joint_name,
                           left_color, right_color, y_label,
                           height=450):
    """Plotly scatter+line chart of one joint angle over time, two legs.

    Styled for the slate-dark UI: card background, slate gridlines,
    light text. Trace colors come from the caller (per the spec's
    per-joint color codes).
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_s, y=left,
        mode="lines+markers",
        name=f"Left {joint_name}",
        line=dict(color=left_color, width=1.2),
        marker=dict(size=4, color=left_color,
                    line=dict(width=0.5, color="#1E293B")),
        hovertemplate=("Time: %{x:.2f}s<br>"
                       f"Left {joint_name}: %{{y:.2f}}°"
                       "<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=time_s, y=right,
        mode="lines+markers",
        name=f"Right {joint_name}",
        line=dict(color=right_color, width=1.2),
        marker=dict(size=4, color=right_color,
                    line=dict(width=0.5, color="#1E293B")),
        hovertemplate=("Time: %{x:.2f}s<br>"
                       f"Right {joint_name}: %{{y:.2f}}°"
                       "<extra></extra>"),
    ))
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Time (seconds)", font=dict(color="#CBD5E1")),
            gridcolor="#334155",
            showline=True, linecolor="#475569",
            zeroline=False,
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(color="#CBD5E1")),
            gridcolor="#334155",
            showline=True, linecolor="#475569",
            zeroline=False,
            tickfont=dict(color="#94A3B8"),
        ),
        plot_bgcolor="#1E293B",
        paper_bgcolor="#1E293B",
        hovermode="closest",
        legend=dict(orientation="h", y=1.08, x=0.5,
                    xanchor="center",
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    font=dict(color="#CBD5E1")),
        margin=dict(l=50, r=30, t=40, b=50),
        height=height,
        font=dict(color="#CBD5E1"),
    )
    return fig


# Per-joint config: (tab_label, detail_title, card_title, joint_name,
#                    left_color, right_color, y_label, features_key)
_JOINT_TABS = [
    ("LEG ANGLES (NORM)",
     "Detailed Normalized Leg Angle Analysis",
     "Leg Angle Analysis (Normalized)",
     "Leg",   "#9C27B0", "#FF6D00", "Leg Angle (degrees)",
     "leg_angles"),
    ("KNEE FLEXION (NORM)",
     "Detailed Normalized Knee Flexion Analysis",
     "Knee Flexion Analysis (Normalized)",
     "Knee",  "#1976D2", "#FF6D00", "Knee Flexion (degrees)",
     "knee_angles"),
    ("HIP FLEXION (NORM)",
     "Detailed Normalized Hip Flexion Analysis",
     "Hip Flexion Analysis (Normalized)",
     "Hip",   "#388E3C", "#D32F2F", "Hip Flexion (degrees)",
     "hip_angles"),
    ("ANKLE DEFLECTION (NORM)",
     "Detailed Normalized Ankle Deflection Analysis",
     "Ankle Deflection Analysis (Normalized)",
     "Ankle", "#78909C", "#8D6E63", "Ankle Deflection (degrees)",
     "ankle_angles"),
]

_INTERACTIVE_HINT = (
    '<div style="color:#94A3B8; font-size:12px; margin-top:8px; '
    'font-style:italic;">'
    '💡 <b>Interactive Features:</b> Mouse wheel to zoom · Drag to pan · '
    'Hover for detailed values · Click legend items to toggle visibility · '
    'Use the toolbar (top right) for zoom in/out/reset'
    '</div>'
)


def _render_interactive_joint_tabs(features: dict) -> None:
    """Five-tab section: Overview (2x2 grid) + per-joint detail tabs.

    Reads per-frame angle arrays from features (added in compute_all_features
    as hip_angles / knee_angles / ankle_angles / leg_angles + time_s).
    """
    if not features.get("knee_angles"):
        return

    st.markdown(
        '<div class="section-title">Interactive Joint Angle Analysis</div>',
        unsafe_allow_html=True,
    )

    time_s = features.get("time_s")
    if time_s is None:
        n = len(features["knee_angles"]["left"])
        fps = features.get("fps", 30.0) or 30.0
        time_s = np.arange(n) / fps

    tab_labels = ["OVERVIEW"] + [cfg[0] for cfg in _JOINT_TABS]
    tabs = st.tabs(tab_labels)

    # ---- OVERVIEW: 2x2 grid of small charts, no modebar ----
    with tabs[0]:
        st.markdown(
            '<h3 style="text-align:center; margin:16px 0; color:#F1F5F9;">'
            'Normalized Gait Analysis Overview</h3>',
            unsafe_allow_html=True,
        )
        # First row: leg + knee
        col1, col2 = st.columns(2)
        for col, cfg in zip((col1, col2), _JOINT_TABS[:2]):
            (_label, _detail_title, card_title, joint, lcol, rcol,
             y_label, key) = cfg
            data = features.get(key, {}) or {}
            left, right = data.get("left"), data.get("right")
            if left is None or right is None:
                continue
            with col:
                st.markdown(
                    f'<div style="font-size:14px; font-weight:500; '
                    f'color:#CBD5E1; margin-bottom:6px;">{card_title}</div>',
                    unsafe_allow_html=True,
                )
                fig = build_joint_timeseries(
                    time_s, left, right, joint, lcol, rcol, y_label,
                    height=340,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
        # Second row: hip + ankle
        col3, col4 = st.columns(2)
        for col, cfg in zip((col3, col4), _JOINT_TABS[2:]):
            (_label, _detail_title, card_title, joint, lcol, rcol,
             y_label, key) = cfg
            data = features.get(key, {}) or {}
            left, right = data.get("left"), data.get("right")
            if left is None or right is None:
                continue
            with col:
                st.markdown(
                    f'<div style="font-size:14px; font-weight:500; '
                    f'color:#CBD5E1; margin-bottom:6px;">{card_title}</div>',
                    unsafe_allow_html=True,
                )
                fig = build_joint_timeseries(
                    time_s, left, right, joint, lcol, rcol, y_label,
                    height=340,
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

    # ---- Detail tabs: one big chart each, with modebar + hint ----
    for tab, cfg in zip(tabs[1:], _JOINT_TABS):
        (_label, detail_title, card_title, joint, lcol, rcol,
         y_label, key) = cfg
        data = features.get(key, {}) or {}
        left, right = data.get("left"), data.get("right")
        with tab:
            st.markdown(
                f'<h3 style="text-align:center; margin:16px 0; '
                f'color:#F1F5F9;">{detail_title}</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:16px; font-weight:500; '
                f'color:#F1F5F9; margin-bottom:8px;">{card_title}</div>',
                unsafe_allow_html=True,
            )
            if left is None or right is None:
                st.info(
                    f"{joint} angle series not available for this video."
                )
                continue
            fig = build_joint_timeseries(
                time_s, left, right, joint, lcol, rcol, y_label,
                height=500,
            )
            st.plotly_chart(
                fig, use_container_width=True,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "lasso2d", "select2d", "toggleSpikelines",
                    ],
                },
            )
            st.markdown(_INTERACTIVE_HINT, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# STEP 4 — RESULTS  (legacy dashboard wrapped, until Phase 5 carves it up)
# ──────────────────────────────────────────────
def _render_step_4_legacy_dashboard() -> None:
    """Renders the existing dashboard from session-state results.

    This is the same metric/graphs/insights flow that used to live at the
    bottom of app.py — now invoked only after Step 3 has populated
    `features` / `insights` / `fps` / `total_frames`. Phases 3-5 of the
    UI overhaul will replace each section here with the new design.
    """
    features      = st.session_state["features"]
    insights      = st.session_state["insights"]
    fps           = st.session_state["fps"]
    total_frames  = st.session_state["total_frames"]
    user_height_cm = st.session_state["patient"].get("height_cm", DEFAULT_HEIGHT_CM)

    if features is None or insights is None:
        st.warning(
            "No analysis results yet. Go back to Step 3 and upload a video."
        )
        if st.button("← Back to Upload"):
            st.session_state["step"] = 3
            st.rerun()
        return

    st.markdown('<h1 class="hero-title">GaitVision</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Analysis Results</p>',
        unsafe_allow_html=True,
    )

    st.success(
        f" Processed **{total_frames}** frames at "
        f"**{fps:.1f} FPS** — {features['duration_sec']:.1f}s total"
    )

    mpp         = features.get("meters_per_pixel")
    npasses     = features.get("num_passes", 0)
    frames_used = features.get("frames_used", 0)
    if mpp:
        st.caption(
            f" Calibration: {mpp * 1000:.3f} mm/px  ·  "
            f"{npasses} valid pass{'es' if npasses != 1 else ''}  ·  "
            f"{frames_used}/{total_frames} frames used  ·  "
            f"height = {user_height_cm} cm"
        )
    else:
        st.caption(
            f" Scale calibration failed (no clear stance frames). "
            f"{npasses} valid pass{'es' if npasses != 1 else ''}  ·  "
            f"{frames_used}/{total_frames} frames used. "
            f"Distances will be reported in pixels."
        )

    ab = features.get("ankle_baseline") or {}
    if ab:
        st.caption(
            f" Ankle baseline correction: "
            f"L = {ab['offset_deg_left']:+.1f}°, "
            f"R = {ab['offset_deg_right']:+.1f}°  "
            f"({ab['method']}"
            + (f", n = {ab['n_frames']} frames" if ab["n_frames"] > 0 else "")
            + ")"
        )

    st.markdown("---")
    _render_metrics(features)

    st.markdown("---")
    _render_graphs(features)

    st.markdown("---")
    _render_interactive_joint_tabs(features)

    # Clinical observations + HIP/KNEE/ANKLE expanders moved here from
    # inside the Gait Cycle tab so the page reads charts-then-text:
    # matplotlib figure → interactive plotly tabs → interpretation block.
    # Gated on the same K>=3 check the Gait Cycle tab uses, so we don't
    # render explanations of curves that weren't computable.
    _gc = features.get("gait_cycle_curves") or {}
    _knee = _gc.get("knee", {}) or {}
    _KL = (_knee.get("left",  {}) or {}).get("K", 0)
    _KR = (_knee.get("right", {}) or {}).get("K", 0)
    if min(_KL, _KR) >= 3:
        st.markdown("---")
        _render_cycle_observations(features)
        _render_cycle_explanations()

    st.markdown("---")
    _render_insights(insights)

    st.markdown("---")
    col_pdf, col_re = st.columns([1, 1])
    with col_pdf:
        try:
            pdf_bytes = _build_pdf_bytes(features, st.session_state["patient"])
            patient_id = st.session_state["patient"].get("patient_id", "report")
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"GaitReport_{patient_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"PDF build failed: {exc}")
    with col_re:
        if st.button("← Re-analyze with different video",
                     use_container_width=True):
            # Clear file + analysis only; keep patient data.
            st.session_state["video_file"]   = None
            st.session_state["features"]     = None
            st.session_state["insights"]     = None
            st.session_state["fps"]          = None
            st.session_state["total_frames"] = None
            st.session_state["step"]         = 3
            st.rerun()


# ──────────────────────────────────────────────
# WIZARD DISPATCH (replaces the old single-page flow)
# ──────────────────────────────────────────────
_render_sidebar_stepper()

_step = st.session_state.get("step", 1)
if _step == 1:
    _render_step_1()
elif _step == 2:
    _render_step_2()
elif _step == 3:
    _render_step_3()
elif _step == 4:
    _render_step_4_legacy_dashboard()
else:
    st.error(f"Unknown wizard step: {_step}")
    st.session_state["step"] = 1


