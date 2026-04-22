"""
app.py
Streamlit entry point for the Video-Based Gait Analysis System.
"""

import os
import urllib.request
import tempfile
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
    page_title="AI Gait Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Default adult height — used until the user changes it in the height input on the main page.
DEFAULT_HEIGHT_CM = 170

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #C9D1D9;
  }

  /* Header */
  .hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58A6FF, #3FB950, #FF7B72);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 1.1rem;
    color: #6E7681;
    margin-bottom: 1.5rem;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(145deg, #161B22, #0D1117);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(88,166,255,0.18);
  }
  .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #6E7681;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #58A6FF;
    line-height: 1;
  }
  .metric-unit {
    font-size: 0.75rem;
    color: #6E7681;
    margin-top: 4px;
  }

  /* Section headings */
  .section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #E6EDF3;
    border-left: 4px solid #58A6FF;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
  }

  /* Insight cards */
  .obs-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-left: 3px solid #3FB950;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.92rem;
    line-height: 1.5;
  }
  .sug-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-left: 3px solid #F78166;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.92rem;
    line-height: 1.5;
  }

  /* Disclaimer */
  .disclaimer {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.8rem;
    color: #6E7681;
    text-align: center;
    margin-top: 2rem;
  }

  /* Streamlit tab styling - give tabs visible spacing and button look */
  div[role="tablist"] {
    gap: 12px;
  }
  div[role="tablist"] button[role="tab"] {
    background-color: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 8px;
    padding: 8px 16px;
    color: #cbd5e1;
    font-weight: 500;
    transition: all 0.15s ease;
  }
  div[role="tablist"] button[role="tab"]:hover {
    background-color: rgba(148, 163, 184, 0.18);
    border-color: rgba(148, 163, 184, 0.45);
    color: #f1f5f9;
  }
  div[role="tablist"] button[role="tab"][aria-selected="true"] {
    background-color: rgba(96, 165, 250, 0.15);
    border-color: #60a5fa;
    color: #60a5fa;
  }
  /* Hide the default red underline indicator since we use background instead */
  div[role="tablist"] div[data-baseweb="tab-highlight"] {
    display: none;
  }

  /* Upload area */
  .upload-zone {
    background: #161B22;
    border: 2px dashed #30363D;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.2s;
  }

  /* Progress bar */
  .stProgress > div > div {
    background: linear-gradient(90deg, #58A6FF, #3FB950) !important;
  }

  /* Streamlit default overrides */
  .stFileUploader { 
    border-radius: 10px; 
  }
  div[data-testid="stMetricValue"] { 
    color: #58A6FF !important; 
  }
  .reportview-container .main footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


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
    # The gait-cycle tab uses _render_cycle_explanations() instead.
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
                    _render_cycle_observations(features)
                    _render_cycle_explanations()
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


# ──────────────────────────────────────────────
# SECTION 1: HEADER
# ──────────────────────────────────────────────
st.markdown('<h1 class="hero-title">GaitVision</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI-powered video gait analysis — pose estimation · metrics · insights</p>',
            unsafe_allow_html=True)

st.markdown("""
<div style="background:#161B22;border:1px solid #30363D;border-radius:10px;padding:14px 18px;margin-bottom:1.5rem;">
    <b style="color:#E6EDF3;"> Instructions</b><br>
    <span style="font-size:0.88rem;color:#8B949E;">
    1. Upload a walking video in <b>side (lateral) view</b> with the <b>full body visible</b>.<br>
    2. Supported format: <code>.mp4</code> · Keep video under <b>60 seconds</b> for best performance.<br>
    3. The system will extract pose landmarks, compute 10 gait features, and generate insights automatically.
    </span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SECTION 2 — HEIGHT (calibrates pixel → meters for step length)
# ──────────────────────────────────────────────
height_col = st.columns([1, 3])[0]
with height_col:
    user_height_cm = st.number_input(
        " Your height (cm) — used for step-length calibration",
        min_value=100, max_value=220,
        value=DEFAULT_HEIGHT_CM, step=1,
        help="2D video has no absolute scale, so step length is calibrated from "
             "your height via the leg-length ratio (leg ≈ 53% of height). "
             "Leave at default for an average-adult estimate.",
    )

# ──────────────────────────────────────────────
# SECTION 3: UPLOAD
# ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a walking video",
    type=["mp4", "mov", "avi", "mkv"],
    help="Side-view full-body walking video recommended",
    label_visibility="collapsed",
)

if uploaded is not None:
    # Preview
    st.video(uploaded)

    # ──────────────────────────────────────────
    # PROCESSING
    # ──────────────────────────────────────────
    st.markdown('<div class="section-title"> Processing</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0, text="Initializing pose model…")
    status_text = st.empty()

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        # Load model
        pose_options = load_pose_model_options()
        progress_bar.progress(0.02, text=" Extracting poses frame-by-frame…")

        def _progress_cb(frac: float):
            pct = min(int(frac * 80), 80)           # 0-80% for extraction
            progress_bar.progress(pct / 100, text=f" Extracting poses… {pct}%")

        # Stage 1 — Extract
        raw, fps, total_frames = extract_poses(tmp_path, pose_options, _progress_cb)

        progress_bar.progress(0.82, text=" Preprocessing time series…")
        status_text.markdown("*Smoothing & interpolating landmark trajectories…*")

        # Stage 2 — Preprocess
        ts = build_time_series(raw)

        progress_bar.progress(0.88, text=" Computing 10 gait features…")
        status_text.markdown("*Running feature extraction algorithms…*")

        # Stage 3 — Features. Pixel→meter scaling uses an assumed adult height
        # (true height is unknowable from 2D video without a calibration object).
        features = compute_all_features(ts, fps, total_frames, user_height_cm=user_height_cm)
        features["_ts"] = ts   # pass time-series for heel plots

        progress_bar.progress(0.95, text=" Generating graphs…")
        status_text.markdown("*Building visualizations…*")

        # Stage 4 — Insights
        insights = interpret(features)

        progress_bar.progress(1.0, text=" Analysis complete!")
        status_text.empty()

        st.success(f" Processed **{total_frames}** frames at **{fps:.1f} FPS** — {features['duration_sec']:.1f}s total")

        # ── Audit / sanity-check info (per Section 3 spec) ──
        mpp = features.get("meters_per_pixel")
        npasses = features.get("num_passes", 0)
        frames_used = features.get("frames_used", 0)
        if mpp:
            st.caption(
                f" Calibration: {mpp*1000:.3f} mm/px  ·  "
                f"{npasses} valid pass{'es' if npasses != 1 else ''}  ·  "
                f"{frames_used}/{total_frames} frames used  ·  "
                f"height = {user_height_cm} cm"
                + (" (default)" if user_height_cm == DEFAULT_HEIGHT_CM else " (user-set)")
            )
        else:
            st.caption(
                f" Scale calibration failed (no clear stance frames). "
                f"{npasses} valid pass{'es' if npasses != 1 else ''}  ·  "
                f"{frames_used}/{total_frames} frames used. "
                f"Distances will be reported in pixels."
            )

        # Ankle baseline diagnostic — surfaces per-leg correction so it's
        # auditable. Static-detected = subject paused somewhere; running-median
        # = no static frames found, so we used each leg's own median over the
        # full clip as the per-side neutral.
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

        # ── Section 3: Metrics ─────────────────
        _render_metrics(features)

        st.markdown("---")

        # ── Section 4: Graphs ─────────────────
        _render_graphs(features)

        st.markdown("---")

        # ── Section 5: Insights ───────────────
        _render_insights(insights)

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f" Processing failed: {e}")
        st.exception(e)

    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

else:
    # Hero placeholder
    st.markdown("""
    <div style="background:linear-gradient(135deg,#161B22,#0D1117);border:2px dashed #30363D;
                border-radius:16px;padding:3rem;text-align:center;margin-top:1rem;">
        <div style="font-size:3.5rem;margin-bottom:1rem"></div>
        <div style="font-size:1.15rem;color:#8B949E;font-weight:500;">
            Upload a side-view walking video to begin gait analysis
        </div>
        <div style="font-size:0.85rem;color:#6E7681;margin-top:0.8rem;">
            Supports MP4 · MOV · AVI · MKV
        </div>
    </div>
    """, unsafe_allow_html=True)


