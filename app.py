"""
app.py
Streamlit entry point for the Video-Based Gait Analysis System.
"""

import os
import urllib.request
import tempfile
import streamlit as st
import mediapipe as mp

from gait_engine import extract_poses, build_time_series, compute_all_features, interpret
from gait_plots import build_all_figures

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

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background-color: #161B22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: #6E7681;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background-color: #21262D !important;
    color: #58A6FF !important;
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
        '<div class="section-title">📊 Total Metrics '
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
        f"🧭 Walking direction: <b style='color:#C9D1D9'>{direction}</b></p>",
        unsafe_allow_html=True,
    )


def _render_cycle_explanations() -> None:
    """Caption + three per-joint expandable sections explaining the cycle chart.
    Renders below the gait-cycle figure. Dual-audience copy (clinician + layperson)."""
    st.caption(
        "The grey band shows the normal healthy-adult range at each point of the "
        "step. Your measurements are the blue (left leg) and red (right leg) lines."
    )

    with st.expander("Understanding the HIP graph"):
        st.markdown("""
**What you're looking at:** How your thigh swings forward and backward during one complete step, starting the moment your heel hits the ground.

**The grey band = normal range.** A healthy adult's hip moves through about 40° of motion each step — swung forward (about +30°) when the heel lands, swinging backward (about −10°) as the body rolls over the planted foot, then forward again to set up the next step.

**The blue and red lines = your left and right hip.** Your lines should stay roughly within the grey band and follow its shape — a smooth wave pattern.

**What deviations mean:**
- **Lines stay flat or have small range (under 25°):** reduced hip mobility. Common with tight hip flexors, sedentary lifestyle, hip arthritis, or Parkinsonian gait.
- **Minimum doesn't go below 0° (curve stays in positive territory):** the hip is never extending past neutral. This limits stride length and often shows up as shorter steps.
- **Blue and red curves separated vertically:** one hip has more or less range than the other — often a sign of favoring one side due to pain, weakness, or a previous injury.
- **Curves offset in time (shifted left or right from the grey band):** unusual timing of hip motion relative to the stride. Can indicate pathological stride patterns.
""")

    with st.expander("Understanding the KNEE graph"):
        st.markdown("""
**What you're looking at:** How much your knee bends during one complete step. 0° means a perfectly straight leg; higher numbers mean more bending.

**The grey band = normal range.** A healthy knee shows two distinct peaks per step:

1. **First peak (~18°) right after heel strike:** the knee bends slightly to absorb landing impact, like a shock absorber.
2. **Second, larger peak (~65°) during mid-swing:** the knee bends sharply to lift your foot off the ground so your toes don't drag as the leg swings forward.

**The blue and red lines = your left and right knee.** You should see two clear humps on each line, matching the shape of the grey band.

**What deviations mean:**
- **Missing first peak (flat curve during loading, 0–20%):** stiff-knee landing. Common after knee injury, surgery, or with quadriceps weakness. Over time this causes joint pain because impact goes straight into the joint instead of being absorbed.
- **Low swing peak (second hump under 50°):** foot isn't clearing the ground well — a trip-and-fall risk. Often associated with weakness or neurological conditions.
- **Very different blue vs red:** you're favoring one leg. Common during recovery from injury or with unilateral pain.
- **Wider shaded band around your line:** your knee motion varies a lot from step to step. May indicate instability, fatigue, or poor neuromuscular control.
""")

    with st.expander("Understanding the ANKLE graph"):
        st.markdown("""
**What you're looking at:** How your ankle rotates up and down during one complete step.
- **Positive values (dorsiflexion):** your toes are pulled toward your shin — this happens when your foot is flat on the ground and your body rolls forward over it.
- **Negative values (plantarflexion):** your toes are pointed down — this happens when you push off the ground to propel yourself forward.

**The grey band = normal range.** A healthy ankle shows:
- A small negative dip (~−5°) right after heel strike as the forefoot slaps down.
- A gradual rise to about +10° as you roll forward over the planted foot.
- A sharp plunge to about −18° at push-off (around 60% of the step) — this is where most of your walking power comes from.
- A return to near 0° during swing so your toes clear the ground.

**The blue and red lines = your left and right ankle.**

**What deviations mean:**
- **Weak push-off (minimum less negative than −10°):** reduced calf strength or Achilles tendon issues. Usually shows up as slower walking speed and shorter strides.
- **Ankle doesn't return to neutral during swing (stays negative past 70%):** toe drag risk. Can indicate foot drop, often from nerve involvement.
- **Flat curve with little variation:** generally reduced ankle mobility. Common with stiff ankles, older age, or after ankle injury.
- **Very different left vs right:** asymmetric push-off, a common compensation for pain or weakness on one side.
""")


def _render_graphs(features: dict) -> None:
    """Render plot tabs (6 existing + 1 new gait-cycle tab)."""
    st.markdown('<div class="section-title">📈 Analysis Graphs</div>', unsafe_allow_html=True)

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
        "knee": "**Understanding this graph:** This graph tracks the flexion (bending) angle of your knees over time. 0° represents full extension (a straight leg), while higher values indicate knee bending. The grey shaded band represents the typical range of motion for normal walking (0° to 65°). If your leg cannot reach ~0°, it may indicate a limp or extension deficit.",
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
        "📏 Your height (cm) — used for step-length calibration",
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

        progress_bar.progress(0.82, text="🔧 Preprocessing time series…")
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
                f"🔬 Calibration: {mpp*1000:.3f} mm/px  ·  "
                f"{npasses} valid pass{'es' if npasses != 1 else ''}  ·  "
                f"{frames_used}/{total_frames} frames used  ·  "
                f"height = {user_height_cm} cm"
                + (" (default)" if user_height_cm == DEFAULT_HEIGHT_CM else " (user-set)")
            )
        else:
            st.caption(
                f"⚠️ Scale calibration failed (no clear stance frames). "
                f"{npasses} valid pass{'es' if npasses != 1 else ''}  ·  "
                f"{frames_used}/{total_frames} frames used. "
                f"Distances will be reported in pixels."
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


