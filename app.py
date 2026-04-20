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
    page_title="GaitVision — AI Gait Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
def load_pose_model():
    model_path = "pose_landmarker_lite.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )
    return PoseLandmarker.create_from_options(options)


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


def _render_metrics(features: dict) -> None:
    """Render 8 metric cards in a 4×2 grid."""
    st.markdown('<div class="section-title">📊 Gait Metrics</div>', unsafe_allow_html=True)

    cadence = features["cadence"]
    sym = features["symmetry"]
    knee = features["knee_angles"]["overall_mean"]
    stride_cv = features["stride_cv"]
    step_count = features["step_data"]["total_steps"]
    step_len = features["step_length"]["mean"]
    torso = features["torso_lean"]["mean"]
    mean_step_t = features["step_timing"]["mean_step_time"]
    direction = features["direction"]

    row1 = st.columns(4)
    row2 = st.columns(4)

    cards = [
        ("Step Count", str(step_count), "steps"),
        ("Cadence", f"{cadence}", "steps / min"),
        ("Symmetry", f"{sym:.2%}", "L/R match"),
        ("Avg Knee Angle", f"{knee:.1f}°", "flexion"),
        ("Stride CV", f"{stride_cv:.3f}", "consistency"),
        ("Step Length", f"{step_len:.3f}", "normalized"),
        ("Torso Lean", f"{torso:.1f}°", "+ fwd / - back"),
        ("Step Time", f"{mean_step_t:.3f}s" if mean_step_t > 0 else "N/A", "avg interval"),
    ]

    for i, (label, val, unit) in enumerate(cards):
        col = row1[i] if i < 4 else row2[i - 4]
        with col:
            st.markdown(_metric_card(label, val, unit), unsafe_allow_html=True)

    st.markdown(f"<p style='color:#6E7681;font-size:0.82rem;margin-top:0.5rem;'>🧭 Walking direction: <b style='color:#C9D1D9'>{direction}</b></p>",
                unsafe_allow_html=True)


def _render_graphs(features: dict) -> None:
    """Render 6 plot tabs."""
    st.markdown('<div class="section-title">📈 Analysis Graphs</div>', unsafe_allow_html=True)

    figs = build_all_figures(features)

    tab_labels = [
        "Knee Angles",
        "Heel Position",
        "Step Length",
        "Timing",
        "Torso Lean",
        "Ankle Trajectory",
    ]
    tabs = st.tabs(tab_labels)

    tab_keys = ["knee", "heel", "step", "timing", "torso", "ankle"]
    
    descriptions = {
        "knee": "**Understanding this graph:** This graph tracks the flexion (bending) angle of your knees over time. 0° represents full extension (a straight leg), while higher values indicate knee bending. The grey shaded band represents the typical range of motion for normal walking (0° to 65°). If your leg cannot reach ~0°, it may indicate a limp or extension deficit.",
        "heel": "**Understanding this graph:** This plots the raw horizontal forward progression of your heels. The grey dotted line represents an idealized smoothed envelope of your overall motion to help highlight any sudden jitter, dragging, or instability.",
        "step": "**Understanding this graph:** Step length is normalized by dividing the physical ground distance of a step by your estimated body height (shoulder to heel mapping). This scaling prevents taller subjects from artificially looking like they overstride. A normal healthy step length is roughly 41.5% of height. The grey band (0.35 – 0.55) represents a standard healthy ratio. Values dropping below indicate short, shuffling steps; values above suggest over-striding.",
        "timing": "**Understanding this graph:** This chart displays the elapsed time between consecutive steps. Consistent timing is a key indicator of gait balance. The grey band highlights an acceptable variation margin (±10%) around your personal average. Spikes outside this box suggest irregular pacing or limping.",
        "torso": "**Understanding this graph:** The torso lean angle is measured against a perfect vertical axis (0°). The grey shaded zone (±7°) highlights an acceptable physiological comfort zone for upright walking. Consistent numbers outside this layer highlight a tendency to lean heavily forward (+ values) or backward (- values).",
        "ankle": "**Understanding this graph:** This shows the ankles crossing each other as you walk. (Note: position is normalized to screen width from 0 to 1). The dotted grey curve is an idealized harmonic sine wave model, representing the expected smooth, pendulum-like motion of a perfect walking swing phase. Sharp deviations imply jerky movements."
    }

    for tab, key in zip(tabs, tab_keys):
        with tab:
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
# SECTION 2: UPLOAD
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
        pose_model = load_pose_model()
        progress_bar.progress(0.02, text=" Extracting poses frame-by-frame…")

        def _progress_cb(frac: float):
            pct = min(int(frac * 80), 80)           # 0-80% for extraction
            progress_bar.progress(pct / 100, text=f" Extracting poses… {pct}%")

        # Stage 1 — Extract
        raw, fps, total_frames = extract_poses(tmp_path, pose_model, _progress_cb)

        progress_bar.progress(0.82, text="🔧 Preprocessing time series…")
        status_text.markdown("*Smoothing & interpolating landmark trajectories…*")

        # Stage 2 — Preprocess
        ts = build_time_series(raw)

        progress_bar.progress(0.88, text=" Computing 10 gait features…")
        status_text.markdown("*Running feature extraction algorithms…*")

        # Stage 3 — Features
        features = compute_all_features(ts, fps, total_frames)
        features["_ts"] = ts   # pass time-series for heel plots

        progress_bar.progress(0.95, text=" Generating graphs…")
        status_text.markdown("*Building visualizations…*")

        # Stage 4 — Insights
        insights = interpret(features)

        progress_bar.progress(1.0, text=" Analysis complete!")
        status_text.empty()

        st.success(f" Processed **{total_frames}** frames at **{fps:.1f} FPS** — {features['duration_sec']:.1f}s total")

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


