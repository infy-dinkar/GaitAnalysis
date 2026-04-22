#  GaitVision — AI Gait Analyzer

GaitVision is an end-to-end, lightweight, rule-based system for gait analysis. It takes a side-view walking video, extracts biomechanical metrics, plots cycle-normalized joint kinematics against healthy-adult reference bands, and surfaces plain-language observations.

##  Features

### Metrics — computed twice (whole-video Total + steady-state Clean)
- Step count, cadence, gait symmetry
- Average knee flexion, stride variation (CV)
- Step length (anatomically calibrated to the subject's height)
- Step timing intervals
- Torso lean
- Ankle and heel trajectories

### Cycle-normalized joint kinematics
- Per-stride hip, knee, and ankle angles resampled to 0–100% of the gait cycle
- Mean ± 1 SD curves overlaid on healthy-adult reference bands (Perry & Burnfield, Winter, Kadaba)
- Auto-generated kinematic observations: ROM, peak timing, and amplitude ratio per leg

### Robustness layers
- Direction-aware pass segmentation (handles bidirectional walking videos)
- Anatomical pixel-to-meter scaling from user-supplied height (no on-screen ruler needed)
- Per-leg ankle baseline correction (auto-detected static stance with running-median fallback)
- Heel-strike amplitude floor — rejects foot-drag events with low heel clearance
- Cycle duration filter — rejects spans outside 0.7–1.5× the leg's median stride duration
- Static-stance detection restricted to clean walking windows (prevents turning-point poisoning of the baseline)

### UI
- Streamlit app with dark theme and pill-styled chart tabs
- Subject height input on the main screen
- Stacked Total + Clean metric grids
- 7 chart tabs: Knee Angles · Heel Position · Step Length · Timing · Torso Lean · Ankle Trajectory · Gait Cycle
- Per-joint expanders with neutral, descriptive interpretation guidance + clinical disclaimer

##  Pipeline

1. **Pose extraction** — MediaPipe Pose landmarks per frame
2. **Pass segmentation** — Sav-Gol smoothed hip-midpoint velocity → stable-direction passes with trimmed cores
3. **Calibration** — `meters_per_pixel = (height_cm × 0.53) / median(leg-length-px)`
4. **Per-frame angles** — hip / knee / ankle in pixel coordinates with sign + baseline correction
5. **Heel-strike detection** — heel-to-heel separation peaks (metrics) and heel-y peaks with amplitude floor (cycle plots)
6. **Cycle assembly** — extract → duration-filter → MAD-filter → ensemble-mean per leg per joint
7. **Rule-based interpretation** — threshold engine reads `clean_metrics` and emits observations & suggestions

##  Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, layout, captions, observations panel |
| `gait_engine.py` | Pose loading, pass segmentation, scaling, angle math, metrics, interpreter |
| `gait_cycle.py` | Heel-strike detection, cycle extraction, duration filter, ensemble stats |
| `gait_plots.py` | Matplotlib figures + healthy-adult reference curves |
| `requirements.txt` | Python dependencies |
| `packages.txt` | apt dependencies for Streamlit Cloud (Debian trixie) |
| `.python-version` | Pinned Python 3.11 (mediapipe wheel availability) |

##  Tech Stack

- **UI**: Streamlit
- **Pose extraction**: MediaPipe Pose (`@st.cache_resource` for fast model load)
- **Computer vision**: OpenCV (headless)
- **Signal processing**: NumPy + SciPy (Savitzky-Golay smoothing, `find_peaks`, MAD filtering)
- **Charting**: Matplotlib (Agg backend, custom dark theme)

##  Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch
streamlit run app.py
```

Upload a side-profile walking video that shows the full body. Enter the subject's height (drives the stride-length calibration), then watch the metrics, charts, and observations populate.

Python 3.11 is recommended — see `.python-version`. For Streamlit Cloud deployment, system packages are declared in `packages.txt`.

> ** Disclaimer**: GaitVision is a educational prototype. All outputs are derived from 2D pose estimation and do not replace professional 3D clinical gait laboratories or medical diagnosis. Auto-generated observations are descriptive only — clinical interpretation belongs to a qualified professional considering the full clinical picture, medical history, and other assessments.
