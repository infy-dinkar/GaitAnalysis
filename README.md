# 🦶 GaitVision — AI Gait Analyzer

GaitVision is an end-to-end, lightweight, rule-based AI system for gait analysis. It processes simple side-view walking videos to extract and visualize 10 vital gait metrics alongside physiological baselines.

## ✨ Features

- **No Sensors or Wearables**: Runs purely on consumer video (MP4, MOV, AVI).
- **10 Biomechanical Features**: 
  - Step Count
  - Cadence
  - Gait Symmetry
  - Average Knee Flexion
  - Stride Variation (CV)
  - Normalized Step Length
  - Step Timing Intervals
  - Torso Lean
  - Ankle and Heel Trajectories
- **Comparative Visuals**: Generates highly-detailed matplotlib graphs. All charts automatically overlay the user's data against "Normal Reference Ranges" (e.g., standard ±10% timing variations, 120°–160° knee flexion bands) to make visual interpretation immediate.
- **Rule-Based Insights**: An inference engine scans the calculated metrics against standard threshold triggers and automatically prints custom "Observations" and "Suggestions".

## 🛠️ Tech Stack

- **UI & Routing**: [Streamlit](https://streamlit.io/)
- **Pose Extraction**: [MediaPipe Pose](https://developers.google.com/mediapipe) (Cached for high performance)
- **Computer Vision**: OpenCV (Headless)
- **Signal Processing**: NumPy & SciPy (Savitzky-Golay filters, peak detection)
- **Charting**: Matplotlib (Agg backend custom-styled for dark UI)

## 🚀 Running the App Locally

Ensure your Python environment (like the `Gait` Conda env) is activated, then run:

```bash
# 1. Install required dependencies
pip install -r requirements.txt

# 2. Boot the application
streamlit run app.py
```

Upload a side-profile walking video containing a full-body view to see your metrics!

> **⚠️ Disclaimer**: GaitVision is a research and educational prototype. All outputs are derived from 2D mathematical video estimations and do not replace professional 3D clinical gait laboratories or medical diagnosis.
