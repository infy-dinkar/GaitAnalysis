"""
gait_plots.py
Six matplotlib plot functions for gait analysis.
All use the Agg backend and return Figure objects (Streamlit-compatible).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# DESIGN TOKENS
# ──────────────────────────────────────────────
PALETTE = {
    "bg":         "#0D1117",
    "panel":      "#161B22",
    "border":     "#30363D",
    "left":       "#58A6FF",
    "right":      "#FF7B72",
    "accent":     "#3FB950",
    "warn":       "#E3B341",
    "text":       "#C9D1D9",
    "muted":      "#6E7681",
    "grid":       "#21262D",
    "normal":     "#8b949e",
}

def _base_fig(nrows=1, ncols=1, figsize=(12, 4.5), title=""):
    """Return a styled (fig, axes) pair."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              facecolor=PALETTE["bg"])
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.8)

    if title:
        fig.suptitle(title, color=PALETTE["text"], fontsize=13,
                     fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


# ──────────────────────────────────────────────
# PLOT 1: KNEE ANGLES
# ──────────────────────────────────────────────
def plot_knee_angles(features: dict, fps: float) -> plt.Figure:
    """Left & right knee angle over time."""
    fig, axes = _base_fig(1, 2, figsize=(14, 5), title="Knee Flexion Angles Over Time")
    ka = features["knee_angles"]
    times = np.arange(len(ka["left"])) / fps

    for ax, side, color, label in [
        (axes[0], "left", PALETTE["left"], "Left Knee (User Data)"),
        (axes[1], "right", PALETTE["right"], "Right Knee (User Data)"),
    ]:
        ax.fill_between(times, 120, 160, color=PALETTE["normal"], alpha=0.15, label="Normal Range (120°-160°)")
        ax.plot(times, ka[side], color=color, linewidth=1.6, label=label, alpha=0.9)
        mean_val = ka[f"{side}_mean"]
        ax.axhline(mean_val, color=PALETTE["accent"], linestyle="--",
                   linewidth=1.2, label=f"Mean: {mean_val:.1f}°")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Angle (°)", fontsize=10)
        ax.set_title(f"{label} — Mean: {mean_val:.1f}°",
                     color=PALETTE["text"], fontsize=11, pad=8)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=8, framealpha=0.6)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# PLOT 2: HEEL POSITION TRAJECTORY
# ──────────────────────────────────────────────
def plot_heel_trajectory(features: dict, fps: float) -> plt.Figure:
    """Heel X-coordinate trajectories with detected step peaks."""
    fig, axes = _base_fig(1, 2, figsize=(14, 5), title="Heel Position Trajectory")
    sd = features["step_data"]

    for ax, side, color, label in [
        (axes[0], "left", PALETTE["left"], "Left Heel (User Data)"),
        (axes[1], "right", PALETTE["right"], "Right Heel (User Data)"),
    ]:
        # We need the time series from features — stored in ankle_trajectory proxy
        # Instead pass full ts via features["_ts"] trick
        ts = features.get("_ts", {})
        if not ts:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=PALETTE["muted"], transform=ax.transAxes)
            continue

        heel_x = ts[f"{side}_heel"]["x"]
        times = np.arange(len(heel_x)) / fps

        ax.plot(times, heel_x, color=color, linewidth=1.4, alpha=0.85, label=label)

        idx = sd[f"{side}_indices"]
        if len(idx) > 0:
            valid = idx[idx < len(heel_x)]
            ax.scatter(valid / fps, heel_x[valid],
                       color=PALETTE["accent"], zorder=5, s=30, label="Steps", alpha=0.9)

        if len(heel_x) > 20:
            smoothed_heel = np.convolve(heel_x, np.ones(15)/15, mode='same')
            ax.plot(times, smoothed_heel, color=PALETTE["normal"], linestyle=":", linewidth=2, alpha=0.6, label="Smoothed Normal Pattern")

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Heel X (normalized)", fontsize=10)
        ax.set_title(f"{label} — {sd[f'{side}_count']} steps",
                     color=PALETTE["text"], fontsize=11, pad=8)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=8, framealpha=0.6)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# PLOT 3: STEP LENGTH
# ──────────────────────────────────────────────
def plot_step_length(features: dict) -> plt.Figure:
    """Bar chart + line of normalized step lengths detected."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.8)

    vals = features["step_length"]["values"]
    mean_val = features["step_length"]["mean"]

    if len(vals) == 0:
        ax.text(0.5, 0.5, "Insufficient steps detected for step length analysis",
                ha="center", va="center", color=PALETTE["muted"], fontsize=12,
                transform=ax.transAxes)
    else:
        x = np.arange(len(vals))
        ax.axhspan(0.4, 0.6, color=PALETTE["normal"], alpha=0.15, label="Normal Range (0.4 - 0.6)")
        bars = ax.bar(x, vals, color=PALETTE["left"], alpha=0.75, edgecolor=PALETTE["border"],
                      linewidth=0.8, label="Step Length (User Data)")
        # Color bars above/below mean
        for bar, v in zip(bars, vals):
            bar.set_color(PALETTE["accent"] if v >= mean_val else PALETTE["right"])
            bar.set_alpha(0.8)

        ax.plot(x, vals, color=PALETTE["warn"], linewidth=1.4, marker="o",
                markersize=4, alpha=0.9, label="Trend")
        ax.axhline(mean_val, color=PALETTE["text"], linestyle="--",
                   linewidth=1.5, label=f"Mean: {mean_val:.3f}")

        ax.set_xlabel("Step Index", fontsize=10, color=PALETTE["text"])
        ax.set_ylabel("Normalized Step Length", fontsize=10, color=PALETTE["text"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=9, framealpha=0.6)

    ax.set_title("Step Length (Normalized to Hip Width)",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# PLOT 4: STEP / STRIDE TIMING
# ──────────────────────────────────────────────
def plot_timing(features: dict) -> plt.Figure:
    """Timing distribution: left vs right step intervals."""
    fig, axes = _base_fig(1, 2, figsize=(14, 5), title="Step & Stride Timing")
    timing = features["step_timing"]

    for ax, side, color, label in [
        (axes[0], "left", PALETTE["left"], "Left Side (User Data)"),
        (axes[1], "right", PALETTE["right"], "Right Side (User Data)"),
    ]:
        diffs = timing[side]
        if len(diffs) < 2:
            ax.text(0.5, 0.5, "Insufficient steps", ha="center", va="center",
                    color=PALETTE["muted"], transform=ax.transAxes, fontsize=11)
            ax.set_title(f"{label} — No Data", color=PALETTE["text"], fontsize=11)
            continue

        x = np.arange(len(diffs))
        mean_t = np.mean(diffs)
        ax.axhspan(mean_t * 0.9, mean_t * 1.1, color=PALETTE["normal"], alpha=0.15, label="Normal Variation (±10%)")
        ax.bar(x, diffs, color=color, alpha=0.65, edgecolor=PALETTE["border"], linewidth=0.8)
        ax.plot(x, diffs, color=PALETTE["warn"], linewidth=1.4, marker="o",
                markersize=5, alpha=0.9)
        ax.axhline(mean_t, color=PALETTE["accent"], linestyle="--",
                   linewidth=1.2, label=f"Mean: {mean_t:.3f}s")

        ax.set_xlabel("Step Interval Index", fontsize=10)
        ax.set_ylabel("Time (s)", fontsize=10)
        ax.set_title(f"{label} — Mean: {mean_t:.3f}s",
                     color=PALETTE["text"], fontsize=11, pad=8)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=8, framealpha=0.6)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# PLOT 5: TORSO LEAN
# ──────────────────────────────────────────────
def plot_torso_lean(features: dict, fps: float) -> plt.Figure:
    """Torso lean angle over time with shaded comfort zone."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.8)

    tl = features["torso_lean"]
    angles = tl["angles"]
    mean_a = tl["mean"]
    times = np.arange(len(angles)) / fps

    ax.fill_between(times, -7, 7, alpha=0.15, color=PALETTE["normal"], label="Normal Range (±7°)")
    ax.plot(times, angles, color=PALETTE["warn"], linewidth=1.5, alpha=0.85, label="Torso Lean (User Data)")
    ax.axhline(0, color=PALETTE["muted"], linestyle="-", linewidth=0.8)
    ax.axhline(mean_a, color=PALETTE["right"], linestyle="--",
               linewidth=1.3, label=f"Mean: {mean_a:.1f}°")

    ax.set_xlabel("Time (s)", fontsize=10, color=PALETTE["text"])
    ax.set_ylabel("Lean Angle (°)", fontsize=10, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.set_title("Torso Lean Angle Over Time (+ = Forward, − = Backward)",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
              fontsize=9, framealpha=0.6)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# PLOT 6: ANKLE TRAJECTORY
# ──────────────────────────────────────────────
def plot_ankle_trajectory(features: dict, fps: float) -> plt.Figure:
    """Left and right ankle X-coordinate over time."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.8)

    at = features["ankle_trajectory"]
    n = max(len(at["left_x"]), len(at["right_x"]))
    times = np.arange(n) / fps

    ax.plot(np.arange(len(at["left_x"])) / fps, at["left_x"],
            color=PALETTE["left"], linewidth=1.4, alpha=0.85, label="Left Ankle (User Data)")
    ax.plot(np.arange(len(at["right_x"])) / fps, at["right_x"],
            color=PALETTE["right"], linewidth=1.4, alpha=0.85, label="Right Ankle (User Data)")

    ax.fill_between(
        np.arange(min(len(at["left_x"]), len(at["right_x"]))) / fps,
        at["left_x"][:min(len(at["left_x"]), len(at["right_x"]))],
        at["right_x"][:min(len(at["left_x"]), len(at["right_x"]))],
        alpha=0.08, color=PALETTE["accent"]
    )

    y_mean = (np.mean(at["left_x"]) + np.mean(at["right_x"])) / 2
    if not np.isnan(y_mean):
        y_amp = max(np.std(at["left_x"]), 0.05) * 1.5
        ideal_sine = y_mean + y_amp * np.sin(2 * np.pi * times)
        ax.plot(times, ideal_sine, color=PALETTE["normal"], linestyle=":", linewidth=2, alpha=0.5, label="Ideal Harmonic Pattern")

    ax.set_xlabel("Time (s)", fontsize=10, color=PALETTE["text"])
    ax.set_ylabel("Ankle X (normalized)", fontsize=10, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.set_title("Ankle X-Trajectory Over Time (Foot Clearance Pattern)",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
              fontsize=9, framealpha=0.6)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# CONVENIENCE BUILDER
# ──────────────────────────────────────────────
def build_all_figures(features: dict) -> dict:
    """
    Call all 6 plot functions and return a dict of Figure objects.
    """
    fps = features.get("fps", 30.0)
    return {
        "knee":    plot_knee_angles(features, fps),
        "heel":    plot_heel_trajectory(features, fps),
        "step":    plot_step_length(features),
        "timing":  plot_timing(features),
        "torso":   plot_torso_lean(features, fps),
        "ankle":   plot_ankle_trajectory(features, fps),
    }
