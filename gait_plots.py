"""
gait_plots.py
Six matplotlib plot functions for gait analysis.
All use the Agg backend and return Figure objects (Streamlit-compatible).

After the engine rewrite:
  • Distance axes are in METERS (when meters_per_pixel is available).
  • Time axes are in SECONDS (using the actual fps from cv2).
  • Each time-domain plot shades the validated PASSES used for metric
    computation, so the user can see which sections were kept vs discarded.
  • Normal-range bands match adult-gait literature targets supplied by the user.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# DESIGN TOKENS  (dark theme — preserved)
# ──────────────────────────────────────────────
PALETTE = {
    "bg":     "#0D1117",
    "panel":  "#161B22",
    "border": "#30363D",
    "left":   "#58A6FF",
    "right":  "#FF7B72",
    "accent": "#3FB950",
    "warn":   "#E3B341",
    "text":   "#C9D1D9",
    "muted":  "#6E7681",
    "grid":   "#21262D",
    "normal": "#8b949e",
    "pass":   "#58A6FF",   # tint for pass shading
}


def _base_fig(nrows=1, ncols=1, figsize=(12, 4.5), title=""):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=PALETTE["bg"])
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
        fig.suptitle(title, color=PALETTE["text"], fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


def _shade_passes(ax, passes, fps, label_first=True):
    """Shade the validated passes (core ranges) on a time-domain axis."""
    if not passes:
        return
    for i, p in enumerate(passes):
        a = p["core_start"] / fps
        b = p["core_end"]   / fps
        ax.axvspan(
            a, b,
            color=PALETTE["pass"], alpha=0.06,
            label="Validated Pass" if (label_first and i == 0) else None,
        )


# ══════════════════════════════════════════════
# PLOT 1 — KNEE FLEXION
# ══════════════════════════════════════════════
def plot_knee_angles(features: dict, fps: float):
    fig, axes = _base_fig(1, 2, figsize=(14, 5), title="Knee Flexion Angles Over Time")
    ka       = features["knee_angles"]
    passes   = features.get("pass_segments", [])
    times    = np.arange(len(ka["left"])) / fps

    for ax, side, color, label in [
        (axes[0], "left",  PALETTE["left"],  "Left Knee"),
        (axes[1], "right", PALETTE["right"], "Right Knee"),
    ]:
        ax.fill_between(times, 0, 70, color=PALETTE["normal"], alpha=0.12,
                        label="Normal Swing Range (0°–70°)")
        _shade_passes(ax, passes, fps)
        ax.plot(times, ka[side], color=color, linewidth=1.6, alpha=0.9, label=label)

        mean_val = ka[f"{side}_mean"]
        ax.axhline(mean_val, color=PALETTE["accent"], linestyle="--",
                   linewidth=1.2, label=f"Mean: {mean_val:.1f}°")
        peak_key = f"peak_{side}"
        if peak_key in ka:
            ax.axhline(ka[peak_key], color=PALETTE["warn"], linestyle=":",
                       linewidth=1.0, alpha=0.7, label=f"Peak: {ka[peak_key]:.1f}°")

        ax.set_xlabel("Time (s)",  fontsize=10)
        ax.set_ylabel("Flexion (°)", fontsize=10)
        ax.set_title(f"{label} — Mean {mean_val:.1f}°",
                     color=PALETTE["text"], fontsize=11, pad=8)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=8, framealpha=0.6)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════
# PLOT 2 — HEEL TRAJECTORY
# ══════════════════════════════════════════════
def plot_heel_trajectory(features: dict, fps: float):
    fig, axes = _base_fig(1, 2, figsize=(14, 5), title="Heel Position Trajectory")
    sd      = features["step_data"]
    passes  = features.get("pass_segments", [])
    ts      = features.get("_ts", {})

    for ax, side, color, label in [
        (axes[0], "left",  PALETTE["left"],  "Left Heel"),
        (axes[1], "right", PALETTE["right"], "Right Heel"),
    ]:
        if not ts:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=PALETTE["muted"], transform=ax.transAxes)
            continue

        heel_x = ts[f"{side}_heel"]["x"]      # normalized for visualization
        times  = np.arange(len(heel_x)) / fps
        _shade_passes(ax, passes, fps)
        ax.plot(times, heel_x, color=color, linewidth=1.4, alpha=0.85, label=label)

        idx = sd[f"{side}_indices"]
        if len(idx) > 0:
            valid = idx[idx < len(heel_x)]
            ax.scatter(valid / fps, heel_x[valid],
                       color=PALETTE["accent"], zorder=5, s=30,
                       label=f"Heel Strikes ({len(valid)})", alpha=0.9)

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Heel X (image-normalized)", fontsize=10)
        ax.set_title(f"{label} — {sd[f'{side}_count']} steps",
                     color=PALETTE["text"], fontsize=11, pad=8)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=8, framealpha=0.6)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════
# PLOT 3 — STEP LENGTH
# ══════════════════════════════════════════════
def plot_step_length(features: dict):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    for s in ax.spines.values():
        s.set_edgecolor(PALETTE["border"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.8)

    sl       = features["step_length"]
    vals     = sl["values"]
    mean_val = sl["mean"]
    std_val  = sl.get("std", 0.0)
    unit     = sl.get("unit", "m")

    if len(vals) == 0:
        ax.text(0.5, 0.5, "Insufficient steps detected for step length analysis",
                ha="center", va="center", color=PALETTE["muted"], fontsize=12,
                transform=ax.transAxes)
    else:
        x = np.arange(len(vals))

        # Normal-range band only meaningful when scale is in meters
        if unit == "m":
            ax.axhspan(0.55, 0.80, color=PALETTE["normal"], alpha=0.15,
                       label="Adult Normal (0.55 – 0.80 m)")

        bars = ax.bar(x, vals, color=PALETTE["left"], alpha=0.75,
                      edgecolor=PALETTE["border"], linewidth=0.8, label="Step Length")
        for bar, v in zip(bars, vals):
            bar.set_color(PALETTE["accent"] if v >= mean_val else PALETTE["right"])
            bar.set_alpha(0.8)

        ax.plot(x, vals, color=PALETTE["warn"], linewidth=1.4, marker="o",
                markersize=4, alpha=0.9, label="Trend")
        ax.axhline(mean_val, color=PALETTE["text"], linestyle="--",
                   linewidth=1.5, label=f"Mean: {mean_val:.3f} {unit}")
        if std_val > 0:
            ax.fill_between(x, mean_val - std_val, mean_val + std_val,
                            color=PALETTE["text"], alpha=0.07,
                            label=f"±1 σ ({std_val:.3f} {unit})")

        ax.set_xlabel("Step Index", fontsize=10, color=PALETTE["text"])
        ax.set_ylabel(f"Step Length ({unit})", fontsize=10, color=PALETTE["text"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=9, framealpha=0.6)

    title = "Step Length per Step" + (" (meters, anatomically scaled)" if unit == "m"
                                      else " (pixels — set height in sidebar to scale)")
    ax.set_title(title, color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════
# PLOT 4 — STEP TIMING
# ══════════════════════════════════════════════
def plot_timing(features: dict):
    fig, axes = _base_fig(1, 2, figsize=(14, 5), title="Step & Stride Timing")
    timing = features["step_timing"]

    for ax, side, color, label in [
        (axes[0], "left",  PALETTE["left"],  "Left Side"),
        (axes[1], "right", PALETTE["right"], "Right Side"),
    ]:
        diffs = timing[side]
        if len(diffs) < 2:
            ax.text(0.5, 0.5, "Insufficient steps", ha="center", va="center",
                    color=PALETTE["muted"], transform=ax.transAxes, fontsize=11)
            ax.set_title(f"{label} — No Data", color=PALETTE["text"], fontsize=11)
            continue

        x = np.arange(len(diffs))
        mean_t = float(np.mean(diffs))
        ax.axhspan(mean_t * 0.9, mean_t * 1.1, color=PALETTE["normal"], alpha=0.15,
                   label="Normal Variation (±10%)")
        ax.bar(x, diffs, color=color, alpha=0.65,
               edgecolor=PALETTE["border"], linewidth=0.8)
        ax.plot(x, diffs, color=PALETTE["warn"], linewidth=1.4, marker="o",
                markersize=5, alpha=0.9)
        ax.axhline(mean_t, color=PALETTE["accent"], linestyle="--",
                   linewidth=1.2, label=f"Mean: {mean_t:.3f} s")

        ax.set_xlabel("Step Interval Index", fontsize=10)
        ax.set_ylabel("Time (s)", fontsize=10)
        ax.set_title(f"{label} — Mean {mean_t:.3f} s",
                     color=PALETTE["text"], fontsize=11, pad=8)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=8, framealpha=0.6)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════
# PLOT 5 — TORSO LEAN
# ══════════════════════════════════════════════
def plot_torso_lean(features: dict, fps: float):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    for s in ax.spines.values():
        s.set_edgecolor(PALETTE["border"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.8)

    tl     = features["torso_lean"]
    angles = tl["angles"]
    mean_a = tl["mean"]
    std_a  = tl.get("std", 0.0)
    passes = features.get("pass_segments", [])
    times  = np.arange(len(angles)) / fps

    ax.fill_between(times, -5, 5, alpha=0.15, color=PALETTE["normal"],
                    label="Normal Range (±5°)")
    _shade_passes(ax, passes, fps)
    ax.plot(times, angles, color=PALETTE["warn"], linewidth=1.5, alpha=0.85,
            label="Torso Lean")
    ax.axhline(0, color=PALETTE["muted"], linestyle="-", linewidth=0.8)
    ax.axhline(mean_a, color=PALETTE["right"], linestyle="--",
               linewidth=1.3, label=f"Mean: {mean_a:.1f}° (σ={std_a:.1f}°)")

    ax.set_xlabel("Time (s)", fontsize=10, color=PALETTE["text"])
    ax.set_ylabel("Lean (°)", fontsize=10, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.set_title("Torso Lean Over Time (+ = forward in walking direction)",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
              fontsize=9, framealpha=0.6)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════
# PLOT 6 — ANKLE TRAJECTORY
# ══════════════════════════════════════════════
def plot_ankle_trajectory(features: dict, fps: float):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    for s in ax.spines.values():
        s.set_edgecolor(PALETTE["border"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.8)

    at     = features["ankle_trajectory"]
    passes = features.get("pass_segments", [])
    n      = max(len(at["left_x"]), len(at["right_x"]))
    times  = np.arange(n) / fps

    _shade_passes(ax, passes, fps)
    ax.plot(np.arange(len(at["left_x"]))  / fps, at["left_x"],
            color=PALETTE["left"],  linewidth=1.4, alpha=0.85, label="Left Ankle")
    ax.plot(np.arange(len(at["right_x"])) / fps, at["right_x"],
            color=PALETTE["right"], linewidth=1.4, alpha=0.85, label="Right Ankle")

    ax.fill_between(
        np.arange(min(len(at["left_x"]), len(at["right_x"]))) / fps,
        at["left_x"][:min(len(at["left_x"]), len(at["right_x"]))],
        at["right_x"][:min(len(at["left_x"]), len(at["right_x"]))],
        alpha=0.08, color=PALETTE["accent"],
    )

    ax.set_xlabel("Time (s)", fontsize=10, color=PALETTE["text"])
    ax.set_ylabel("Ankle X (image-normalized)", fontsize=10, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.set_title("Ankle X-Trajectory Over Time",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
    ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
              fontsize=9, framealpha=0.6)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════
# PLOT 7 — GAIT-CYCLE-NORMALIZED JOINT ANGLES (CLINICAL OUTPUT)
# ══════════════════════════════════════════════
_GAIT_CYCLE_ROWS = (
    # (title,         ylim)
    ("HIP ANGLE",    (-20, 40)),
    ("KNEE ANGLE",   (-5,  85)),
    ("ANKLE ANGLE",  (-25, 20)),
)

# ──────────────────────────────────────────────
# NORMAL ADULT REFERENCE BANDS
# ──────────────────────────────────────────────
# Mean ± 1 SD curves at each percent of the gait cycle for healthy adults
# walking at self-selected speed. Sourced from the canonical clinical-gait
# literature:
#   • Perry J, Burnfield JM. Gait Analysis: Normal and Pathological
#     Function (2nd ed., 2010).
#   • Winter DA. Biomechanics and Motor Control of Human Movement
#     (4th ed., 2009).
#   • Kadaba MP et al. "Measurement of lower extremity kinematics during
#     level walking." J Orthop Res 1990;8(3):383–392.
# SD values are conservative inter-subject estimates from the same sources.

NORMAL_BAND_COLOR = "#94a3b8"          # neutral grey, sits behind subject curves
NORMAL_BAND_ALPHA = 0.18               # low alpha — reads as background reference


def normal_hip_reference():
    """Healthy adult hip flexion/extension across gait cycle (Perry & Burnfield).
    Returns (mean, sd) each of length 101. Degrees.
    Positive = flexion, negative = extension."""
    x = np.linspace(0, 100, 101)
    mean = 10 + 20 * np.cos(2 * np.pi * x / 100)
    sd = np.full_like(mean, 4.0)
    return mean, sd


def normal_knee_reference():
    """Healthy adult knee flexion across gait cycle. Double-hump pattern.
    Loading peak ~18 deg at 15%, midstance ~5 deg, swing peak ~65 deg at 73%."""
    x = np.linspace(0, 100, 101)
    loading = 18 * np.exp(-((x - 15) ** 2) / (2 * 7 ** 2))
    swing   = 65 * np.exp(-((x - 73) ** 2) / (2 * 11 ** 2))
    mean = 3 + loading + swing
    sd = np.full_like(mean, 5.0)
    return mean, sd


def normal_ankle_reference():
    """Healthy adult ankle dorsi/plantarflexion across gait cycle.
    Loading dip, midstance dorsiflex ~+10, push-off plantarflex ~-18 at 62%."""
    x = np.linspace(0, 100, 101)
    loading_dip = -5  * np.exp(-((x - 5)  ** 2) / (2 * 4  ** 2))
    midst_df    = 10  * np.exp(-((x - 40) ** 2) / (2 * 15 ** 2))
    pushoff     = -18 * np.exp(-((x - 62) ** 2) / (2 * 6  ** 2))
    swing_rec   =  3  * np.exp(-((x - 85) ** 2) / (2 * 15 ** 2))
    mean = loading_dip + midst_df + pushoff + swing_rec
    sd = np.full_like(mean, 3.0)
    return mean, sd


_NORMAL_REFS = {
    "HIP ANGLE":   normal_hip_reference,
    "KNEE ANGLE":  normal_knee_reference,
    "ANKLE ANGLE": normal_ankle_reference,
}


def plot_gait_cycle_curves(
    hip_L_mean,   hip_L_std,   hip_R_mean,   hip_R_std,   K_hip_L,   K_hip_R,
    knee_L_mean,  knee_L_std,  knee_R_mean,  knee_R_std,  K_knee_L,  K_knee_R,
    ankle_L_mean, ankle_L_std, ankle_R_mean, ankle_R_std, K_ankle_L, K_ankle_R,
):
    """
    Three-row figure (hip, knee, ankle) showing mean ± 1 SD curves across
    the normalized gait cycle (0–100%) overlaid on the healthy-adult
    normal reference band (grey).

    Styling matches the rest of the dark-theme charts (PALETTE constants).
    Stance phase 0–60% lightly shaded; toe-off marker at 60%; 0° baseline.
    Y-axis limits per spec: hip [-20, 40], knee [-5, 85], ankle [-25, 20].
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), facecolor=PALETTE["bg"], sharex=True
    )

    rows_data = [
        ("HIP ANGLE",   hip_L_mean,   hip_L_std,   K_hip_L,
                        hip_R_mean,   hip_R_std,   K_hip_R,   (-20, 40)),
        ("KNEE ANGLE",  knee_L_mean,  knee_L_std,  K_knee_L,
                        knee_R_mean,  knee_R_std,  K_knee_R,  (-5,  85)),
        ("ANKLE ANGLE", ankle_L_mean, ankle_L_std, K_ankle_L,
                        ankle_R_mean, ankle_R_std, K_ankle_R, (-25, 20)),
    ]

    x = np.linspace(0, 100, 101)

    for ax, (title, mL, sL, KL, mR, sR, KR, ylim) in zip(axes, rows_data):
        ax.set_facecolor(PALETTE["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.8)

        # Stance phase 0-60%, swing 60-100% (unshaded). Lowest visual layer.
        ax.axvspan(0, 60, color=PALETTE["normal"], alpha=0.08,
                   label="Stance (0–60%)", zorder=0)

        # ---- Normal adult reference band (sits behind subject curves) ----
        ref_mean, ref_sd = _NORMAL_REFS[title]()
        ax.fill_between(
            x, ref_mean - ref_sd, ref_mean + ref_sd,
            color=NORMAL_BAND_COLOR, alpha=NORMAL_BAND_ALPHA,
            linewidth=0, zorder=1, label="Normal adult range",
        )
        ax.plot(
            x, ref_mean,
            color=NORMAL_BAND_COLOR, linewidth=1.2, linestyle="--",
            alpha=0.6, zorder=1,
        )

        # Toe-off + zero baseline above the normal band so they remain visible
        ax.axvline(60, color=PALETTE["muted"], linestyle="--",
                   linewidth=1.0, alpha=0.7, zorder=1.5)
        ax.axhline(0,  color=PALETTE["muted"], linestyle="-",
                   linewidth=0.6, alpha=0.5, zorder=1.5)

        # LEFT leg (blue) — subject curves on top
        if mL is not None and KL > 0 and len(mL) == len(x):
            if sL is not None:
                ax.fill_between(x, mL - sL, mL + sL,
                                color=PALETTE["left"], alpha=0.20,
                                linewidth=0, zorder=2)
            ax.plot(x, mL, color=PALETTE["left"], linewidth=2.0,
                    label=f"Left (K={KL})", zorder=3)

        # RIGHT leg (red) — subject curves on top
        if mR is not None and KR > 0 and len(mR) == len(x):
            if sR is not None:
                ax.fill_between(x, mR - sR, mR + sR,
                                color=PALETTE["right"], alpha=0.20,
                                linewidth=0, zorder=2)
            ax.plot(x, mR, color=PALETTE["right"], linewidth=2.0,
                    label=f"Right (K={KR})", zorder=3)

        ax.set_ylim(ylim)
        ax.set_xlim(0, 100)
        ax.set_ylabel("Angle (°)", fontsize=10, color=PALETTE["text"])
        ax.set_title(f"{title} — mean ± 1 SD",
                     color=PALETTE["text"], fontsize=11, pad=8)
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"],
                  fontsize=8, framealpha=0.6, loc="upper right")

    axes[-1].set_xlabel("Gait Cycle (%)", fontsize=10, color=PALETTE["text"])
    fig.suptitle("Gait-Cycle-Normalized Joint Angles",
                 color=PALETTE["text"], fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════
# CONVENIENCE BUILDER
# ══════════════════════════════════════════════
def _build_cycle_figure_if_ready(features: dict):
    """Return a gait-cycle figure when at least 3 cycles per knee leg are
    available; otherwise return None so the calling tab can show an info box."""
    gc = features.get("gait_cycle_curves")
    if not gc:
        return None
    knee = gc.get("knee", {})
    KL = knee.get("left", {}).get("K", 0)
    KR = knee.get("right", {}).get("K", 0)
    if min(KL, KR) < 3:
        return None
    return plot_gait_cycle_curves(
        gc["hip"]["left"]["mean"],   gc["hip"]["left"]["std"],
        gc["hip"]["right"]["mean"],  gc["hip"]["right"]["std"],
        gc["hip"]["left"]["K"],      gc["hip"]["right"]["K"],
        gc["knee"]["left"]["mean"],  gc["knee"]["left"]["std"],
        gc["knee"]["right"]["mean"], gc["knee"]["right"]["std"],
        gc["knee"]["left"]["K"],     gc["knee"]["right"]["K"],
        gc["ankle"]["left"]["mean"], gc["ankle"]["left"]["std"],
        gc["ankle"]["right"]["mean"],gc["ankle"]["right"]["std"],
        gc["ankle"]["left"]["K"],    gc["ankle"]["right"]["K"],
    )


def build_all_figures(features: dict) -> dict:
    fps = features.get("fps", 30.0)
    figs = {
        "knee":   plot_knee_angles(features, fps),
        "heel":   plot_heel_trajectory(features, fps),
        "step":   plot_step_length(features),
        "timing": plot_timing(features),
        "torso":  plot_torso_lean(features, fps),
        "ankle":  plot_ankle_trajectory(features, fps),
    }
    cycle_fig = _build_cycle_figure_if_ready(features)
    if cycle_fig is not None:
        figs["cycle"] = cycle_fig
    return figs
