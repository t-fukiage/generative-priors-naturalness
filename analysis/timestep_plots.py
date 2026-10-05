"""Timestep dynamics and peak position figures for generative diffusion models.

Produces Figure 5 (Main text) and Appendix Figures S14-S16:
- Figure 5: Schedule dynamics for Wan2.1 T2V 14B and cross-model alignment vs. sensitivity peaks
- Figure S14: Condition-specific peak positions across 25 models
- Figure S15 & S16: Condition-specific timestep profiles for each of the 25 models (Thatcher and Illumination)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis.plotting import (
    figure_style, model_styles, MODEL_TYPE_COLORS, INK
)


PANELS = (
    ("thatcher", "str", "Upright", "#0072B2"),
    ("thatcher", "inv", "Inverted", "#D55E00"),
    ("swap", "shadow_swap", "Shadow", "#0072B2"),
    ("swap", "reflection_swap", "Reflection", "#CC79A7"),
    ("swap", "lightdir_swap", "Light direction", "#009E73"),
)

MODEL_LABELS = {
    "jit_b32": "JiT-B/32", "jit_l32": "JiT-L/32", "jit_h32": "JiT-H/32", "pixelgen_512": "PixelGen 512",
    "hidream_o1": "HiDream-O1", "stable_diffusion_15": "SD v1.5", "sdxl": "SDXL", "stable_diffusion_3": "SD3 Medium",
    "flux1_dev": "FLUX.1 dev", "flux1_schnell": "FLUX.1 schnell", "qwen_image": "Qwen-Image",
    "qwen_image_2512": "Qwen-Image-2512", "flux2_klein_base_4b": "FLUX.2 klein base 4B", "flux2_klein_4b": "FLUX.2 klein 4B",
    "flux2_klein_base_9b": "FLUX.2 klein base 9B", "flux2_klein_9b": "FLUX.2 klein 9B", "cogvideox_2b": "CogVideoX 2B",
    "cogvideox_5b": "CogVideoX 5B", "cogvideox15_5b": "CogVideoX1.5 5B", "ltx_2b": "LTX-Video 2B",
    "ltx_13b": "LTX-Video 13B dev", "hunyuan_video": "HunyuanVideo", "wan21_1_3b": "Wan2.1 T2V 1.3B",
    "wan21_14b": "Wan2.1 T2V 14B", "wan22_a14b": "Wan2.2 T2V A14B",
}

STYLE = figure_style(fontsize=7)


def save(figure, output_dir: Path, stem: str):
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


def plot_timestep_profiles_and_peaks(result: dict, output_dir: Path):
    """Render Figure 5: Timestep profiles (Wan2.1) and peak positions (all 25 models)."""
    with plt.rc_context(figure_style(fontsize=7)):
        fig = plt.figure(figsize=(5.5, 1.28), facecolor="white")

        def axes(x, y, width, height):
            figure_height = fig.get_size_inches()[1]
            return fig.add_axes([x / 5.5, y / figure_height, width / 5.5, height / figure_height])

        def text(x, y, label, **kwargs):
            fig.text(x / 5.5, y / fig.get_size_inches()[1], label, va="center", color="#182536", **kwargs)

        text(0.24, 1.21, "A", fontsize=9.5, fontweight="bold")
        text(1.43, 1.21, "Wan2.1 T2V 14B", fontsize=6.8, ha="center")
        text(3.12, 1.21, "B", fontsize=9.5, fontweight="bold")
        for index, (task, label) in enumerate((("thatcher", "Thatcher"), ("swap", "Illumination"))):
            axis = axes((0.385, 1.623)[index], 0.26, 0.864, 0.78)
            curve = result["timestep_profiles"].query("task == @task and model == 'wan21_14b'").sort_values("timestep_index")
            twin = axis.twinx()
            peak_row = result["timestep_peaks"].query("task == @task and model == 'wan21_14b'").iloc[0]
            for local, column, color, linestyle in ((axis, "alignment", "#0072B2", "-"), (twin, "sensitivity", "#D55E00", (0, (3, 1.8)))):
                values = curve[column].to_numpy()
                peak = int(peak_row[f"{column}_peak_index"])
                local.plot(curve.schedule_position, values, color=color, linewidth=1, linestyle=linestyle)
                local.scatter(curve.schedule_position.iloc[peak], values[peak], color=color, s=10, linewidths=0, zorder=5)
                local.axvline(curve.schedule_position.iloc[peak], color=color, linewidth=0.55, linestyle=":", alpha=0.75)
            axis.set(xlim=(0, 1), ylim=(-0.2, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
            twin.set(ylim=(-0.6, 3), yticks=[0, 1, 2, 3])
            axis.set_title(label, loc="center", fontsize=7.2, fontweight="bold", pad=2.5)
            axis.grid(color="#E5E7EB", linewidth=0.35, zorder=0)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["bottom", "left"]].set_linewidth(0.5)
            axis.spines["left"].set_color("#0072B2")
            twin.spines[["top", "left", "bottom"]].set_visible(False)
            twin.spines["right"].set(color="#D55E00", linewidth=0.5)
            axis.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5)
            axis.tick_params(axis="y", colors="#0072B2")
            twin.tick_params(axis="y", colors="#D55E00", labelsize=6.5, length=2, width=0.5, pad=1.5)
            if index == 0:
                axis.set_ylabel("Alignment $r$", fontsize=6.5, color="#0072B2")
                twin.tick_params(right=False, labelright=False)
                twin.spines["right"].set_visible(False)
            else:
                axis.tick_params(labelleft=False)
                twin.set_ylabel("Sensitivity", fontsize=6.5, color="#D55E00")

            scatter = axes((3.421, 4.560)[index], 0.26, 0.864, 0.78)
            scatter.plot([0, 1], [0, 1], color="#9CA3AF", linewidth=0.6, linestyle="--")
            peaks = result["timestep_peaks"].query("task == @task")
            for row in peaks.itertuples():
                color = MODEL_TYPE_COLORS.get(getattr(row, "model_type", None), "#0072B2")
                scatter.scatter(row.alignment_peak_position, row.sensitivity_peak_position, marker="o",
                                color=[color], s=14, edgecolor="white", linewidth=0.3, zorder=3)
            example = peaks[peaks.model == "wan21_14b"].iloc[0]
            scatter.scatter(example.alignment_peak_position, example.sensitivity_peak_position, marker="o",
                            s=36, facecolor="none", edgecolor="#182536", linewidth=0.85, zorder=4)
            scatter.set(xlim=(-0.045, 1.045), ylim=(-0.045, 1.045), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
            scatter.set_title(label, loc="center", fontsize=7.2, fontweight="bold", pad=2.5)
            scatter.spines[["top", "right"]].set_visible(False)
            scatter.spines[["left", "bottom"]].set_linewidth(0.5)
            scatter.tick_params(labelsize=6.5, length=2, width=0.5, pad=1.5)
            scatter.grid(color="#E5E7EB", linewidth=0.35, zorder=0)
            scatter.set_axisbelow(True)
            if index == 0:
                scatter.set_ylabel("Sensitivity peak", fontsize=6.5, labelpad=2)
            else:
                scatter.tick_params(labelleft=False)

        text(1.43, 0.055, "Schedule position", fontsize=6.5, ha="center")
        text(4.41, 0.055, "Alignment peak position", fontsize=6.5, ha="center")
        save(fig, output_dir, "timestep_profiles_and_peaks")


def plot_condition_peaks(result: dict, output_dir: Path):
    """Render Appendix Figure S14: Condition-specific peak positions across 25 models."""
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 5, figsize=(11.8, 3.5))
        figure.subplots_adjust(left=.052, right=.99, bottom=.34, top=.87, wspace=.32)
        for i, (axis, (task, condition, label, _)) in enumerate(zip(axes, PANELS)):
            points = result["condition_timestep_peaks"].query("task == @task and condition == @condition")
            summary = result["condition_peak_summary"].query("task == @task and condition == @condition").iloc[0]
            for row in points.itertuples():
                color = MODEL_TYPE_COLORS.get(getattr(row, "model_type", None), "#0072B2")
                axis.scatter(row.alignment_peak_position, row.sensitivity_peak_position, marker="o",
                             color=color, s=24, edgecolor="white", linewidth=.4, zorder=3)
            axis.plot([0, 1], [0, 1], "--", color="#888888", lw=.7)
            axis.set(xlim=(-.045, 1.045), ylim=(-.045, 1.045), xticks=[0, .5, 1], yticks=[0, .5, 1],
                     xlabel="Alignment peak", title=f"{chr(65+i)}  {label}", aspect="equal")
            axis.text(.5, -.32, f"Sensitivity later: {int(summary.sensitivity_peak_later)}/25\nMedian shift: {summary.peak_difference_median:+.3f}",
                      ha="center", va="top", transform=axis.transAxes, fontsize=7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(length=2.5, width=.6)
        axes[0].set_ylabel("Sensitivity peak")
        figure.suptitle("Condition-specific peak positions on each model's saved schedule", fontsize=10, fontweight="bold", y=.975)
        handles = [Line2D([], [], marker="o", color=color, lw=0, markersize=5, label=name)
                   for name, color in MODEL_TYPE_COLORS.items()]
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035),
                      ncol=3, frameon=False, fontsize=7.5, handletextpad=.5, columnspacing=2.5)
        figure.text(.5, .003, "Position = saved index / (T − 1). Signed maxima; first exact maximum on ties. Descriptive points, without CIs.",
                    ha="center", fontsize=7)
        save(figure, output_dir, "condition_timestep_peaks")


def plot_condition_profiles(result: dict, output_dir: Path, task: str):
    """Render Appendix Figures S15 & S16: Condition-specific timestep profiles for each of the 25 models."""
    styles = model_styles()
    profile = result["condition_timestep_profiles"].query("task == @task")
    peaks = result["condition_timestep_peaks"].query("task == @task").set_index(["model", "condition"])
    conditions = [panel for panel in PANELS if panel[0] == task]
    sensitivity_limit = max(1., np.ceil(np.nanmax(abs(profile.sensitivity)) * 1.05 * 2) / 2)
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(5, 5, figsize=(10.5, 11.5))
        figure.subplots_adjust(left=.07, right=.925, bottom=.075, top=.905, wspace=.24, hspace=.48)
        for i, (model, style) in enumerate(styles.items()):
            axis = axes.flat[i]
            twin = axis.twinx()
            for _, condition, _, color in conditions:
                curve = profile[(profile.model == model) & (profile.condition == condition)].sort_values("timestep_index")
                axis.plot(curve.schedule_position, curve.alignment, color=color, lw=.8)
                twin.plot(curve.schedule_position, curve.sensitivity, color=color, lw=.8, ls="--")
                peak = peaks.loc[(model, condition)]
                axis.plot(peak.alignment_peak_position, peak.alignment_peak_r, "o", color=color, ms=3, mec="white", mew=.35)
                twin.plot(peak.sensitivity_peak_position, peak.sensitivity_peak_value, "o", color=color, ms=3, mfc="white", mew=.7)
            axis.set(xlim=(-.025, 1.025), ylim=(-1.05, 1.05), xticks=[0, .5, 1], yticks=[-1, 0, 1])
            twin.set(ylim=(-sensitivity_limit, sensitivity_limit), yticks=[-sensitivity_limit, 0, sensitivity_limit])
            axis.axhline(0, color="#cccccc", lw=.5, zorder=0)
            axis.plot([0, 1], [1.08, 1.08], transform=axis.transAxes, color=style["color"], lw=3, clip_on=False)
            axis.set_title(MODEL_LABELS[model], fontsize=7, pad=13)
            axis.spines["top"].set_visible(False)
            twin.spines[["top", "left", "bottom"]].set_visible(False)
            axis.tick_params(labelsize=6.5, length=2, width=.5, labelleft=i % 5 == 0)
            twin.tick_params(labelsize=6.5, length=2, width=.5, labelright=i % 5 == 4)
            if i % 5 == 0:
                axis.set_ylabel("Alignment r", fontsize=7)
            if i % 5 == 4:
                twin.set_ylabel("Sensitivity", fontsize=7)
        title = "Thatcher" if task == "thatcher" else "Illumination"
        figure.suptitle(f"{title}: condition-specific timestep profiles for all 25 models", y=.985, fontsize=11, fontweight="bold")
        handles = [Line2D([], [], color=color, lw=1, label=label) for _, _, label, color in conditions]
        handles.extend([Line2D([], [], color="#444444", marker="o", ms=4, lw=.8, label="Alignment (left)"),
                        Line2D([], [], color="#444444", marker="o", mfc="white", ms=4, ls="--", lw=.8, label="Sensitivity (right)")])
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .961), ncol=len(handles), frameon=False, fontsize=8)
        figure.text(.5, .035, "Saved schedule position: index / (T − 1)", ha="center", fontsize=9)
        figure.text(.5, .012, "Shared raw axis limits and aligned zeros within each task. Circles mark discrete peaks; no smoothing or interpolation.",
                    ha="center", fontsize=7)
        save(figure, output_dir, f"condition_timestep_profiles_{task}")


def plot_timestep_figures(result: dict, output_dir: Path):
    """Generate all timestep-related figures (Figure 5, S14, S15, S16)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_timestep_profiles_and_peaks(result, output_dir)
    plot_condition_peaks(result, output_dir)
    for task in ("thatcher", "swap"):
        plot_condition_profiles(result, output_dir, task)


if __name__ == "__main__":
    from data_io import configured_path
    from analysis.sensitivity_alignment import analyze

    parser = argparse.ArgumentParser(description="Render timestep-related figures (Figure 5, S14, S15, S16).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Target directory for generated figures")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Data root directory containing input tables")
    args = parser.parse_args()

    target_dir = args.output_dir or (configured_path("output_root") / "analysis/sensitivity_alignment")
    results = analyze(data_root=args.data_root)
    plot_timestep_figures(results, target_dir)
    print(f"Timestep figures saved to: {target_dir}")
