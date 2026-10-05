"""Condition-level appendix figures drawn from the main analyses' result tables."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from analysis.plotting import (
    interval_point, MODEL_FAMILIES, checkpoint_legend, model_styles, figure_style,
    MODEL_TYPE_COLORS, INK, marker_area, to_rgba
)


PANELS = (("thatcher", "str", "Upright", "#0072B2"), ("thatcher", "inv", "Inverted", "#D55E00"),
          ("swap", "shadow_swap", "Shadow", "#0072B2"), ("swap", "reflection_swap", "Reflection", "#CC79A7"),
          ("swap", "lightdir_swap", "Light direction", "#009E73"))
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


def save(figure, output_dir, stem):
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"{stem}.{extension}", dpi=240, facecolor="white")
    plt.close(figure)


def family_legend(figure, extra=()):
    handles = [Line2D([], [], marker=marker, color=color, lw=0, markersize=4, label=name)
               for name, (color, marker, _) in MODEL_FAMILIES.items()]
    figure.legend(handles=[*handles, *extra], loc="lower center", bbox_to_anchor=(.5, .025),
                  ncol=6, frameon=False, fontsize=7, handletextpad=.5, columnspacing=1.5)


def plot_sensitivity_alignment_heatmap(result, output_dir):
    """Draw both measures already computed together by sensitivity_alignment."""
    styles = model_styles()
    order = list(styles)
    conditions = [(task, condition) for task, condition, _, _ in PANELS]
    models = result["condition_model_metrics"].set_index(["model", "task", "condition"], verify_integrity=True)
    humans = result["condition_human_references"].set_index(["task", "condition"], verify_integrity=True)
    order = [m for m in order if all((m, *condition) in models.index for condition in conditions)]
    matrices = []
    for metric in ("sensitivity", "alignment"):
        values = np.array([[models.loc[(model, *condition), metric] for condition in conditions] for model in order])
        reference = np.array([humans.loc[condition, metric] for condition in conditions])
        if not np.isfinite(values).all() or not np.isfinite(reference).all():
            raise ValueError(f"Heatmap requires finite {metric} estimates.")
        matrices.append(np.vstack([reference, np.median(values, axis=0), np.full((1, 5), np.nan), values]))
    row_names = ["human_reference", "model_median", ""] + order
    row_labels = ["Human reference", "Model median", ""] + [MODEL_LABELS[model] for model in order]
    records = []
    with plt.rc_context({**STYLE, "font.size": 6}):
        figure = plt.figure(figsize=(5.5, 5.55))
        left, width, gap = .282, .3385, .035
        x_positions = np.array([0., 1., 2.18, 3.18, 4.18])
        cmap = plt.get_cmap("RdBu")
        for panel, (metric, matrix, limit, title, ticks, bar_label) in enumerate(zip(
            ("sensitivity", "alignment"), matrices, (3.5, 1.),
            ("A  Directional sensitivity", "B  Human alignment"),
            ((-3.5, -2, 0, 2, 3.5), (-1, -.5, 0, .5, 1)),
            (r"Task-scaled sensitivity $S_{m,c}$", r"Human alignment (Pearson $r$)"),
        )):
            norm = Normalize(vmin=-limit, vmax=limit)
            axis = figure.add_axes([left + panel * (width + gap), .145, width, .715])
            for row, values in enumerate(matrix):
                if not row_names[row]:
                    continue
                for column, value in enumerate(values):
                    color = cmap(norm(np.clip(value, -limit, limit)))
                    label = "0.00" if abs(value) < .005 else f"{value:.2f}"
                    axis.add_patch(Rectangle((x_positions[column] - .5, row - .5), 1, 1,
                                             facecolor=color, edgecolor="none"))
                    axis.text(x_positions[column], row, label, ha="center", va="center",
                              fontsize=6, fontweight="bold" if row < 2 else "normal",
                              color="white" if np.dot(color[:3], [.2126, .7152, .0722]) < .48 else "#111827")
                    task, condition = conditions[column]
                    records.append({"metric": metric, "row": row_names[row], "task": task,
                                    "condition": condition, "value": float(value), "label": label})
            for i in range(1, len(order)):
                before, after = styles[order[i - 1]], styles[order[i]]
                if before["family"] != after["family"]:
                    modality_change = (before["marker"] == "^") != (after["marker"] == "^")
                    axis.axhline(i + 2.5, color="white", linewidth=1.5)
                    axis.axhline(i + 2.5, color="#6B7280" if modality_change else "#D1D5DB",
                                 linewidth=.8 if modality_change else .35)
            axis.axhline(2.5, color="#6B7280", linewidth=.65)
            axis.set(xlim=(-.5, 4.68), ylim=(len(matrix) - .5, -.5),
                     xticks=x_positions, xticklabels=["Upright", "Inverted", "Shadow", "Reflection", "Light\ndirection"],
                     yticks=np.arange(len(matrix)), yticklabels=row_labels if panel == 0 else [""] * len(matrix))
            axis.xaxis.tick_top()
            axis.tick_params(axis="x", length=0, pad=3, labelsize=5.1)
            axis.tick_params(axis="y", length=0, pad=4, labelsize=6.1)
            for label in axis.get_yticklabels()[:2]:
                label.set_fontweight("bold")
            axis.spines[:].set_visible(False)
            for center, task_label in ((.5, "Thatcher"), (3.18, "Illumination")):
                axis.text(center, 1.075, task_label, transform=axis.get_xaxis_transform(),
                          ha="center", va="bottom", fontsize=6.8)
            axis.text(0, 1.14, title, transform=axis.transAxes, ha="left", va="bottom", fontsize=7.5, fontweight="bold")
            bar_axis = figure.add_axes([left + panel * (width + gap), .075, width, .018])
            bar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar_axis,
                                  orientation="horizontal", ticks=ticks)
            bar.set_ticklabels([f"{value:g}" for value in ticks])
            bar.set_label(bar_label, fontsize=6.4, labelpad=3)
            bar.ax.tick_params(labelsize=5.8, length=2, pad=2)
            bar.outline.set_linewidth(.4)
        for extension in ("pdf", "png"):
            figure.savefig(output_dir / f"condition_sensitivity_alignment_heatmap.{extension}",
                           dpi=400, bbox_inches="tight", pad_inches=.035, facecolor="white")
        plt.close(figure)
    pd.DataFrame(records).to_csv(output_dir / "condition_heatmap_values.csv", index=False)


def plot_condition_alignment(result, output_dir):
    with plt.rc_context({**STYLE, "font.size": 8.3, "axes.titlesize": 8.8,
                         "axes.labelsize": 8.5, "xtick.labelsize": 7.8,
                         "ytick.labelsize": 7.8, "axes.linewidth": .7}):
        figure = plt.figure(figsize=(7.05, 5.25))
        grid = figure.add_gridspec(2, 3, left=.085, right=.985, bottom=.105, top=.915,
                                  hspace=.50, wspace=.25)
        axes = [figure.add_subplot(grid[row, column])
                for row, column in ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2))]
        limits = {}
        for task in ("thatcher", "swap"):
            points = result["condition_model_metrics"].query("task == @task")
            humans = result["condition_human_references"].query("task == @task")
            x = np.concatenate([points.sensitivity_ci_low, points.sensitivity_ci_high,
                                humans.sensitivity_ci_low, humans.sensitivity_ci_high])
            y = np.concatenate([points.alignment_ci_low, points.alignment_ci_high,
                                humans.alignment_ci_low, humans.alignment_ci_high, [0.]])
            xpad, ypad = max(.12, .04 * np.ptp(x)), max(.04, .05 * np.ptp(y))
            limits[task] = ((x.min() - xpad, x.max() + xpad), (y.min() - ypad, y.max() + ypad))
        y_limits = (min(bounds[1][0] for bounds in limits.values()), max(bounds[1][1] for bounds in limits.values()))

        for i, (axis, (task, condition, label, _)) in enumerate(zip(axes, PANELS)):
            points = result["condition_model_metrics"].query("task == @task and condition == @condition")
            human = result["condition_human_references"].query("task == @task and condition == @condition").iloc[0]
            association = result["condition_associations"].query("task == @task and condition == @condition").iloc[0]

            # Human reference shaded bands & crosshairs
            axis.axhspan(human.alignment_ci_low, human.alignment_ci_high, color=INK, alpha=.04, lw=0, zorder=0)
            axis.axvspan(human.sensitivity_ci_low, human.sensitivity_ci_high, color=INK, alpha=.04, lw=0, zorder=0)
            axis.axhline(human.alignment, color=INK, ls=(0, (4, 2)), lw=.65, alpha=.8, zorder=1)
            axis.axvline(human.sensitivity, color=INK, ls=(0, (4, 2)), lw=.65, alpha=.8, zorder=1)

            # Regression line
            x_line = np.linspace(points.sensitivity.min(), points.sensitivity.max(), 100)
            axis.plot(x_line, association.intercept + association.slope * x_line, color="#64748B", lw=1, alpha=.75, zorder=2)

            # Model error bars
            for row in points.itertuples():
                color = MODEL_TYPE_COLORS.get(row.model_type, "#0072B2")
                axis.hlines(row.alignment, row.sensitivity_ci_low, row.sensitivity_ci_high,
                            color=color, lw=.55, alpha=.24, zorder=2)
                axis.vlines(row.sensitivity, row.alignment_ci_low, row.alignment_ci_high,
                            color=color, lw=.55, alpha=.24, zorder=2)

            # Model scatter circles (larger first, so smaller appear on top)
            for row in points.sort_values("parameter_b", ascending=False).itertuples():
                color = MODEL_TYPE_COLORS.get(row.model_type, "#0072B2")
                axis.scatter(row.sensitivity, row.alignment, s=row.marker_area_pt2,
                             facecolors=to_rgba(color, .08), edgecolors=color,
                             linewidths=.8, zorder=3)
            axis.scatter(points.sensitivity, points.alignment, s=2.5,
                         c=points.model_type.map(MODEL_TYPE_COLORS), linewidths=0, zorder=4)

            # Human diamond
            axis.hlines(human.alignment, human.sensitivity_ci_low, human.sensitivity_ci_high, color=INK, lw=.8, zorder=5)
            axis.vlines(human.sensitivity, human.alignment_ci_low, human.alignment_ci_high, color=INK, lw=.8, zorder=5)
            axis.scatter(human.sensitivity, human.alignment, s=36, marker="D",
                         color=INK, edgecolors="white", linewidths=.5, zorder=6)

            axis.set(xlim=limits[task][0], ylim=y_limits, box_aspect=1)
            axis.set_title(f"{chr(65+i)}  {label}", loc="left", fontweight="bold", pad=4)
            axis.text(.97, .045, f"$r_{{models}}={association.pearson_r:.3f}$", transform=axis.transAxes,
                      ha="right", va="bottom", fontsize=7.4, color=INK,
                      bbox={"facecolor": "white", "edgecolor": "#E2E8F0", "alpha": .90, "pad": 1.5, "linewidth": 0.5})
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#64748B")
            axis.axhline(0, color="#94A3B8", lw=.55, ls=(0, (2, 2)), zorder=0)
            axis.grid(True, color="#E5E7EB", lw=.5, alpha=.8, zorder=0)
            axis.set_ylabel("Human alignment $r$" if i in (0, 2) else "")
            axis.tick_params(length=2.5, pad=2)
            axis.tick_params(axis="y", left=True, labelleft=i in (0, 2))

        # Dedicated clean legend in top-right slot (grid[0, 2])
        legend_axis = figure.add_subplot(grid[0, 2])
        legend_axis.set_axis_off()
        legend_axis.set(xlim=(0, 1), ylim=(0, 1))

        legend_axis.text(0.02, 0.92, "Model type", fontsize=8.2, fontweight="bold", color=INK, va="center")
        y_type = 0.77
        for x, (type_name, color) in zip((0.08, 0.40, 0.74), MODEL_TYPE_COLORS.items()):
            legend_axis.scatter(x, y_type, s=24, facecolors=to_rgba(color, .08), edgecolors=color, linewidths=0.8)
            legend_axis.scatter(x, y_type, s=2.5, color=color, linewidths=0)
            legend_axis.text(x + 0.08, y_type, type_name, fontsize=7.8, color=INK, va="center")

        legend_axis.text(0.02, 0.58, "Size (B; log scale)", fontsize=8.2, fontweight="bold", color=INK, va="center")
        y_size = 0.42
        for x, size in zip((0.08, 0.32, 0.56, 0.82), (0.1, 1, 5, 20)):
            legend_axis.scatter(x, y_size, s=marker_area(size), facecolors="none", edgecolors="#586674", linewidths=0.8)
            legend_axis.scatter(x, y_size, s=2.5, color="#586674", linewidths=0)
            legend_axis.text(x + 0.08, y_size, f"{size:g}", fontsize=7.8, color=INK, va="center")

        legend_axis.text(0.02, 0.23, "Reference", fontsize=8.2, fontweight="bold", color=INK, va="center")
        y_ref = 0.08
        legend_axis.scatter(0.08, y_ref, s=36, marker="D", color=INK, edgecolors="white", linewidths=0.6, zorder=5)
        legend_axis.text(0.16, y_ref, "Human", fontsize=8.0, color=INK, va="center")

        figure.text(.085, .982, "Thatcher", ha="left", va="top", fontsize=10, fontweight="bold")
        figure.text(.085, .497, "Illumination", ha="left", va="top", fontsize=10, fontweight="bold")
        figure.text(.535, .025, "Sensitivity", ha="center", va="bottom", fontsize=8.7)
        for extension in ("pdf", "png"):
            figure.savefig(output_dir / f"condition_sensitivity_alignment.{extension}", dpi=400, facecolor="white")
        plt.close(figure)


def plot_condition_figures(result, output_dir):
    """Render condition-level heatmap and scatter figures (Figures S11, S13)."""
    plot_sensitivity_alignment_heatmap(result, output_dir)
    plot_condition_alignment(result, output_dir)


def plot_alignment_appendices(result, output_dir):
    """Backward-compatible alias for plot_condition_figures."""
    plot_condition_figures(result, output_dir)

