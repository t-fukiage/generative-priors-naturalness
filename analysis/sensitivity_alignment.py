"""Generate cross-model sensitivity and human alignment figures."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from data_io import configured_path, load_generative_means, load_human_pairs, load_human_split_half, paired_scores
from analysis.condition_plots import plot_alignment_appendices, plot_condition_figures
from analysis.metrics import center_within, pearson, sensitivity
from analysis.plotting import (
    interval_point, checkpoint_legend, model_styles, figure_style,
    MODEL_TYPE_COLORS, INK, marker_area, load_generative_model_metadata
)
from analysis.uncertainty import apply_intervals


TASKS = ("thatcher", "swap")
CONDITIONS = {"thatcher": ("str", "inv"), "swap": ("shadow_swap", "reflection_swap", "lightdir_swap")}


def scope_metrics(scores, human, mask):
    """Condition means / task SD, and Pearson r within the selected condition."""
    return {"sensitivity": sensitivity(scores, mask), "alignment": pearson(scores[mask], human[mask]),
            "human_sensitivity": sensitivity(human, mask)}


def profile_peaks(profile):
    """Signed maxima on each saved grid; first exact maximum on ties, ignoring NaN."""
    rows = []
    keys = ["task", "model"] if "condition" not in profile else ["task", "condition", "model"]
    for key, frame in profile.groupby(keys, sort=False):
        frame = frame.sort_values("timestep_index")
        alignment, effect = frame.alignment.to_numpy(), frame.sensitivity.to_numpy()
        a, s = int(np.nanargmax(alignment)), int(np.nanargmax(effect))
        n = len(frame)
        rows.append({
            **dict(zip(keys, key)), "n_timesteps": n,
            "alignment_peak_index": a, "sensitivity_peak_index": s,
            "alignment_peak_position": a / (n - 1), "sensitivity_peak_position": s / (n - 1),
            "peak_position_difference": (s - a) / (n - 1),
            "alignment_peak_r": alignment[a], "alignment_r_at_sensitivity_peak": alignment[s],
            "alignment_drop_at_sensitivity_peak": alignment[a] - alignment[s],
            "sensitivity_peak_value": effect[s], "sensitivity_at_alignment_peak": effect[a],
            "alignment_tied_maxima": int(np.sum(alignment == alignment[a])),
            "sensitivity_tied_maxima": int(np.sum(effect == effect[s])),
            "alignment_missing_points": int(np.isnan(alignment).sum()),
            "sensitivity_missing_points": int(np.isnan(effect).sum()),
            "alignment_peak_at_boundary": a in (0, n - 1), "sensitivity_peak_at_boundary": s in (0, n - 1),
        })
    return pd.DataFrame(rows)


def _split_half_reference(split_half, task, scope):
    """Retrieve the unique human split-half record for a task and scope."""
    reference = split_half[(split_half.task == task) & (split_half.scope == scope)]
    if len(reference) != 1:
        raise ValueError(f"Missing {scope} split-half reference: {task}/{scope}")
    return reference.iloc[0]


def _fit_association(table, **extra):
    """Linear regression and Pearson r between sensitivity and alignment."""
    slope, intercept = np.polyfit(table.sensitivity, table.alignment, 1)
    return {
        **extra,
        "n_models": len(table),
        "pearson_r": float(pearson(table.sensitivity, table.alignment)),
        "slope": slope,
        "intercept": intercept,
    }


def analyze(items=None, losses=None, models=None, split_half=None, data_root=None):
    data_root = configured_path("data_root", data_root)
    if items is None:
        items = load_human_pairs(data_root=data_root)
    if losses is None or models is None:
        losses, models = load_generative_means(items, data_root=data_root)
    if split_half is None:
        split_half = load_human_split_half(data_root=data_root)
    model_names = sorted(losses)
    scores = np.column_stack([paired_scores({"loss_mean": losses[model]}) for model in model_names])

    points, profiles, single_images, human_rows = [], [], [], []
    condition_points, condition_profiles, condition_humans = [], [], []

    for task in TASKS:
        selected = items.task.eq(task).to_numpy()
        local = items.loc[selected]
        x, y = scores[selected], local.score.to_numpy()
        groups, conditions = local.base_scene_group.to_numpy(), local.condition.to_numpy()
        masks = {
            "pooled": np.ones(len(local), dtype=bool),
            **{condition: conditions == condition for condition in CONDITIONS[task]},
        }

        # 1. Condition-specific model and human metrics
        counts_task = {"n_task_pairs": len(local), "n_base_scene_groups": len(np.unique(groups))}
        for condition in CONDITIONS[task]:
            mask = masks[condition]
            estimates = scope_metrics(x, y, mask)
            counts = {"n_pairs": int(mask.sum()), **counts_task}

            for column, model in enumerate(model_names):
                condition_points.append({
                    "task": task, "condition": condition, "model": model,
                    **counts,
                    "sensitivity": estimates["sensitivity"][column],
                    "sensitivity_ci_low": np.nan, "sensitivity_ci_high": np.nan, "sensitivity_valid_draws": 0,
                    "alignment": estimates["alignment"][column],
                    "alignment_ci_low": np.nan, "alignment_ci_high": np.nan, "alignment_valid_draws": 0,
                })

            reference = _split_half_reference(split_half, task, condition)
            condition_humans.append({
                "task": task, "condition": condition,
                **counts,
                "sensitivity": estimates["human_sensitivity"],
                "sensitivity_ci_low": np.nan, "sensitivity_ci_high": np.nan, "sensitivity_valid_draws": 0,
                "alignment": reference.r_mean,
                "alignment_ci_low": reference.ci_low,
                "alignment_ci_high": reference.ci_high,
                "n_splits": int(reference.n_splits),
            })

        # 2. Task-pooled model metrics, timestep profiles, and single-image alignment
        observed = scope_metrics(x, y, masks["pooled"])
        centered_alignments = pearson(center_within(x, conditions), center_within(y, conditions))

        for column, model in enumerate(model_names):
            points.append({
                "task": task, "model": model,
                "sensitivity": float(observed["sensitivity"][column]),
                "alignment": float(observed["alignment"][column]),
                "sensitivity_ci_low": np.nan, "sensitivity_ci_high": np.nan,
                "alignment_ci_low": np.nan, "alignment_ci_high": np.nan,
                "within_condition_alignment": float(centered_alignments[column]),
                "n_pairs": len(x),
                "n_base_scene_groups": counts_task["n_base_scene_groups"],
                "sensitivity_valid_draws": 0, "alignment_valid_draws": 0,
            })

            means = losses[model][selected]
            timestep_scores = means[:, 1] - means[:, 0]
            n = timestep_scores.shape[1]
            profiles.append(pd.DataFrame({
                "task": task, "model": model, "timestep_index": np.arange(n),
                "n_timesteps": n, "schedule_position": np.linspace(0, 1, n),
                "mean_score": timestep_scores.mean(axis=0), "sample_sd": timestep_scores.std(axis=0, ddof=1),
                "sensitivity": sensitivity(timestep_scores),
                "alignment": pearson(timestep_scores, y),
                "within_condition_alignment": pearson(center_within(timestep_scores, conditions), center_within(y, conditions)),
            }))

            for condition in CONDITIONS[task]:
                mask = masks[condition]
                condition_profiles.append(pd.DataFrame({
                    "task": task, "condition": condition, "model": model,
                    "timestep_index": np.arange(n), "n_timesteps": n, "schedule_position": np.linspace(0, 1, n),
                    "mean_score": timestep_scores[mask].mean(axis=0), "task_sample_sd": timestep_scores.std(axis=0, ddof=1),
                    "sensitivity": sensitivity(timestep_scores, mask), "alignment": pearson(timestep_scores[mask], y[mask]),
                    "n_pairs": int(mask.sum()), "n_task_pairs": len(local),
                }))

            for role, index in (("original", 0), ("modified", 1)):
                unnaturalness = 5 - local[f"{role}_mean_naturalness"].to_numpy()
                single_images.append({
                    "task": task, "model": model, "role": role,
                    "alignment": float(pearson(means[:, index].mean(axis=1), unnaturalness)),
                })

        # 3. Task-pooled human reference
        reference = _split_half_reference(split_half, task, "pooled")
        human_rows.append({
            "task": task,
            "sensitivity": float(observed["human_sensitivity"]),
            "sensitivity_ci_low": np.nan, "sensitivity_ci_high": np.nan,
            "alignment": reference.r_mean,
            "alignment_ci_low": reference.ci_low,
            "alignment_ci_high": reference.ci_high,
            "n_splits": int(reference.n_splits),
            "sensitivity_valid_draws": 0,
        })

    points = pd.DataFrame(points)
    profiles = pd.concat(profiles, ignore_index=True)
    single_images = pd.DataFrame(single_images)
    peaks = profile_peaks(profiles)

    # 4. Across-model associations and task summaries
    associations, summaries = [], []
    for task in TASKS:
        table = points[points.task == task]
        associations.append(_fit_association(table, task=task))

        task_peaks = peaks[peaks.task == task]
        single = single_images[single_images.task == task]
        summaries.append({
            "task": task, "n_models": len(table),
            "alignment_median": float(table.alignment.median()),
            "alignment_max": float(table.alignment.max()),
            "within_condition_alignment_median": float(table.within_condition_alignment.median()),
            "within_condition_alignment_max": float(table.within_condition_alignment.max()),
            "within_condition_positive_models": int((table.within_condition_alignment > 0).sum()),
            "sensitivity_peak_later": int((task_peaks.peak_position_difference > 0).sum()),
            "same_peak": int((task_peaks.peak_position_difference == 0).sum()),
            "peak_difference_median": float(task_peaks.peak_position_difference.median()),
            **{f"{role}_alignment_median": float(single[single.role == role].alignment.median())
               for role in ("original", "modified")},
        })

    condition_points = pd.DataFrame(condition_points)
    condition_profiles = pd.concat(condition_profiles, ignore_index=True)
    condition_peaks = profile_peaks(condition_profiles)

    # 5. Condition associations and peak summaries
    condition_associations, condition_peak_summary = [], []
    for (task, condition), table in condition_points.groupby(["task", "condition"], sort=False):
        condition_associations.append(_fit_association(table, task=task, condition=condition))

        peaks_in_condition = condition_peaks[(condition_peaks.task == task) & (condition_peaks.condition == condition)]
        shifts = peaks_in_condition.peak_position_difference
        condition_peak_summary.append({
            "task": task, "condition": condition, "n_models": len(shifts),
            "sensitivity_peak_later": int((shifts > 0).sum()),
            "same_peak": int((shifts == 0).sum()),
            "sensitivity_peak_earlier": int((shifts < 0).sum()),
            "peak_difference_median": shifts.median(),
        })

    points = pd.DataFrame(points)
    condition_points = pd.DataFrame(condition_points)
    result = {
        "model_metrics": points, "human_references": pd.DataFrame(human_rows),
        "across_model_associations": pd.DataFrame(associations), "timestep_profiles": profiles,
        "timestep_peaks": peaks, "single_image_alignment": single_images,
        "task_summary": pd.DataFrame(summaries), "condition_model_metrics": condition_points,
        "condition_human_references": pd.DataFrame(condition_humans),
        "condition_associations": pd.DataFrame(condition_associations),
        "condition_timestep_profiles": condition_profiles, "condition_timestep_peaks": condition_peaks,
        "condition_peak_summary": pd.DataFrame(condition_peak_summary),
    }
    result = apply_intervals(result, "sensitivity_alignment", data_root)
    meta = load_generative_model_metadata(data_root)
    result["model_metrics"] = result["model_metrics"].merge(meta, on="model", how="left")
    result["condition_model_metrics"] = result["condition_model_metrics"].merge(meta, on="model", how="left")
    result["timestep_peaks"] = result["timestep_peaks"].merge(meta, on="model", how="left")
    result["condition_timestep_peaks"] = result["condition_timestep_peaks"].merge(meta, on="model", how="left")
    try:
        from analysis.uncertainty import load_tables
        tables = load_tables(data_root=data_root)
        if "family_correlations" in tables:
            result["family_correlations"] = tables["family_correlations"]
    except Exception:
        pass
    return result


def save_results(result, output_dir, data_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def plot_cross_model_sensitivity_alignment(result, output_dir):
    """Render Figure 4: Scatter plots (A) and cross-model correlations (B)."""
    points = result["model_metrics"]
    human = result["human_references"]
    fits = result["across_model_associations"]
    summary = result.get("family_correlations")

    width, height = 7.35, 2.80
    plot_top = 2.53
    scatter_bottom, summary_bottom = .94, .45
    title_y = 2.68
    with plt.rc_context(figure_style(fontsize=8.5, **{"mathtext.fontset": "dejavusans"})):
        fig = plt.figure(figsize=(width, height), facecolor="white")

        def text(x, y, label, **kwargs):
            return fig.text(x / width, y / height, label, color=INK, va="center", **kwargs)

        def axes(x, y, w, h):
            return fig.add_axes([x / width, y / height, w / width, h / height])

        text(0.45, title_y, "A", fontsize=11, fontweight="bold")
        for index, (task, label, center_x) in enumerate((("thatcher", "Thatcher", 1.34), ("swap", "Illumination", 3.28))):
            x = (0.49, 2.43)[index]
            axis = axes(x, scatter_bottom, 1.70, plot_top - scatter_bottom)
            local = points.query("task == @task")
            h = human.query("task == @task").iloc[0]
            fit = fits.query("task == @task").iloc[0]
            text(center_x, title_y, label, fontsize=10.5, fontweight="bold", ha="center")
            axis.axvspan(h.sensitivity_ci_low, h.sensitivity_ci_high, color=INK, alpha=.04, lw=0)
            axis.axhspan(h.alignment_ci_low, h.alignment_ci_high, color=INK, alpha=.04, lw=0)
            axis.axvline(h.sensitivity, color=INK, lw=.85, ls=(0, (4, 2)), alpha=.8)
            axis.axhline(h.alignment, color=INK, lw=.85, ls=(0, (4, 2)), alpha=.8)
            line_x = np.linspace(local.sensitivity.min(), local.sensitivity.max(), 100)
            axis.plot(line_x, fit.intercept + fit.slope * line_x, color="#64748B", lw=1, alpha=.75)
            for row in local.itertuples():
                color = MODEL_TYPE_COLORS.get(row.model_type, "#0072B2")
                axis.hlines(row.alignment, row.sensitivity_ci_low, row.sensitivity_ci_high,
                            color=color, alpha=.24, lw=.55, zorder=2)
                axis.vlines(row.sensitivity, row.alignment_ci_low, row.alignment_ci_high,
                            color=color, alpha=.24, lw=.55, zorder=2)
            for row in local.sort_values("parameter_b", ascending=False).itertuples():
                color = MODEL_TYPE_COLORS.get(row.model_type, "#0072B2")
                axis.scatter(row.sensitivity, row.alignment, s=row.marker_area_pt2,
                             facecolors=to_rgba(color, .08), edgecolors=color,
                             linewidths=.8, zorder=3)
            axis.scatter(local.sensitivity, local.alignment, s=2.5,
                         c=local.model_type.map(MODEL_TYPE_COLORS), linewidths=0, zorder=4)
            axis.hlines(h.alignment, h.sensitivity_ci_low, h.sensitivity_ci_high, color=INK, lw=1, zorder=5)
            axis.vlines(h.sensitivity, h.alignment_ci_low, h.alignment_ci_high, color=INK, lw=1, zorder=5)
            axis.scatter(h.sensitivity, h.alignment, s=40, marker="D", color=INK,
                         edgecolors="white", linewidths=.6, zorder=6)
            axis.annotate("Human", (h.sensitivity, h.alignment), xytext=(6, -4),
                          textcoords="offset points", va="top", color=INK, fontsize=7.7)
            lo = min(0, local.sensitivity_ci_low.min(), h.sensitivity_ci_low)
            hi = max(local.sensitivity_ci_high.max(), h.sensitivity_ci_high)
            pad = (hi - lo) * .11
            axis.set(xlim=(lo-pad, hi+pad), ylim=(-.1, 1.), yticks=[0, .2, .4, .6, .8, 1.])
            axis.set_xticks([0, 1, 2, 3] if task == "thatcher" else [0, .5, 1, 1.5])
            axis.set_xlabel("Sensitivity", fontsize=8.4, labelpad=3)
            if index == 0:
                axis.set_ylabel("Human alignment $r$", fontsize=8.4, labelpad=3)
            else:
                axis.tick_params(labelleft=False)
            axis.axhline(0, color="#94A3B8", lw=.6, ls=":")
            axis.axvline(0, color="#94A3B8", lw=.6, ls=":")
            axis.grid(color="#E4E7EC", lw=.45)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set(color="#64748B", linewidth=.65)
            axis.tick_params(labelsize=8, length=2.5, pad=2)

        # Panel B: Cross-model Pearson r for pooled and conditions
        axis = axes(5.36, summary_bottom, 1.78, plot_top - summary_bottom)
        text(4.42, title_y, "B", fontsize=11, fontweight="bold")
        text(4.62, title_y, "Sensitivity–alignment", fontsize=10.5, fontweight="bold")
        axis.set(xlim=(-1.04, 1.04), ylim=(-.60, 8.65), yticks=[], xticks=[-1, -.5, 0, .5, 1])
        axis.axvline(0, color="#64748B", lw=.85, ls=(0, (3, 2)), zorder=1)
        axis.grid(axis="x", color="#E4E7EC", lw=.45)
        axis.set_axisbelow(True)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set(color="#64748B", linewidth=.65)
        axis.tick_params(labelsize=8, length=2.5, pad=2)
        axis.set_xticklabels(["−1", "−.5", "0", ".5", "1"])
        axis.set_xlabel("Cross-model Pearson $r$", fontsize=8.4, labelpad=3)
        for label, y in (("Thatcher", 8.1), ("Illumination", 3.9)):
            axis.text(-.52, y, label, transform=axis.get_yaxis_transform(), ha="left", va="center",
                      fontsize=8.5, color=INK, fontweight="bold", clip_on=False)
        for task, condition, label, y in (
            ("thatcher", "pooled", "All", 7.2),
            ("thatcher", "str", "Upright", 6.2), ("thatcher", "inv", "Inverted", 5.2),
            ("swap", "pooled", "All", 3.0),
            ("swap", "shadow_swap", "Shadow", 2.0), ("swap", "reflection_swap", "Reflection", 1.0),
            ("swap", "lightdir_swap", "Light direction", .0),
        ):
            pooled = condition == "pooled"
            if summary is not None and not summary.empty:
                sub = summary.query("task == @task and condition == @condition")
                if not sub.empty:
                    row = sub.iloc[0]
                    axis.hlines(y, row.ci_low, row.ci_high, color=INK, lw=1.35, zorder=2)
                    axis.vlines([row.ci_low, row.ci_high], y-.095, y+.095, color=INK, lw=1, zorder=2)
                    axis.scatter(row.estimate, y, s=25, marker="o",
                                 color=INK, edgecolors="white", lw=.5, zorder=3)
            elif "condition_associations" in result:
                if condition == "pooled":
                    row = fits.query("task == @task").iloc[0]
                    val = row.pearson_r
                else:
                    cond_sub = result["condition_associations"].query("task == @task and condition == @condition")
                    val = cond_sub.iloc[0].pearson_r if not cond_sub.empty else None
                if val is not None:
                    axis.scatter(val, y, s=25, marker="o", color=INK, edgecolors="white", lw=.5, zorder=3)

            axis.text(-.52, y, label, transform=axis.get_yaxis_transform(), ha="left", va="center",
                      color=INK, fontsize=8, fontweight="bold" if pooled else "normal", clip_on=False)

        # Legend
        legend = axes(.49, .08, 3.64, .50)
        legend.set(xlim=(.49, 4.13), ylim=(.08, .58))
        legend.set_axis_off()

        def legend_label(x, y, label, **kwargs):
            return legend.text(x, y, label, fontsize=8, color=INK, va="center", **kwargs)

        legend_label(.49, .44, "Model type", fontweight="bold")
        for x, (label, color) in zip((1.91, 2.68, 3.46), MODEL_TYPE_COLORS.items()):
            legend.scatter(x, .44, s=24, facecolors=to_rgba(color, .08), edgecolors=color, lw=.8)
            legend.scatter(x, .44, s=2.5, color=color, lw=0)
            legend_label(x+.12, .44, label)
        legend_label(.49, .17, "Size", fontweight="bold")
        legend_label(.85, .17, "(B; log scale)")
        for x, size in zip((1.91, 2.47, 3.02, 3.64), (.1, 1, 5, 20)):
            legend.scatter(x, .17, s=marker_area(size), facecolors="none", edgecolors="#586674", lw=.8)
            legend.scatter(x, .17, s=2.5, color="#586674", lw=0)
            legend_label(x+.13, .17, f"{size:g}")

        for extension in ("pdf", "png"):
            fig.savefig(output_dir / f"cross_model_sensitivity_alignment.{extension}", dpi=400, facecolor="white")
        plt.close(fig)


def plot(result, output_dir, *, split=True):
    """Render Figure 4 (cross-model sensitivity and alignment)."""
    plot_cross_model_sensitivity_alignment(result, output_dir)



def plot_single_image_alignment(result, output_dir):
    """Compare the existing 100 single-image and 50 paired correlations."""
    order = list(model_styles())
    variants = ("original", "modified", "paired_difference")
    single = result["single_image_alignment"].rename(columns={"role": "variant"})
    paired = result["model_metrics"][["task", "model", "alignment"]].assign(variant="paired_difference")
    points = pd.concat([single, paired], ignore_index=True)
    indexed = points.set_index(["task", "model", "variant"], verify_integrity=True)
    order = [m for m in order if all((task, m, variant) in indexed.index for task in TASKS for variant in variants)]
    if not np.isfinite(points.alignment).all() or points.alignment.abs().gt(1).any():
        raise ValueError("Single-image comparison requires finite correlations.")
    blue, ink = "#2b8cbe", "#1e293b"
    plot_rows = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.75), sharey=True)
        figure.subplots_adjust(left=.112, right=.978, bottom=.265, top=.855, wspace=.23)
        positions = np.array([0., 1., 2.25])
        for axis, task, title, letter in zip(axes, TASKS, ("Thatcher", "Illumination"), ("A", "B")):
            values = np.array([[indexed.loc[(task, model, variant), "alignment"] for variant in variants]
                               for model in order])
            for model, row, offset in zip(order, values, np.linspace(-.055, .055, len(order))):
                axis.plot(positions + offset, row, color=blue, alpha=.19, lw=.55,
                          marker="o", markersize=2.2, markeredgewidth=0, zorder=2)
                plot_rows.extend({"task": task, "model": model, "variant": variant,
                                  "x": float(x), "alignment": float(value)}
                                 for variant, x, value in zip(variants, positions + offset, row))
            medians = np.median(values, axis=0)
            axis.plot(positions, medians, color="white", lw=3.4, zorder=3)
            axis.plot(positions, medians, color=blue, lw=1.7, marker="D", markersize=5.2,
                      markeredgecolor=ink, markeredgewidth=.8, zorder=4)
            for variant, x, value, offset in zip(variants, positions, medians, ((8, 8), (8, -10), (8, 0))):
                label = f"{value:.3f}".replace("0.", ".").replace("-", "−")
                axis.annotate(label, (x, value), xytext=offset, textcoords="offset points",
                              ha="left", va="center", fontsize=7.5, weight="semibold", color=ink,
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94, "pad": 1.2}, zorder=5)
                plot_rows.append({"task": task, "model": "model_median",
                                  "variant": variant, "x": float(x),
                                  "alignment": float(value)})
            axis.set(xlim=(-.30, 2.86), ylim=(-.60, 1.0), xticks=positions,
                     xticklabels=["Original\nimage", "Modified\nimage", "Paired\ndifference"],
                     yticks=np.arange(-.6, 1.01, .2))
            axis.axhline(0, color="#9aa6b2", lw=.75, ls=(0, (3, 2)), zorder=1)
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#dde3e8", linewidth=.45)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set(color="#64748b", linewidth=.65)
            axis.tick_params(length=2.5, width=.65, pad=3)
            axis.text(0, 1.075, letter, transform=axis.transAxes, fontsize=10, weight="bold")
            axis.text(.095, 1.075, title, transform=axis.transAxes, fontsize=9, weight="semibold")
        axes[0].set_ylabel("Model–human Pearson $r$", labelpad=5)
        figure.legend(handles=[
            Line2D([], [], color=blue, alpha=.35, lw=.65, marker="o", markersize=2.5,
                   markeredgewidth=0, label="Individual models (n = 25)"),
            Line2D([], [], color=blue, lw=1.7, marker="D", markersize=4.6,
                   markeredgecolor=ink, markeredgewidth=.8, label="Median across models"),
        ], loc="lower center", bbox_to_anchor=(.55, .015), ncol=2, frameon=False,
           fontsize=6.8, handlelength=2.1, columnspacing=1.8, handletextpad=.6)
        for extension in ("pdf", "png"):
            figure.savefig(output_dir / f"single_image_naturalness.{extension}", dpi=400, facecolor="white")
        plt.close(figure)
    pd.DataFrame(plot_rows).to_csv(output_dir / "single_image_plot_values.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing data (default: settings.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/analysis/sensitivity_alignment)")
    args = parser.parse_args()

    data_root = configured_path("data_root", args.data_root)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path("output_root") / "analysis/sensitivity_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_human_pairs(data_root=data_root)
    losses, models = load_generative_means(items, data_root=data_root)
    result = analyze(items, losses, models, load_human_split_half(data_root=data_root), data_root=data_root)
    save_results(result, output_dir, data_root)
    plot(result, output_dir, split=True)
    plot_single_image_alignment(result, output_dir)
    plot_condition_figures(result, output_dir)
    print(result["task_summary"].to_string(index=False))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
