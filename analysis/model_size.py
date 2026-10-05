"""Appendix: pooled and condition-wise model size, sensitivity, and alignment."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from data_io import configured_path, load_human_pairs, load_model_candidates
from analysis.metrics import pearson, sensitivity
from analysis.plotting import MODEL_FAMILIES, model_styles, figure_style
from analysis.uncertainty import apply_intervals
from analysis.selection import candidate_sets, select_model_candidates


GROUPS = ("generative", "encoder")
TASKS = (("thatcher", "Thatcher"), ("swap", "Illumination"))
METRICS = ("sensitivity", "alignment")
CONDITION_PANELS = (
    ("thatcher", "str", "Upright"),
    ("thatcher", "inv", "Inverted"),
    ("swap", "shadow_swap", "Shadow"),
    ("swap", "reflection_swap", "Reflection"),
    ("swap", "lightdir_swap", "Light direction"),
)
TASK_TITLES = {"thatcher": "Thatcher", "swap": "Swap / Illumination"}


def load_sizes(metadata, data_root=None):
    """Read parameter counts and their definitions for the complete model panel."""
    root = configured_path("data_root", data_root) / "model_sizes"
    path = root / "models.csv"
    if not path.exists():
        raise FileNotFoundError(f"Model sizes not found at {path}")
    return pd.read_csv(path, float_precision="round_trip")


def model_statistics(human, scores, candidates):
    """Choose one candidate per model and calculate its sensitivity and alignment."""
    correlations = pearson(scores, human)
    chosen = select_model_candidates(correlations, candidates)
    columns = np.concatenate([chosen[group] for group in GROUPS])
    values = np.column_stack([sensitivity(scores[:, columns]), correlations[columns]])
    return columns, values


def size_associations(points):
    """Group-wise correlations and intercept-inclusive OLS against log10 size.

    Spearman uses average ranks for ties. Associations describe the fixed model
    panel and have no model-population confidence interval or p-value.
    """
    rows = []
    for (task, group), local in points.groupby(["task", "group"], sort=False):
        x = np.log10(local.parameter_b.to_numpy())
        ranks = local[["parameter_b", *METRICS]].rank(method="average")
        for metric in METRICS:
            y = local[metric].to_numpy()
            intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
            rows.append({"task": task, "group": group, "metric": metric, "n_models": len(x),
                         "pearson_log_size": float(pearson(x, y)),
                         "spearman_size": float(pearson(ranks.parameter_b, ranks[metric])),
                         "intercept": intercept, "slope_log10_size": slope})
    return pd.DataFrame(rows)


def condition_size_associations(points):
    """Correlations and intercept-inclusive OLS against log10 size across condition subsets.

    Evaluated separately for all 25 models, 16 image models, and 9 video models.
    Spearman correlations use average ranks for ties. Associations describe the
    evaluated model panel and have no model-population confidence interval.
    """
    rows = []
    for task, condition, label in CONDITION_PANELS:
        local = points.loc[points.task.eq(task) & points.condition.eq(condition)]
        for subset, n in (("all", 25), ("image", 16), ("video", 9)):
            selected = local if subset == "all" else local.loc[local.modality.eq(subset)]
            if len(selected) != n:
                raise ValueError(f"Expected {n} models for subset '{subset}' in {task}/{condition}, got {len(selected)}")
            x = np.log10(selected.parameter_b.to_numpy())
            ranks = selected[["parameter_b", *METRICS]].rank(method="average")
            for metric in METRICS:
                y = selected[metric].to_numpy()
                intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
                rows.append(dict(task=task, condition=condition, condition_label=label, subset=subset,
                                 metric=metric, n_models=n,
                                 pearson_log_size=float(pearson(x, y)),
                                 spearman_size=float(pearson(ranks.parameter_b, ranks[metric])),
                                 slope_log10_size=slope, intercept=intercept))
    return pd.DataFrame(rows)


def analyze(items, scores, metadata, sizes, data_root=None):
    all_candidates = candidate_sets(metadata)
    candidates = {group: all_candidates[group] for group in GROUPS}
    points, layers, frequencies = [], [], []

    for task, _ in TASKS:
        mask = items.task.eq(task).to_numpy()
        local, x = items.loc[mask], scores[mask]
        human = local.score.to_numpy()
        columns, values = model_statistics(human, x, candidates)

        # 1. Model metrics with parameter sizes and confidence interval placeholders
        table = metadata.iloc[columns].merge(sizes, on=["group", "model"], validate="one_to_one", how="left")
        table["task"] = task
        table["n_pairs"] = len(local)
        table["n_scene_groups"] = local.base_scene_group.nunique()
        for index, metric in enumerate(METRICS):
            table[metric] = values[:, index]
            table[f"{metric}_ci_low"] = np.nan
            table[f"{metric}_ci_high"] = np.nan
            table[f"{metric}_valid_draws"] = 0
        points.append(table)

        # 2. Selected representative encoder layers
        layers.append(table.loc[table.group.eq("encoder"), ["task", "model", "candidate", "readout", "alignment"]])

        # 3. Layer selection frequencies (placeholder counts, overwritten by bootstrap)
        counts = np.zeros(x.shape[1], dtype=int)
        frequency = metadata[metadata.group.eq("encoder")].copy()
        frequency["task"] = task
        frequency["draws"] = 0
        frequency["selected_count"] = counts[frequency.index]
        frequencies.append(frequency)

    points = pd.concat(points, ignore_index=True)
    result = {
        "model_metrics": points,
        "size_associations": size_associations(points),
        "selected_layers": pd.concat(layers, ignore_index=True),
        "layer_selection_frequency": pd.concat(frequencies, ignore_index=True),
    }
    return apply_intervals(result, "model_size", data_root)


def analyze_conditions(items, scores, metadata, sizes, data_root=None):
    """Calculate condition-specific sensitivity and alignment for all 25 generators."""
    columns = np.flatnonzero(metadata.group.eq("generative"))
    panel = metadata.iloc[columns].reset_index(drop=True)
    if len(panel) != 25 or panel.model.nunique() != 25:
        raise ValueError("Expected one scoring column for each of the 25 generators")
    values = np.asarray(scores)[:, columns]
    rows = []
    for task, condition, _ in CONDITION_PANELS:
        task_mask = items.task.eq(task).to_numpy()
        local, x = items.loc[task_mask], values[task_mask]
        condition_mask = local.condition.eq(condition).to_numpy()
        s = sensitivity(x, condition_mask)
        a = pearson(x[condition_mask], local.loc[condition_mask, "score"].to_numpy())
        for index, model in enumerate(panel.model):
            rows.append(dict(task=task, condition=condition, model=model,
                             n_pairs=int(condition_mask.sum()), n_task_pairs=len(local),
                             n_base_scene_groups=local.base_scene_group.nunique(),
                             sensitivity=s[index], alignment=a[index]))
    points = pd.DataFrame(rows)
    result = apply_intervals({"condition_model_metrics": points}, "model_size_conditions", data_root)
    size_columns = ["model", "display_name", "parameter_b", "count_kind", "count_scope", "count_note"]
    points = result["condition_model_metrics"].merge(
        sizes.loc[sizes.group.eq("generative"), size_columns], on="model", how="left", validate="many_to_one")
    styles = model_styles()
    points["family"] = points.model.map(lambda model: styles[model]["family"])
    points["modality"] = points.model.map(lambda model: "video" if styles[model]["marker"] == "^" else "image")
    required = ["parameter_b", "sensitivity", "alignment", "sensitivity_ci_low", "sensitivity_ci_high",
                "alignment_ci_low", "alignment_ci_high"]
    if len(points) != 125 or not np.isfinite(points[required]).all().all() or not points.parameter_b.gt(0).all():
        raise ValueError("Expected 125 complete model-condition estimates, sizes, and intervals")
    return {"condition_model_metrics": points, "condition_size_associations": condition_size_associations(points)}


def save_results(result, output_dir, data_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def plot(result, output_dir):
    styles = model_styles()
    encoder_color = "#377D43"
    points, associations = result["model_metrics"], result["size_associations"]
    with plt.rc_context(figure_style(fontsize=8)):
        figure, axes = plt.subplots(2, 2, figsize=(8.4, 7.5), sharex=True, sharey="row")
        figure.subplots_adjust(left=.095, right=.975, top=.84, bottom=.22, hspace=.68, wspace=.16)
        for row, metric in enumerate(METRICS):
            for col, (task, label) in enumerate(TASKS):
                axis = axes[row, col]
                for group in GROUPS:
                    local = points.query("task == @task and group == @group")
                    for model in local.itertuples():
                        if group == "generative":
                            style = styles[model.model]
                            color, marker, face = style["color"], style["marker"], style["color"]
                        else:
                            color, marker, face = encoder_color, "D", "white"
                        low, high = getattr(model, metric + "_ci_low"), getattr(model, metric + "_ci_high")
                        axis.vlines(model.parameter_b, low, high, color=color, alpha=.4, lw=.8, zorder=2)
                        axis.scatter(model.parameter_b, getattr(model, metric), s=22, marker=marker,
                                     facecolor=face, edgecolor=color, linewidth=.8, zorder=3)
                    stat = associations.query("task == @task and group == @group and metric == @metric").iloc[0]
                    x = np.linspace(np.log10(local.parameter_b.min()), np.log10(local.parameter_b.max()), 100)
                    color = "#444444" if group == "generative" else encoder_color
                    axis.plot(10 ** x, stat.intercept + stat.slope_log10_size * x,
                              color=color, ls="-" if group == "generative" else "--", lw=1, zorder=1)
                    axis.text(0, 1.13 if group == "generative" else 1.035,
                              f"{group.title()}: r = {stat.pearson_log_size:.3f}, ρ = {stat.spearman_size:.3f}",
                              transform=axis.transAxes, color=color, fontsize=8)
                axis.set_title(f"{'ABCD'[2 * row + col]}  {label}", loc="left", y=1.23, fontsize=10, fontweight="bold")
                axis.set_xscale("log")
                axis.set(xlim=(.018, 28), xticks=[.03, .1, 1, 10], xticklabels=["0.03", "0.1", "1", "10"])
                axis.tick_params(axis="x", labelbottom=True)
                axis.axhline(0, color="#bbbbbb", ls="--", lw=.6, zorder=0)
                axis.grid(color="#eeeeee", lw=.5)
                axis.spines[["top", "right"]].set_visible(False)
                if col == 0:
                    axis.set_ylabel("Sensitivity (mean score / SD)" if row == 0 else "Human alignment (Pearson r)")
        figure.suptitle("Model size, sensitivity, and human alignment", y=.985, fontsize=13)
        figure.text(.54, .175, "Scoring parameters (billions; log axis)", ha="center", fontsize=9)
        handles = [Line2D([], [], color=color, lw=2, label=family) for family, (color, _, _) in MODEL_FAMILIES.items()]
        handles.append(Line2D([], [], color=encoder_color, marker="D", markerfacecolor="white", lw=0, label="Encoder (12)"))
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .072), ncol=4,
                      frameon=False, fontsize=8, handlelength=1.3, columnspacing=1.8)
        handles = [Line2D([], [], color="#555555", marker=marker, lw=0, label=label)
                   for marker, label in (("o", "Pixel image"), ("s", "Latent image"), ("^", "Video"))]
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .037), ncol=3, frameon=False, fontsize=8)
        figure.text(.54, .018, "Bars: 95% bootstrap CIs (participants + scenes; generator sensitivity: scenes). Lines: fits by group.", ha="center", fontsize=7.5)
        for extension in ("pdf", "png"):
            figure.savefig(output_dir / f"model_size_sensitivity_alignment_with_encoders.{extension}", dpi=240, facecolor="white")
        plt.close(figure)


def plot_conditions(points, stats, task, output_dir):
    """Draw condition-wise sensitivity and alignment panels across model sizes for a task."""
    panels = [(c, title) for t, c, title in CONDITION_PANELS if t == task]
    styles = model_styles()
    width = 8.4 if len(panels) == 2 else 11.8
    with plt.rc_context(figure_style(fontsize=10)):
        fig, axes = plt.subplots(2, len(panels), figsize=(width, 7.6), sharex=True, sharey="row", squeeze=False)
        fig.subplots_adjust(left=.10 if len(panels) == 2 else .075, right=.98,
                            bottom=.25, top=.855, hspace=.44, wspace=.13)
        for col, (condition, title) in enumerate(panels):
            local = points.loc[points.task.eq(task) & points.condition.eq(condition)]
            for row, metric in enumerate(METRICS):
                ax = axes[row, col]
                for p in local.itertuples():
                    style = styles[p.model]
                    ax.vlines(p.parameter_b, getattr(p, metric + "_ci_low"), getattr(p, metric + "_ci_high"),
                              color=style["color"], lw=.75, alpha=.35, zorder=2)
                    ax.scatter(p.parameter_b, getattr(p, metric), marker=style["marker"], color=style["color"],
                               s=40, edgecolors="white", linewidths=.45, zorder=3)
                stat = stats.loc[stats.task.eq(task) & stats.condition.eq(condition)
                                 & stats.metric.eq(metric) & stats.subset.eq("all")].iloc[0]
                x = np.linspace(np.log10(local.parameter_b.min()), np.log10(local.parameter_b.max()), 100)
                ax.plot(10**x, stat.intercept + stat.slope_log10_size * x, color="#333333", lw=1, zorder=1)
                ax.set_title(f"{'ABCDEF'[row * len(panels) + col]}  {title}", loc="left", fontsize=11,
                             fontweight="bold", pad=24)
                ax.text(0, 1.025, f"All 25: r = {stat.pearson_log_size:+.3f}, ρ = {stat.spearman_size:+.3f}",
                        transform=ax.transAxes, fontsize=9)
                ax.set_xscale("log")
                ax.set(xlim=(.1, 26), xticks=[.1, 1, 10], xticklabels=["0.1", "1", "10"])
                ax.axhline(0, color="#999999", lw=.6, ls="--", zorder=0)
                ax.grid(color="#ebedf0", lw=.5)
                ax.spines[["top", "right"]].set_visible(False)
                if col == 0:
                    ax.set_ylabel("Sensitivity\n(condition mean / task SD)" if row == 0
                                  else "Human alignment\n(within-condition Pearson r)")
        # Same scale for a given metric across the two task figures, including all interval endpoints.
        for row, metric in enumerate(METRICS):
            low = min(0., points[metric + "_ci_low"].min())
            high = points[metric + "_ci_high"].max()
            margin = .07 * (high - low)
            axes[row, 0].set_ylim(low - margin, high + margin)
        fig.suptitle(f"{TASK_TITLES[task]}: model size by condition", fontsize=14, fontweight="bold", y=.98)
        fig.text(.5, .935, "25 generative models: 16 image + 9 video", ha="center", fontsize=10, color="#555555")
        fig.text(.54, .192, "Denoiser / flow-backbone parameters (billions; log axis)", ha="center", fontsize=10)
        handles = [Line2D([], [], color=color, marker=marker, lw=0, markersize=5, label=family)
                   for family, (color, marker, _) in MODEL_FAMILIES.items()]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .065), ncol=4,
                   frameon=False, fontsize=8.5, columnspacing=1.7)
        fig.text(.53, .016, "Bars: 95% bootstrap CIs. Black line: linear fit against log parameter count. Annotations: r and ρ across all 25 models.",
                 ha="center", fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"model_size_conditions_{task}.{ext}", dpi=220, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing data (default: settings.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/analysis/model_size)")
    args = parser.parse_args()

    data_root = configured_path("data_root", args.data_root)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path("output_root") / "analysis/model_size"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_human_pairs(data_root=data_root)
    scores, metadata = load_model_candidates(items, data_root=data_root)
    sizes = load_sizes(metadata, data_root=data_root)
    result = analyze(items, scores, metadata, sizes, data_root=data_root)
    result.update(analyze_conditions(items, scores, metadata, sizes, data_root=data_root))
    save_results(result, output_dir, data_root)
    plot(result, output_dir)
    for task, _ in TASKS:
        plot_conditions(result["condition_model_metrics"], result["condition_size_associations"], task, output_dir)
    print(f"Saved three model-size appendix figures and numerical tables to {output_dir}")


if __name__ == "__main__":
    main()
