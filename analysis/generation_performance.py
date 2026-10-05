"""Relate generative model scores to external generation evaluations."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from data_io import configured_path, load_generative_scores, load_human_pairs
from analysis.metrics import pearson, residualize, sensitivity
from analysis.plotting import model_styles, figure_style
from analysis.uncertainty import apply_intervals


TASKS = (("thatcher", "Thatcher"), ("swap", "Illumination"))
# Outcome and covariates; ranks are formed before either regression.
RELATIONS = (("alignment", "none", ()),
             ("alignment", "sensitivity", ("sensitivity",)),
             ("sensitivity", "alignment", ("alignment",)),
             ("alignment", "sensitivity_size", ("sensitivity", "parameter_b")))


def load_evaluations(models, data_root=None):
    """Read the fixed matched panel, its source records, and parameter counts."""
    root = configured_path("data_root", data_root) / "generation_evaluations"
    matched = pd.read_csv(root / "matched_models.csv").sort_values(["modality", "model"])
    # Preserve the manuscript's IDs, including gaps for unmatched models.
    counters = {"o": 0, "s": 0, "^": 0}
    labels = {}
    for model, style in model_styles().items():
        marker = style["marker"]
        counters[marker] += 1
        labels[model] = f"{dict(o='P', s='L', **{'^': 'V'})[marker]}{counters[marker]:02}"
    matched["label"] = matched.model.map(labels)
    return matched.reset_index(drop=True)


def compute_model_scores(items, scores, matched):
    """Compute sensitivity and alignment for each matched model across tasks."""
    model_names = matched.model.tolist()
    score_matrix = scores.loc[items.pair_id, model_names].to_numpy()
    points = []
    for task, _ in TASKS:
        task_mask = items.task.eq(task).to_numpy()
        local_items = items.loc[task_mask]
        x = score_matrix[task_mask]
        human = local_items.score.to_numpy()

        table = matched.copy()
        table["task"] = task
        table["sensitivity"] = sensitivity(x)
        table["alignment"] = pearson(x, human)
        table["alignment_ci_low"] = np.nan
        table["alignment_ci_high"] = np.nan
        table["alignment_valid_draws"] = 0
        table["n_pairs"] = len(local_items)
        table["n_scene_groups"] = local_items.base_scene_group.nunique()
        points.append(table)
    return pd.concat(points, ignore_index=True)


def compute_associations(points):
    """Compute Spearman and partial Spearman associations and residual plot coordinates."""
    associations, coordinates = [], []
    for (modality, task), panel in points.groupby(["modality", "task"], sort=False):
        values = panel[["external_score", "sensitivity", "alignment", "parameter_b"]]
        ranked = values.rank(method="average")
        for outcome, adjustment, controls in RELATIONS:
            columns = ["external_score", outcome]
            data = ranked[columns].to_numpy()
            if controls:
                data = residualize(data, ranked[list(controls)])
            sd = data.std(axis=0, ddof=1)
            if np.any(sd == 0):
                raise ValueError(f"Undefined correlation: {modality}/{task}/spearman/{adjustment}")

            associations.append({
                "modality": modality,
                "task": task,
                "method": "spearman",
                "adjustment": adjustment,
                "outcome": outcome,
                "covariates": ";".join(controls),
                "n_models": len(panel),
                "coefficient": float(pearson(data[:, 0], data[:, 1])),
            })

            if controls:
                coords = panel[["model", "label", "display_name", "modality", "task"]].copy()
                coords["method"] = "spearman"
                coords["outcome"] = outcome
                coords["adjustment"] = adjustment
                for control in ("sensitivity", "alignment", "parameter_b"):
                    coords[f"{control}_covariate"] = ranked[control] if control in controls else np.nan
                coords["external_score_transformed"] = ranked["external_score"]
                coords["external_score_residual"] = data[:, 0]
                coords["external_score_plot"] = data[:, 0] / sd[0]
                coords["outcome_transformed"] = ranked[outcome]
                coords["outcome_residual"] = data[:, 1]
                coords["outcome_plot"] = data[:, 1] / sd[1]
                coordinates.append(coords)

    associations_df = pd.DataFrame(associations)
    coordinates_df = pd.concat(coordinates, ignore_index=True) if coordinates else pd.DataFrame()
    return associations_df, coordinates_df


def format_main_table(associations):
    """Format summary table of raw and partial Spearman associations."""
    main = []
    for (modality, task), panel in associations.query("method == 'spearman'").groupby(["modality", "task"]):
        lookup = {(row.outcome, row.adjustment): row.coefficient for row in panel.itertuples()}
        main.append({
            "modality": modality,
            "task": task,
            "n_models": int(panel.n_models.iloc[0]),
            "alignment_raw": lookup[("alignment", "none")],
            "alignment_given_sensitivity": lookup[("alignment", "sensitivity")],
            "sensitivity_given_alignment": lookup[("sensitivity", "alignment")],
            "alignment_given_sensitivity_size": lookup[("alignment", "sensitivity_size")],
        })
    return pd.DataFrame(main)


def analyze(items, scores, matched, data_root=None):
    """Compute local model scores, descriptive associations, and plot coordinates."""
    points = compute_model_scores(items, scores, matched)
    associations, coordinates = compute_associations(points)
    main_table = format_main_table(associations)
    result = {
        "model_scores": points,
        "associations": associations,
        "partial_coordinates": coordinates,
        "main_table": main_table,
    }
    return apply_intervals(result, "generation_performance", data_root)


def save_results(result, output_dir, data_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def label_points(axis, rows):
    """Place short model IDs near points, avoiding text and marker overlaps."""
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    bounds = axis.get_window_extent(renderer).padded(-3)
    occupied = []
    for row in rows:
        x, y = axis.transData.transform((row[1], row[2]))
        occupied.append(Bbox.from_extents(x - 4, y - 4, x + 4, y + 4))
    for label, x, y in sorted(rows, key=lambda row: (row[2], row[1])):
        text = axis.annotate(label, (x, y), xytext=(5, 7), textcoords="offset points", fontsize=6.5,
                             va="center", color="#182536", arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#94A3B8"})
        options = []
        for dy in (7, -7, 15, -15, 23, -23, 31, -31):
            for dx, align in ((5, "left"), (-5, "right"), (0, "center")):
                text.set_position((dx, dy))
                text.set_ha(align)
                text.update_positions(renderer)
                box = plt.Text.get_window_extent(text, renderer=renderer).padded(1)
                inside = bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                cost = 1000 * (not inside) + 100 * sum(box.overlaps(other) for other in occupied) + abs(dx) + abs(dy)
                options.append((cost, dx, dy, align, box))
        _, dx, dy, align, box = min(options, key=lambda option: option[0])
        text.set_position((dx, dy))
        text.set_ha(align)
        occupied.append(box)


def plot(result, output_dir):
    styles = model_styles()
    points = result["model_scores"]
    legend = points.drop_duplicates("model").sort_values("label")
    with plt.rc_context(figure_style(fontsize=8)):
        for adjustment, outcome, stem in (
            ("none", "alignment", "generation_performance_alignment"),
            ("sensitivity", "alignment", "sensitivity_adjusted_spearman_scatter"),
            ("alignment", "sensitivity", "alignment_adjusted_sensitivity_scatter"),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.2))
            fig.subplots_adjust(left=0.11, right=0.98, bottom=0.25, top=0.91, hspace=0.55, wspace=0.32)
            for row_index, modality in enumerate(("image", "video")):
                for column, (task, task_label) in enumerate(TASKS):
                    axis = axes[row_index, column]
                    association = result["associations"].query(
                        "modality == @modality and task == @task and adjustment == @adjustment and outcome == @outcome and method == 'spearman'").iloc[0]
                    panel = points.query("modality == @modality and task == @task").set_index("model")
                    if adjustment != "none":
                        coords = result["partial_coordinates"].query(
                            "modality == @modality and task == @task and adjustment == @adjustment and outcome == @outcome and method == 'spearman'").set_index("model")
                        x, y = coords.external_score_plot, coords.outcome_plot
                        axis.set(xlim=(-2.85, 2.85), ylim=(-2.85, 2.85), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
                        axis.axhline(0, color="#94A3B8", lw=0.6, ls=":")
                        axis.axvline(0, color="#94A3B8", lw=0.6, ls=":")
                        line = np.array([-2.5, 2.5])
                        axis.plot(line, association.coefficient * line, color="#6B7280", lw=0.8)
                        axis.set_xlabel("Generation rank residual (SD units)")
                        axis.set_ylabel(f"{outcome.title()} rank residual (SD units)")
                    else:
                        x, y = panel.external_score, panel.alignment
                        axis.set_ylim(-0.05, 1.03)
                        axis.set_xlim((420, 1050) if modality == "image" else (82.0, 85.8))
                        axis.set_xlabel("Published human-preference Elo" if modality == "image" else "VBench Quality (%)")
                        axis.set_ylabel("Human alignment $r$")
                    labels = []
                    for model, item in panel.iterrows():
                        style = styles[model]
                        if adjustment == "none":
                            axis.vlines(x[model], item.alignment_ci_low, item.alignment_ci_high, color=style["color"], lw=0.7, alpha=0.5)
                            if modality == "image":
                                axis.hlines(y[model], item.ci95_lower, item.ci95_upper, color=style["color"], lw=0.7, alpha=0.5)
                        axis.scatter(x[model], y[model], marker=style["marker"], color=[style["color"]],
                                     s=29, edgecolor="white", linewidth=0.4, zorder=3)
                        labels.append((item.label, x[model], y[model]))
                    prefix = "Partial Spearman" if adjustment != "none" else "Spearman"
                    letter = "ABCD"[2 * row_index + column]
                    axis.set_title(f"{letter}  {modality.title()} · {task_label}\n"
                                   f"{prefix} $\\rho={association.coefficient:.3f}$; $n={len(panel)}$", loc="left", fontsize=8.5, pad=7)
                    axis.spines[["top", "right"]].set_visible(False)
                    axis.grid(color="#E5E7EB", lw=0.4)
                    axis.set_axisbelow(True)
                    axis.tick_params(length=2.5, width=0.6)
                    label_points(axis, labels)
            handles = [Line2D([], [], ls="", marker=styles[row.model]["marker"], color=styles[row.model]["color"],
                              markersize=4, label=f"{row.label}  {row.display_name}") for row in legend.itertuples()]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.02), ncol=3,
                       fontsize=6.4, frameon=False, columnspacing=1.2, handletextpad=0.4, handlelength=0.9)
            for extension in ("pdf", "png"):
                fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, facecolor="white")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing data (default: settings.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/analysis/generation_performance)")
    args = parser.parse_args()

    data_root = configured_path("data_root", args.data_root)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path("output_root") / "analysis/generation_performance"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_human_pairs(data_root=data_root)
    scores, models = load_generative_scores(items, data_root=data_root)
    result = analyze(items, scores, load_evaluations(models, data_root=data_root), data_root=data_root)
    save_results(result, output_dir, data_root)
    plot(result, output_dir)
    print(result["main_table"].to_string(index=False))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
