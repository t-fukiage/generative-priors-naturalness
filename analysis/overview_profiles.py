"""Reproduce Figure 1 profiles from ratings and model scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from data_io import configured_path, load_generative_scores, load_human_pairs, load_illumination_factors, load_model_candidates
from analysis.factor_plots import COLORS
from analysis.metrics import pearson
from analysis.plotting import figure_style
from analysis.uncertainty import human_profile_intervals
from analysis.selection import candidate_sets, select_models, top_generative_models

SPECS = (
    ("thatcher", (0, 1), "Orientation", ("Upright", "Inverted")),
    ("shadow_rotation", (5, 3), "Shadow", ("Teapot\n90°↔180°", "Spot\n−45°↔135°")),
    ("reflection_color", (3, 2), "Reflection", ("Large hue\nshift", "Small hue\nshift")),
    ("light_shape", (2, 5), "Light direction", ("Sphere", "Knot")),
)


def analyze(
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    levels: pd.DataFrame,
    membership: pd.DataFrame,
    human_intervals: pd.DataFrame,
    candidate_scores: np.ndarray,
    candidate_metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute human intervals, generative profiles, and task-selected encoder medians for Figure 1.

    Parameters
    ----------
    pairs : pd.DataFrame
        Stimulus pairs table containing human naturalness scores.
    scores : pd.DataFrame
        Pair-by-model matrix of generative model loss scores.
    levels : pd.DataFrame
        Factor level metadata table.
    membership : pd.DataFrame
        Pair-to-factor classification mapping.
    human_intervals : pd.DataFrame
        Precomputed 95% bootstrap confidence intervals for human factor levels.
    candidate_scores : np.ndarray
        Candidate scores in the same pair order as pairs.
    candidate_metadata : pd.DataFrame
        Candidate identities and readouts aligned with candidate_scores columns.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing:
        - 'human_intervals_and_medians': Level means, CIs, and top-5 median values.
        - 'individual_model_profiles': Standardized level means for each top-5 model.
        - 'selected_models': Table of top-5 generative models per task.
        - 'selected_encoders': Task-wise top-5 encoders with selected layers and readouts.
    """
    models_by_task, selected_models = top_generative_models(pairs, scores)
    encoder_candidates = {"encoder": candidate_sets(candidate_metadata)["encoder"]}
    encoder_by_task = {}
    selected_encoders = []
    for task in models_by_task:
        mask = pairs.task.eq(task).to_numpy()
        correlations = pearson(candidate_scores[mask], pairs.loc[mask, "score"].to_numpy())
        _, selected = select_models(correlations, encoder_candidates)
        columns = selected["encoder"]
        values = candidate_scores[mask][:, columns]
        # Preserve zero and use the full-task sample SD for every factor level.
        encoder_by_task[task] = values / values.std(axis=0, ddof=1)
        selected_encoders.append(candidate_metadata.iloc[columns].assign(
            task=task, rank=np.arange(1, 6), pooled_r=correlations[columns]))
    human_rows, model_rows = [], []
    for factor, orders, _, _ in SPECS:
        task = "thatcher" if factor == "thatcher" else "swap"
        local = pairs[pairs.task.eq(task)].copy()
        names = models_by_task[task]
        if task == "thatcher":
            labels = local.condition.map({"str": 0, "inv": 1}).to_numpy(int)
            level_names = ["Upright", "Inverted"]
        else:
            members = membership[membership.factor_id.eq(factor)].set_index("pair_id").level_order
            labels = local.pair_id.map(members).fillna(-1).to_numpy(int)
            level_names = levels[levels.factor_id.eq(factor)].sort_values("level_order").level_label.tolist()
        means = np.array([local.loc[labels == order, "score"].mean()
                          for order in range(len(level_names))]) / local.score.std(ddof=1)
        ci = human_intervals[human_intervals.factor_id.eq(factor)].sort_values("level_order")
        if ci.level_order.tolist() != list(range(len(level_names))):
            raise ValueError("Incomplete human intervals for Figure 1.")
        intervals = ci[["ci_low", "ci_high"]].to_numpy().T
        n_level_pairs = ci.n_pairs.tolist()
        values = scores.loc[local.pair_id, names].to_numpy()
        standardized = values / values.std(axis=0, ddof=1)
        for order in orders:
            model_means = standardized[labels == order].mean(axis=0)
            human_rows.append({"factor": factor, "level_order": order, "level_label": level_names[order],
                               "human_mean": means[order], "ci_low": intervals[0, order],
                               "ci_high": intervals[1, order], "top5_median": np.median(model_means),
                               "encoder_top5_median": np.median(encoder_by_task[task][labels == order].mean(axis=0)),
                               "n_items": n_level_pairs[order]})
            for model, value, offset in zip(names, model_means, np.linspace(.025, .125, 5)):
                model_rows.append({"factor": factor, "level_order": order, "model": model,
                                   "standardized_level_mean": value, "x_offset": offset})
    return {
        "human_intervals_and_medians": pd.DataFrame(human_rows),
        "individual_model_profiles": pd.DataFrame(model_rows),
        "selected_models": selected_models,
        "selected_encoders": pd.concat(selected_encoders, ignore_index=True),
    }


def plot(result: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Render Figure 1 overview profile panels.

    Parameters
    ----------
    result : dict[str, pd.DataFrame]
        Output dictionary from analyze().
    output_dir : Path
        Directory to save factorial_plots_detail.pdf and .png.
    """
    color, ink = COLORS["generative"], "#25252a"
    width, height = 5.5, 1.20
    with plt.rc_context(figure_style(fontsize=7)):
        fig = plt.figure(figsize=(width, height), facecolor="white")
        human = result["human_intervals_and_medians"]
        models = result["individual_model_profiles"]
        for k, ((factor, orders, title, labels), left) in enumerate(zip(SPECS, [.38, 1.69, 3.00, 4.31])):
            span, x = 1.07, np.array([0., 1.])
            fig.text((left + span / 2) / width, 1.13 / height, title, fontsize=7.1,
                     ha="center", va="center", weight="bold", color=ink)
            ax = fig.add_axes([left / width, .37 / height, span / width, .67 / height])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e9e9ed", linewidth=.4)
            ax.axhline(0, color="#bcbcc5", lw=.5)
            local = human[human.factor.eq(factor)].set_index("level_order").loc[list(orders)]
            individual = models[models.factor.eq(factor)]
            for _, model in individual.groupby("model", sort=False):
                model = model.set_index("level_order").loc[list(orders)]
                ax.plot(x + model.x_offset.to_numpy(), model.standardized_level_mean,
                        color=color, alpha=.26, lw=.65, marker="o", ms=2.25, mew=0, zorder=2)
            yerr_low = np.maximum(local.human_mean - local.ci_low, 0)
            yerr_high = np.maximum(local.ci_high - local.human_mean, 0)
            ax.errorbar(x - .065, local.human_mean,
                        yerr=[yerr_low, yerr_high],
                        fmt="o-", color=ink, mfc="white", ms=3.4, mew=.85, lw=.9,
                        elinewidth=.75, capsize=2.2, capthick=.75, zorder=5)
            ax.plot(x + .075, local.top5_median, "D-", color=color, mfc=color,
                    ms=3.8, mew=.45, mec="white", lw=1.25, zorder=6)
            ax.plot(x + .18, local.encoder_top5_median, color=COLORS["encoder"], marker="s", mfc="white",
                    ms=3.2, mew=.9, lw=1.05, linestyle=(0, (3, 1.6)), zorder=7)
            ax.set(xlim=(-.34, 1.34), ylim=(-.2, 3.1), yticks=[0, 1, 2, 3], xticks=[])
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.spines["left"].set_visible(k == 0)
            ax.spines["left"].set_color("#b5b5bf")
            ax.tick_params(axis="y", labelleft=k == 0, labelsize=6.4, length=2 if k == 0 else 0,
                           width=.5, color="#a5a5b0", pad=2)
            for value, label in zip(x, labels):
                fig.text((left + span * (value + .34) / 1.68) / width, .305 / height,
                         label, fontsize=6.3, ha="center", va="top", color=ink, linespacing=1.07)
        fig.text(.12 / width, .705 / height, "Unnaturalness", fontsize=6.6,
                 ha="center", va="center", rotation=90, color=ink)
        handles = [Line2D([], [], color=ink, marker="o", mfc="white", mew=.8, lw=.9, ms=3.4, label="Human mean (95% CI)"),
                   Line2D([], [], color=color, marker="D", mfc=color, mec="white", mew=.45, lw=1.25, ms=3.8, label="Top5 median"),
                   Line2D([], [], color=color, alpha=.30, marker="o", mew=0, lw=.65, ms=2.5, label="Individual Top5 models"),
                   Line2D([], [], color=COLORS["encoder"], marker="s", mfc="white", mew=.9, lw=1.05,
                          linestyle=(0, (3, 1.6)), ms=3.2, label="Encoder Top5 median")]
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(.515, .040 / height), frameon=False,
                   fontsize=6.1, ncol=4, handlelength=1.6, handletextpad=.5, columnspacing=1.5, borderpad=0)
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"factorial_plots_detail.{extension}", dpi=300, facecolor="white")
        plt.close(fig)


def main() -> None:
    """CLI entry point for reproducing Figure 1 profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing data (default: settings.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/analysis/overview_profiles)")
    args = parser.parse_args()

    data_root = configured_path("data_root", args.data_root)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path("output_root") / "analysis/overview_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_human_pairs(data_root=data_root)
    scores, _ = load_generative_scores(pairs, data_root=data_root)
    levels, membership = load_illumination_factors(pairs, data_root=data_root)
    intervals = human_profile_intervals(data_root=data_root)
    candidate_scores, candidate_metadata = load_model_candidates(pairs, data_root=data_root)
    result = analyze(pairs, scores, levels, membership, intervals, candidate_scores, candidate_metadata)
    for name, frame in result.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    plot(result, output_dir)
    print(f"Saved Figure 1 profiles and human intervals to {output_dir}")


if __name__ == "__main__":
    main()
