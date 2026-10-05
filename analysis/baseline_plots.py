"""Draw the main and condition-level comparison figures from numerical tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis.selection import GROUPS
from analysis.plotting import figure_style

COLORS = {"generative": "#2878B5", "encoder": "#4B8F43", "fr_iqa": "#D58227", "nr_iqa": "#BA629B"}
LABELS = {"generative": "Generative", "encoder": "Encoder", "fr_iqa": "FR-IQA", "nr_iqa": "NR-IQA"}
DIRECTIONS = (("g_partial", r"$r(h, G \,|\, X)$", "#7650A0", "o", -0.17),
              ("x_partial", r"$r(h, X \,|\, G)$", "#39865C", "o", 0.17))


def points(
    axis: plt.Axes,
    x: float,
    values: pd.Series | np.ndarray,
    color: str,
    marker: str = "o",
    size: float = 8,
    alpha: float = 1,
    width: float = 0.12,
    zorder: int = 3,
) -> None:
    """Deterministic horizontal jitter, independent of the statistical RNG.

    Parameters
    ----------
    axis : plt.Axes
        Matplotlib axis.
    x : float
        Central horizontal position.
    values : pd.Series | np.ndarray
        Data values to scatter.
    color : str
        Hex color string.
    marker : str, optional
        Matplotlib marker style (default: 'o').
    size : float, optional
        Marker size (default: 8).
    alpha : float, optional
        Marker transparency (default: 1).
    width : float, optional
        Jitter spread width (default: 0.12).
    zorder : int, optional
        Drawing order (default: 3).
    """
    vals = np.asarray(values)
    offsets = (((np.arange(len(vals)) * 0.618033988749895) % 1) - 0.5) * 2 * width
    axis.scatter(x + offsets, vals, s=size, color=color, marker=marker,
                 alpha=alpha, edgecolors="none", zorder=zorder, rasterized=False)


def median_interval(
    axis: plt.Axes,
    x: float,
    estimate: float,
    low: float,
    high: float,
    color: str,
) -> None:
    """Draw a median point and 95% confidence interval error bar.

    Parameters
    ----------
    axis : plt.Axes
        Matplotlib axis.
    x : float
        Horizontal coordinate.
    estimate : float
        Median point estimate.
    low : float
        95% CI lower endpoint.
    high : float
        95% CI upper endpoint.
    color : str
        Line and point color.
    """
    axis.plot([x, x], [low, high], color=color, lw=1.3, zorder=5)
    axis.plot([x - .055, x + .055], [low, low], color=color, lw=.9, zorder=5)
    axis.plot([x - .055, x + .055], [high, high], color=color, lw=.9, zorder=5)
    axis.scatter([x], [estimate], s=24, marker="D", facecolor=color, edgecolor="white", linewidth=.65, zorder=6)


def format_axis(
    axis: plt.Axes,
    labels: list[str],
    ylabel: str | None = None,
    *,
    is_partial: bool = False,
    show_yticklabels: bool = True,
    fontsize: float | None = None,
) -> None:
    """Apply consistent styling, grid, and ticks to comparison panels.

    Parameters
    ----------
    axis : plt.Axes
        Target axis.
    labels : list[str]
        Category labels for x-axis ticks.
    ylabel : str | None, optional
        Vertical axis title.
    is_partial : bool, optional
        Whether this is a partial correlation panel (default: False).
    show_yticklabels : bool, optional
        Whether to display vertical tick numbers (default: True).
    fontsize : float | None, optional
        Explicit tick font size override.
    """
    axis.axhline(0, color="#777777", lw=.65, linestyle="--", zorder=0)
    if is_partial:
        ylim = (-.48, 1.05)
        yticks = [-.4, -.2, 0, .2, .4, .6, .8, 1.0]
    else:
        ylim = (-.06, 1.05)
        yticks = [0, .2, .4, .6, .8, 1.0]
    axis.set(xticks=np.arange(len(labels)), xticklabels=labels, xlim=(-.55, len(labels) - .45),
             ylim=ylim, yticks=yticks)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", length=3.2, width=.7, pad=3.5)
    if fontsize is not None:
        axis.tick_params(axis="both", labelsize=fontsize)
    if not show_yticklabels:
        axis.tick_params(axis="y", labelleft=False)
    axis.grid(axis="y", color="#eeeeee", lw=.55, zorder=0)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=fontsize + 1.6 if fontsize is not None else None, labelpad=5)


def marginal_panel(
    axis: plt.Axes,
    result: dict[str, pd.DataFrame],
    task: str,
    scope: str,
    ylabel: str | None = None,
    *,
    show_yticklabels: bool = True,
    fontsize: float | None = None,
    point_scale: float = 1.0,
) -> None:
    """Draw marginal human alignment distributions and median intervals across model groups.

    Parameters
    ----------
    axis : plt.Axes
        Target axis.
    result : dict[str, pd.DataFrame]
        Dictionary containing comparison summary tables.
    task : str
        Task name ('thatcher' or 'swap').
    scope : str
        Condition scope ('pooled' or specific condition).
    ylabel : str | None, optional
        Vertical axis label.
    show_yticklabels : bool, optional
        Whether to show y-axis tick numbers (default: True).
    fontsize : float | None, optional
        Tick font size override.
    point_scale : float, optional
        Multiplier for scatter marker size (default: 1.0).
    """
    models = result["model_alignment"].query("task == @task and scope == @scope")
    summaries = result["marginal_summary"].query("task == @task and scope == @scope").set_index("group")
    human = result["human_references"].query("task == @task and scope == @scope").iloc[0]
    axis.axhspan(human.ci_low, human.ci_high, color="#666666", alpha=.10, lw=0, zorder=0)
    axis.axhline(human.r_mean, color="#555555", linestyle=":", lw=1.1)
    for i, group in enumerate(GROUPS):
        local = models[models.group.eq(group)]
        chosen = local[local.selected]
        points(axis, i, local.r, COLORS[group], alpha=.18, size=13 * point_scale, width=.23)
        points(axis, i, chosen.r, COLORS[group], size=18 * point_scale, width=.20)
        row = summaries.loc[group]
        median_interval(axis, i, row.median_r, row.ci_low, row.ci_high, COLORS[group])
    format_axis(axis, ["Gen.", "Encoder", "FR-IQA", "NR-IQA"], ylabel,
                is_partial=False, show_yticklabels=show_yticklabels, fontsize=fontsize)


def partial_panel(
    axis: plt.Axes,
    result: dict[str, pd.DataFrame],
    task: str,
    scope: str,
    ylabel: str | None = None,
    *,
    show_yticklabels: bool = True,
    fontsize: float | None = None,
    point_scale: float = 1.0,
) -> None:
    """Draw bidirectional partial correlation points, medians, and CIs.

    Parameters
    ----------
    axis : plt.Axes
        Target axis.
    result : dict[str, pd.DataFrame]
        Dictionary containing comparison summary tables.
    task : str
        Task name ('thatcher' or 'swap').
    scope : str
        Condition scope ('pooled' or specific condition).
    ylabel : str | None, optional
        Vertical axis label.
    show_yticklabels : bool, optional
        Whether to show y-axis tick numbers (default: True).
    fontsize : float | None, optional
        Tick font size override.
    point_scale : float, optional
        Multiplier for scatter marker size (default: 1.0).
    """
    pairs = result["partial_pairs"].query("task == @task and scope == @scope")
    summaries = result["partial_summary"].query("task == @task and scope == @scope").set_index("comparator_group")
    for i, group in enumerate(GROUPS[1:]):
        local = pairs[pairs.comparator_group.eq(group)]
        for statistic, _, color, marker, offset in DIRECTIONS:
            points(axis, i + offset, local[statistic], color, marker, size=4.5 * point_scale, alpha=.10, width=.12, zorder=1)
            points(axis, i + offset, local[local.selected][statistic], color, marker, size=11 * point_scale, alpha=.6)
            row = summaries.loc[group]
            median_interval(axis, i + offset, row[statistic], row[statistic + "_ci_low"], row[statistic + "_ci_high"], color)
            best = local[local.best_pair][statistic]
            axis.scatter(np.full(len(best), i + offset), best, s=60 * point_scale, marker="*", color=color,
                         edgecolor="white", linewidth=.55, zorder=7)
    format_axis(axis, ["Encoder", "FR-IQA", "NR-IQA"], ylabel,
                is_partial=True, show_yticklabels=show_yticklabels, fontsize=fontsize)


def legend(figure: plt.Figure, appendix: bool = False, fontsize: float = 9.2) -> None:
    """Add the shared comparison legend and caption to the bottom of the figure.

    Parameters
    ----------
    figure : plt.Figure
        Target figure.
    appendix : bool, optional
        Whether the figure is for the appendix (default: False).
    fontsize : float, optional
        Legend font size (default: 9.2).
    """
    ms_dot = 4 if appendix else 5.2
    ms_diamond = 4 if appendix else 5.0
    ms_star = 7 if appendix else 8.0
    lw_err = 1.0 if appendix else 1.2
    lw_split = 1.0 if appendix else 1.1

    handles = [Line2D([], [], color=color, marker=marker, lw=0, markersize=ms_dot, label=label)
               for _, label, color, marker, _ in DIRECTIONS]
    handles.extend([
        Line2D([], [], color="#555555", marker="D", lw=lw_err, markersize=ms_diamond, label="Median + 95% CI"),
        Line2D([], [], color="#555555", marker="*", lw=0, markersize=ms_star, label="Task rank-1 pair"),
        Line2D([], [], color="#555555", linestyle=":", lw=lw_split, label="Human split-half"),
    ])
    leg_fs = 7 if appendix else fontsize
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025 if appendix else .035),
                  ncol=5, frameon=False, fontsize=leg_fs, handletextpad=.4, columnspacing=1.6 if appendix else 1.8)
    figure.text(.5, .008 if appendix else .006,
                "Faint: all models/pairs. Saturated: selected five/25 pairs.",
                ha="center", va="bottom", fontsize=6.6 if appendix else (leg_fs - 1.0), color="#444444")


def save(figure: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure in PDF and PNG formats.

    Parameters
    ----------
    figure : plt.Figure
        Figure to save.
    output_dir : Path
        Destination directory.
    name : str
        Base file stem (without extension).
    """
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"{name}.{extension}", dpi=240, facecolor="white")
    plt.close(figure)


def plot(result: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Render Figures 5 and S14 (marginal and partial baseline comparison panels).

    Parameters
    ----------
    result : dict[str, pd.DataFrame]
        Complete analysis tables dictionary.
    output_dir : Path
        Directory to save baseline_comparison and condition_baseline_comparison figures.
    """
    fs_tick = 9.2
    fs_sub = 11.5
    fs_super = 12.8
    fs_legend = 9.2

    style = figure_style(fontsize=fs_tick)
    with plt.rc_context(style):
        figure = plt.figure(figsize=(11.8, 3.25))
        gs_master = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.0),
                                        left=.052, right=.990, bottom=.24, top=.80, wspace=.20)

        # Block A: Marginal alignment (Thatcher & Illumination)
        gs_a = gs_master[0, 0].subgridspec(1, 2, wspace=.11)
        ax_m1 = figure.add_subplot(gs_a[0, 0])
        ax_m2 = figure.add_subplot(gs_a[0, 1])

        # Block B: Bidirectional partial correlation (Thatcher & Illumination)
        gs_b = gs_master[0, 1].subgridspec(1, 2, wspace=.11)
        ax_p1 = figure.add_subplot(gs_b[0, 0])
        ax_p2 = figure.add_subplot(gs_b[0, 1])

        marginal_panel(ax_m1, result, "thatcher", "pooled", "Human alignment (Pearson r)",
                       show_yticklabels=True, fontsize=fs_tick)
        marginal_panel(ax_m2, result, "swap", "pooled", None,
                       show_yticklabels=False, fontsize=fs_tick)
        ax_m1.set_title("Thatcher", loc="center", fontsize=fs_sub, fontweight="bold", pad=7)
        ax_m2.set_title("Illumination", loc="center", fontsize=fs_sub, fontweight="bold", pad=7)

        partial_panel(ax_p1, result, "thatcher", "pooled", "Partial Pearson r",
                      show_yticklabels=True, fontsize=fs_tick)
        partial_panel(ax_p2, result, "swap", "pooled", None,
                      show_yticklabels=False, fontsize=fs_tick)
        ax_p1.set_title("Thatcher", loc="center", fontsize=fs_sub, fontweight="bold", pad=7)
        ax_p2.set_title("Illumination", loc="center", fontsize=fs_sub, fontweight="bold", pad=7)

        pos_m1 = ax_m1.get_position()
        pos_p1 = ax_p1.get_position()

        figure.text(pos_m1.x0, .935, "A", fontsize=fs_super + 2.5, fontweight="bold", va="bottom", ha="left")
        figure.text(pos_m1.x0 + 0.016, .935, "Marginal alignment",
                    fontsize=fs_super, fontweight="bold", va="bottom", ha="left")

        figure.text(pos_p1.x0, .935, "B", fontsize=fs_super + 2.5, fontweight="bold", va="bottom", ha="left")
        figure.text(pos_p1.x0 + 0.016, .935, "Bidirectional partial correlation",
                    fontsize=fs_super, fontweight="bold", va="bottom", ha="left")

        legend(figure, appendix=False, fontsize=fs_legend)
        save(figure, output_dir, "baseline_comparison")

        panels = (("thatcher", "str", "Upright"), ("thatcher", "inv", "Inverted"),
                  ("swap", "shadow_swap", "Shadow"), ("swap", "reflection_swap", "Reflection"),
                  ("swap", "lightdir_swap", "Light direction"))
        figure, axes = plt.subplots(2, 5, figsize=(12, 5.8))
        figure.subplots_adjust(left=.052, right=.99, bottom=.16, top=.9, wspace=.28, hspace=.32)
        for i, (task, scope, label) in enumerate(panels):
            marginal_panel(axes[0, i], result, task, scope, "Human alignment (Pearson r)" if i == 0 else None)
            partial_panel(axes[1, i], result, task, scope, "Partial Pearson r" if i == 0 else None)
            for axis in axes[:, i]:
                axis.set(ylim=(-.8, 1.05), yticks=[-.8, -.4, 0, .4, .8, 1.0])
            axes[0, i].set_title(f"{chr(65+i)}  {label}", loc="left", fontsize=9, fontweight="bold", pad=8)
            axes[1, i].text(-.03, 1.035, chr(70+i), transform=axes[1, i].transAxes, fontsize=9, fontweight="bold")
        figure.text(.235, .962, "Thatcher", ha="center", fontsize=10, fontweight="bold")
        figure.text(.715, .962, "Illumination", ha="center", fontsize=10, fontweight="bold")
        legend(figure, appendix=True)
        save(figure, output_dir, "condition_baseline_comparison")
