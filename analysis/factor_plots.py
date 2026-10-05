"""Draw descriptive illumination profiles and their model–human squared errors."""

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


def save(figure: plt.Figure, output_dir: Path, name: str) -> None:
    """Save a matplotlib figure to disk in both PDF and PNG formats.

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
        figure.savefig(output_dir / f"{name}.{extension}", dpi=220, facecolor="white")
    plt.close(figure)


def mse_points(axis: plt.Axes, values: pd.Series | np.ndarray, y: float, color: str) -> None:
    """Draw individual model MSE points and their group median marker on an axis.

    Parameters
    ----------
    axis : plt.Axes
        Matplotlib axis on which to draw.
    values : pd.Series | np.ndarray
        Array of model MSE values.
    y : float
        Vertical baseline position.
    color : str
        Hex color for the scatter points.
    """
    vals = np.asarray(values)
    axis.scatter(vals, y + np.linspace(-.08, .08, len(vals)), s=14, color=color, alpha=.65, zorder=3)
    axis.scatter(np.median(vals), y, marker="D", s=32, color=color, edgecolor="white", linewidth=.65, zorder=4)


def overview(
    result: dict[str, pd.DataFrame],
    levels: pd.DataFrame,
    output_dir: Path,
    mse_max: float,
) -> None:
    """Render the overall factor-level model-human MSE summary across all factors.

    Parameters
    ----------
    result : dict[str, pd.DataFrame]
        Analysis outputs containing MSE tables.
    levels : pd.DataFrame
        Illumination factor levels table.
    output_dir : Path
        Destination directory to save figures.
    mse_max : float
        Upper limit for the horizontal MSE axis.
    """
    factors = levels.drop_duplicates("factor_id")
    labels = ["Overall", *[f"{r.factor_code}  {r.factor_label}" for r in factors.itertuples()]]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 7.2), sharey=True)
    fig.subplots_adjust(left=.21, right=.98, bottom=.12, top=.83, wspace=.12)
    for axis, group in zip(axes, GROUPS):
        overall = result["mse_by_model_scope"].query("scope == 'overall' and group == @group")
        table = result["mse_by_model_factor"].query("group == @group")
        mse_points(axis, overall.sort_values("rank").mse, 0, COLORS[group])
        for i, factor in enumerate(factors.factor_id, 1):
            mse_points(axis, table[table.factor_id.eq(factor)].sort_values("rank").mse, i, COLORS[group])
        axis.axhspan(-.5, .5, color="#edf0f3", zorder=0)
        for boundary in (.5, 4.5, 9.5):
            axis.axhline(boundary, color="#bbbbbb", lw=.7)
        axis.set(xlim=(-.03 * mse_max, mse_max), ylim=(13.6, -.65), xticks=np.arange(0, mse_max + .1, 2))
        axis.set_title(LABELS[group] + "\n" + ("Top 5" if group in GROUPS[:2] else "All 5"), fontsize=10, color=COLORS[group])
        axis.set_yticks(range(len(labels)), labels)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#e8e8e8", lw=.6)
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Illumination: factor-level model–human MSE", y=.975, fontsize=13)
    fig.text(.60, .918, "Dots: individual models. Diamonds: median of five models.", ha="center", fontsize=9)
    fig.text(.60, .065, "MSE in task-SD score units squared (lower is closer)", ha="center", fontsize=10)
    fig.text(.60, .028, "S: Shadow; R: Reflection; L: Light direction. Overall: three operations weighted equally per model.",
             ha="center", fontsize=7.4)
    save(fig, output_dir, "illumination_factor_disagreement_overview")


def operation_profiles(
    result: dict[str, pd.DataFrame],
    levels: pd.DataFrame,
    operation: str,
    output_dir: Path,
    score_limits: tuple[float, float],
    mse_max: float,
    *,
    human_intervals: pd.DataFrame | None = None,
) -> plt.Figure:
    """Render factor-level profile curves and MSE strips for a specific illumination operation.

    Parameters
    ----------
    result : dict[str, pd.DataFrame]
        Analysis outputs containing profile and error tables.
    levels : pd.DataFrame
        Illumination factor levels table.
    operation : str
        Operation identifier ('shadow', 'reflection', 'lightdir').
    output_dir : Path
        Destination directory to save figures.
    score_limits : tuple[float, float]
        Horizontal axis limits for score profile curves.
    mse_max : float
        Maximum upper limit for MSE strips.
    human_intervals : pd.DataFrame | None, optional
        Precomputed human factor bootstrap confidence intervals.

    Returns
    -------
    plt.Figure
        Rendered figure handle.
    """
    local_levels = levels[levels.operation.eq(operation)]
    factors = local_levels.drop_duplicates("factor_id")
    heights = local_levels.groupby("factor_id", sort=False).size().to_numpy() + 2
    fig = plt.figure(figsize=(12, 2.0 + .39 * sum(heights)))
    grid = fig.add_gridspec(len(factors), 4, height_ratios=heights,
                           left=.21, right=.985, bottom=.11, top=.87, hspace=.50, wspace=.15)
    profiles = result["factor_level_profiles"]
    medians = result["profile_medians"]
    errors = result["mse_by_model_factor"]
    for row, factor in enumerate(factors.itertuples()):
        labels = local_levels[local_levels.factor_id.eq(factor.factor_id)].sort_values("level_order").level_label
        human = profiles.query("factor_id == @factor.factor_id and group == 'human'").sort_values("level_order")
        interval = None
        if human_intervals is not None:
            interval = human_intervals[human_intervals.factor_id.eq(factor.factor_id)].sort_values("level_order")
        y = np.arange(len(labels))
        for col, group in enumerate(GROUPS):
            cell = grid[row, col].subgridspec(2, 1, height_ratios=(len(labels) + .7, 1), hspace=.4)
            axis = fig.add_subplot(cell[0])
            strip = fig.add_subplot(cell[1])
            local = profiles.query("factor_id == @factor.factor_id and group == @group")
            for rank, model in local.groupby("rank", sort=True):
                model = model.sort_values("level_order")
                axis.plot(model.score, y + (rank - 3) * .055, color=COLORS[group],
                          lw=.65, alpha=.26, zorder=2)
                axis.scatter(model.score, y + (rank - 3) * .055, color=COLORS[group], s=10, alpha=.55, zorder=3)
            median = medians.query("factor_id == @factor.factor_id and group == @group").sort_values("level_order")
            axis.plot(human.score, y, color="#222222", marker="o", markerfacecolor="white", markersize=4, lw=.85, zorder=5)
            if interval is not None:
                # Draw endpoints directly: percentile intervals need not contain the estimate.
                axis.hlines(y, interval.ci_low, interval.ci_high, color="#222222", lw=.75, zorder=4.8)
                axis.vlines(interval.ci_low, y - .09, y + .09, color="#222222", lw=.75, zorder=4.8)
                axis.vlines(interval.ci_high, y - .09, y + .09, color="#222222", lw=.75, zorder=4.8)
            axis.plot(median["median"], y, color=COLORS[group], marker="D", markersize=4, lw=.9, zorder=4)
            axis.axvline(0, color="#aaaaaa", lw=.6, ls="--")
            axis.set(xlim=score_limits, ylim=(len(labels) - .55, -.55), xticks=np.arange(np.ceil(score_limits[0]), score_limits[1], 2))
            axis.set_yticks(y, labels if col == 0 else [""] * len(labels))
            axis.tick_params(axis="y", length=0)
            axis.tick_params(axis="x", labelsize=7, length=2)
            axis.grid(axis="x", color="#eeeeee", lw=.6)
            axis.spines[["top", "right", "left"]].set_visible(False)
            if col == 0:
                axis.set_title(f"{factor.factor_code}  {factor.factor_label}", loc="left", fontsize=9, fontweight="bold", pad=7)
            values = errors.query("factor_id == @factor.factor_id and group == @group").sort_values("rank").mse
            strip.set_facecolor("#edf0f3")
            mse_points(strip, values, 0, COLORS[group])
            strip.set(xlim=(-.03 * mse_max, mse_max), ylim=(-.35, .35), yticks=[], xticks=np.arange(0, mse_max + .1, 2))
            strip.tick_params(axis="x", labelsize=7, length=2)
            strip.spines[["top", "right", "left"]].set_visible(False)
            if col == 0:
                strip.set_ylabel("MSE", rotation=0, ha="right", va="center", labelpad=8, fontsize=8)
            if row == 0:
                box = axis.get_position()
                fig.text((box.x0 + box.x1) / 2, .935, LABELS[group] + "\n" + ("Top 5" if group in GROUPS[:2] else "All 5"),
                         ha="center", va="top", fontsize=10, color=COLORS[group])
    fig.suptitle(f"{factors.operation_label.iloc[0]}: factor-level profiles", y=.985, fontsize=13)
    fig.text(.60, .067, "Profile x-axis: score / illumination-wide sample SD (no centering)", ha="center", fontsize=9)
    human_label = "Human (95% CI)" if human_intervals is not None else "Human"
    handles = [Line2D([], [], color="#222222", marker="o", markerfacecolor="white", lw=.8, label=human_label),
               Line2D([], [], color="#777777", marker="o", lw=.65, alpha=.55, markersize=3, label="Individual model"),
               Line2D([], [], color="#777777", marker="D", lw=.8, markersize=4, label="Median of five models")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.60, .033), ncol=3, frameon=False, fontsize=8)
    note = ("Human bars: participant-bootstrap 95% CIs. Grey strips: MSE. Model spread is descriptive."
            if human_intervals is not None else
            "Grey strips: MSE (squared score units). Model spread is descriptive; no confidence intervals.")
    fig.text(.60, .018, note,
             ha="center", fontsize=7.5)
    save(fig, output_dir, f"illumination_factor_profiles_{operation}_paired")
    return fig


def plot(result: dict[str, pd.DataFrame], levels: pd.DataFrame, output_dir: Path) -> None:
    """Render all illumination factor profile figures (overview and 3 operation figures).

    Parameters
    ----------
    result : dict[str, pd.DataFrame]
        Analysis results containing profiles, medians, and error metrics.
    levels : pd.DataFrame
        Illumination factor levels table.
    output_dir : Path
        Directory to save rendered PDF and PNG figures.
    """
    values = result["factor_level_profiles"].score
    intervals = result.get("human_intervals")
    if intervals is not None and len(intervals):
        score_min = min(values.min(), intervals.ci_low.min())
        score_max = max(values.max(), intervals.ci_high.max())
    else:
        score_min, score_max = values.min(), values.max()
    score_limits = (np.floor(score_min) - .15, np.ceil(score_max) + .15)
    mse_max = 2 * np.ceil(result["mse_by_model_factor"].mse.max() / 2)
    with plt.rc_context(figure_style(fontsize=9)):
        overview(result, levels, output_dir, mse_max)
        for operation in levels.operation.unique():
            operation_profiles(result, levels, operation, output_dir, score_limits, mse_max,
                               human_intervals=intervals)
