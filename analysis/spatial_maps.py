"""Figures S25–S27 from scaled, cropped, and resized condition-mean maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_io import configured_path
from analysis.plotting import figure_style

CONDITIONS = (("thatcher", "str", "Upright"), ("thatcher", "inv", "Inverted"),
              ("swap", "shadow_swap", "Shadow"), ("swap", "reflection_swap", "Reflection"),
              ("swap", "lightdir_swap", "Light direction"))
STEMS = ("spatial_maps_all25_image_1", "spatial_maps_all25_image_2", "spatial_maps_all25_video")


def load_maps(
    root: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], pd.DataFrame, float]:
    """Load precomputed spatial condition-mean loss maps, models, and scaling tables.

    Parameters
    ----------
    root : Path
        Directory containing spatial maps files (data/spatial_maps).

    Returns
    -------
    tuple[dict[str, np.ndarray], list[dict[str, Any]], pd.DataFrame, float]
        (maps, models, scales, limit)
        maps   : Mapping of 'task__condition__model' to [96, 96] spatial loss maps.
        models : Model catalog list specifying display names, order, and modalities.
        scales : DataFrame of task-wide sample standard deviations.
        limit  : 99.5th percentile absolute value across all maps used for symmetric color limits.
    """
    models = json.loads((root / "models.json").read_text())
    scales = pd.read_csv(root / "task_scales.csv")
    with np.load(root / "condition_maps.npz", allow_pickle=False) as archive:
        maps = {key: archive[key] for key in archive.files}
    limit = float(np.quantile(np.concatenate([np.abs(a).ravel() for a in maps.values()]), .995, method="linear"))
    return maps, models, scales, limit


def plot(
    maps: dict[str, np.ndarray],
    models: list[dict[str, Any]],
    limit: float,
    output_dir: Path,
) -> None:
    """Render Figures S25–S27 (image models 1, image models 2, and video models).

    Parameters
    ----------
    maps : dict[str, np.ndarray]
        Spatial error maps keyed by 'task__condition__model'.
    models : list[dict[str, Any]]
        List of model descriptor records.
    limit : float
        Symmetric color normalization limit.
    output_dir : Path
        Directory to save rendered PDF and PNG figures.
    """
    images = [m for m in models if m["modality"] == "image"]
    videos = [m for m in models if m["modality"] == "video"]
    norm = plt.Normalize(-limit, limit)
    plotted = set()
    with plt.rc_context(figure_style(fontsize=8)):
        for stem, group in zip(STEMS, (images[:8], images[8:], videos)):
            height = .70 * len(group) + .90
            figure = plt.figure(figsize=(5.5, height))
            grid = figure.add_gridspec(len(group), 5, left=.265, right=.995,
                                      bottom=.63 / height, top=1 - .30 / height, wspace=.075, hspace=.075)
            for row, model in enumerate(group):
                for column, (task, condition, label) in enumerate(CONDITIONS):
                    key = f"{task}__{condition}__{model['model']}"
                    axis = figure.add_subplot(grid[row, column])
                    handle = axis.imshow(maps[key], cmap="RdBu_r", norm=norm, interpolation="nearest")
                    plotted.add(key)
                    axis.set_xticks([])
                    axis.set_yticks([])
                    for spine in axis.spines.values():
                        spine.set_visible(False)
                    if row == 0:
                        axis.set_title(label, fontsize=8, pad=6)
                    if column == 0:
                        name = model["display_name"].replace("Stable Diffusion", "SD")
                        name = name.replace("LTX-Video", "LTX").replace("HunyuanVideo", "Hunyuan")
                        axis.text(-.13, .5, name, transform=axis.transAxes, ha="right", va="center", fontsize=7.1)
            figure.text(.407, 1 - .01 / height, "Thatcher", ha="center", va="top", weight="bold")
            figure.text(.782, 1 - .01 / height, "Illumination", ha="center", va="top", weight="bold")
            color_axis = figure.add_axes([.39, .35 / height, .44, .085 / height])
            bar = figure.colorbar(handle, cax=color_axis, orientation="horizontal", ticks=[-limit, 0, limit])
            bar.ax.set_xticklabels([f"−{limit:.1f}", "0", f"{limit:.1f}"])
            bar.ax.tick_params(labelsize=7, length=2, pad=2)
            bar.set_label("Mean loss change / task score SD", fontsize=8, labelpad=2)
            for extension in ("png", "pdf"):
                figure.savefig(output_dir / f"{stem}.{extension}", dpi=250, facecolor="white")
            plt.close(figure)


def main() -> None:
    """CLI entry point for rendering Figures S25–S27 spatial error maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing data (default: settings.json)")
    parser.add_argument("--spatial-root", type=Path, default=None, help="Direct override for spatial maps directory (default: DATA_ROOT/spatial_maps)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/analysis/spatial_maps)")
    args = parser.parse_args()

    data_root = configured_path("data_root", args.data_root)
    root = args.spatial_root.expanduser().resolve() if args.spatial_root else data_root / "spatial_maps"
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path("output_root") / "analysis/spatial_maps"
    maps, models, scales, limit = load_maps(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        for task, condition, _ in CONDITIONS:
            key = f"{task}__{condition}__{model['model']}"
            values = maps[key]
            scale = scales[(scales.task == task) & (scales.model == model["model"])].iloc[0]
            rows.append({"model": model["model"], "task": task, "condition": condition,
                         "n_pairs": 140 if task == "thatcher" else 96,
                         "task_sample_sd": scale.task_sample_sd, "minimum": float(values.min()),
                         "maximum": float(values.max()), "mean": float(values.mean(dtype=np.float64)),
                         "saturated_fraction": float(np.mean(np.abs(values) > limit))})
    pd.DataFrame(rows).to_csv(output_dir / "map_summary.csv", index=False)
    plot(maps, models, limit, output_dir)
    print(f"Saved Figures S25–S27 and summary to {output_dir} (shared color limit: ±{limit:.8f})")


if __name__ == "__main__":
    main()
