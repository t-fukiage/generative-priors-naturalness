"""Paths, configuration, and array loading for the scalar data release."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RELEASE = Path(__file__).resolve().parent


def load_settings() -> dict[str, Any]:
    """Load runtime path and visualization configuration from settings.json.

    Returns
    -------
    dict[str, Any]
        Configuration mapping loaded from settings.json.
    """
    path = RELEASE / "settings.json"
    if path.exists():
        return json.loads(path.read_text())
    raise FileNotFoundError(f"settings.json not found in {RELEASE}")


def configured_path(name: str, override: Path | str | None = None) -> Path:
    """Resolve a configured directory path relative to the release root.

    Parameters
    ----------
    name : str
        Key name in settings.json (e.g., 'data_root', 'output_root').
    override : Path | str | None, optional
        Optional path override provided via CLI arguments.

    Returns
    -------
    Path
        Resolved absolute Path.
    """
    if override is not None:
        return Path(override).expanduser().resolve()
    settings = load_settings()
    path = Path(settings[name]).expanduser()
    return (path if path.is_absolute() else RELEASE / path).resolve()


def configured_fonts() -> list[str]:
    """Retrieve font family fallback preferences for matplotlib figures.

    Returns
    -------
    list[str]
        List of font family names in order of preference.
    """
    settings = load_settings()
    fonts = settings.get("font_family", ["Arial", "DejaVu Sans"])
    return [fonts] if isinstance(fonts, str) else list(fonts)


def read_table(path: Path | str) -> list[dict[str, str]]:
    """Read a CSV or gzip-compressed CSV table into a list of row dicts.

    Parameters
    ----------
    path : Path | str
        Path to plain (.csv) or compressed (.csv.gz) table.

    Returns
    -------
    list[dict[str, str]]
        List of dictionary rows.
    """
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_losses(model: str, data_root: Path | str | None = None) -> dict[str, np.ndarray]:
    """Load precomputed layer/timestep loss arrays for a generative model.

    Parameters
    ----------
    model : str
        Generative model key identifier.
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of array names to loaded numpy arrays from the model NPZ file.
    """
    root = configured_path("data_root", data_root)
    loss_path = root / "losses" / f"{model}.npz"
    if not loss_path.exists():
        raise FileNotFoundError(f"Loss file not found for model: {model} at {loss_path}")
    with np.load(loss_path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def paired_scores(losses: dict[str, np.ndarray]) -> np.ndarray:
    """Mean modified loss minus mean original loss, retaining float64 precision.

    Parameters
    ----------
    losses : dict[str, np.ndarray]
        Dictionary containing 'loss_mean' array of shape [pair, role, timestep].

    Returns
    -------
    np.ndarray
        1D array of modified-minus-original scores across stimulus pairs.
    """
    means = losses["loss_mean"].mean(axis=2, dtype=np.float64)
    return means[:, 1] - means[:, 0]


def load_pair_metadata(data_root: Path | str | None = None) -> pd.DataFrame:
    """Load the master stimulus pair table and condition assignments.

    Parameters
    ----------
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    pd.DataFrame
        Table of 568 stimulus pairs with task, condition, and image path metadata.
    """
    root = configured_path("data_root", data_root)
    path = root / "pair_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"Pair metadata not found at {path}")
    return pd.read_csv(path)


def load_human_pairs(data_root: Path | str | None = None) -> pd.DataFrame:
    """Convenience facade: Screen individual responses and compute original-minus-modified image means.

    Delegates to analysis.human.aggregate_pairs with dynamic import to preserve clean modular layering.

    Parameters
    ----------
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    pd.DataFrame
        Pair table joined with human perceptual difference scores and task labels.
    """
    from analysis.human import aggregate_pairs
    return aggregate_pairs(load_pair_metadata(data_root), data_root=data_root)


def load_generative_means(
    pairs: pd.DataFrame, data_root: Path | str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read draw means [pair, role, timestep], aligned by pair ID, and model metadata.

    Parameters
    ----------
    pairs : pd.DataFrame
        Reference pair metadata table specifying row order.
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, Any]]
        (means, models)
        means  : Mapping from model key to loss arrays aligned with pairs.
        models : Model configuration catalog for generative models.
    """
    root = configured_path("data_root", data_root)
    models = json.loads((root / "models.json").read_text())["generative"]
    means = {}
    for model in sorted(models):
        loss_path = root / "losses" / f"{model}.npz"
        with np.load(loss_path, allow_pickle=False) as saved:
            pair_ids = saved["pair_ids"]
            losses = saved["loss_mean"]
        positions = pd.Index(pair_ids).get_indexer(pairs["pair_id"])
        means[model] = losses[positions]
    return means, models


def load_generative_scores(
    pairs: pd.DataFrame, data_root: Path | str | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return timestep-averaged scores as a pair-by-model table, plus metadata.

    Parameters
    ----------
    pairs : pd.DataFrame
        Reference pair metadata table specifying row order.
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        (scores_df, models_catalog)
    """
    means, models = load_generative_means(pairs, data_root)
    scores = {model: paired_scores({"loss_mean": values}) for model, values in means.items()}
    return pd.DataFrame(scores, index=pairs["pair_id"]), models


def load_model_candidates(
    pairs: pd.DataFrame, data_root: Path | str | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load generative scores and encoder candidates in a common pair order.

    Columns and metadata have matching positions. Models are alphabetical within
    each group. Encoder candidates put the endpoint first, then numbered blocks;
    this order also defines how exact ties in layer selection are resolved.

    Parameters
    ----------
    pairs : pd.DataFrame
        Reference pair metadata table specifying row order.
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        (values, metadata)
        values   : Score array of shape [N_pairs, N_candidates].
        metadata : DataFrame describing group, model, candidate, and readout.
    """
    root = configured_path("data_root", data_root)
    generative, _ = load_generative_scores(pairs, root)
    models = json.loads((root / "models.json").read_text())
    name = "encoder_distances.csv.gz"
    encoders = pd.read_csv(root / name, dtype={"candidate": str}, float_precision="round_trip")
    encoder_scores = encoders.pivot(index="pair_id", columns=["model", "candidate"], values="score")
    columns, records = [], []

    def append(group, model, candidate, readout, values):
        columns.append(values.loc[pairs.pair_id].to_numpy(float))
        records.append({"group": group, "model": model, "candidate": candidate, "readout": readout})

    for model in sorted(generative):
        append("generative", model, "score", "mean_loss", generative[model])
    for model, config in sorted(models["encoders"].items()):
        candidates = ["endpoint", *(str(layer) for layer in range(1, config["layers"] + 1))]
        for candidate in candidates:
            readout = "endpoint" if candidate == "endpoint" else config["readout"]
            append("encoder", model, candidate, readout, encoder_scores[(model, candidate)])
    values = np.column_stack(columns)
    return values, pd.DataFrame(records)


def load_predictor_candidates(
    pairs: pd.DataFrame, data_root: Path | str | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Append signed FR/NR-IQA scores to the generative and encoder candidates.

    Parameters
    ----------
    pairs : pd.DataFrame
        Reference pair metadata table specifying row order.
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        (values, metadata)
        values   : Concatenated score matrix of shape [N_pairs, N_candidates].
        metadata : Descriptor table matching columns of values.
    """
    root = configured_path("data_root", data_root)
    values, metadata = load_model_candidates(pairs, root)
    models = json.loads((root / "models.json").read_text())
    name = "iqa_scores.csv.gz"
    iqa = pd.read_csv(root / name, float_precision="round_trip")
    iqa_scores = iqa.pivot(index="pair_id", columns="model", values="score")
    columns, records = [values], metadata.to_dict("records")
    for mode, group in (("FR", "fr_iqa"), ("NR", "nr_iqa")):
        names = sorted(model for model, config in models["iqa"].items() if config["mode"] == mode)
        for model in names:
            columns.append(iqa_scores.loc[pairs.pair_id, model].to_numpy(float))
            records.append({"group": group, "model": model, "candidate": "score", "readout": "signed_iqa"})
    values = np.column_stack(columns)
    return values, pd.DataFrame(records)


def load_human_split_half(
    data_root: Path | str | None = None, bootstrap_root: Path | str | None = None
) -> pd.DataFrame:
    """Convenience facade: Read precomputed human split-half noise ceiling reliability tables.

    Parameters
    ----------
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).
    bootstrap_root : Path | str | None, optional
        Root directory containing bootstrap/ (defaults to settings.json).

    Returns
    -------
    pd.DataFrame
        Table of human split-half correlation statistics by task and scope.
    """
    from analysis.uncertainty import load_tables
    return load_tables(data_root=data_root, bootstrap_root=bootstrap_root)["human_split_half"]


def load_illumination_factors(
    pairs: pd.DataFrame | None = None, data_root: Path | str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ordered factor levels and their pair membership, without scores.

    Parameters
    ----------
    pairs : pd.DataFrame | None, optional
        Unused, maintained for backwards compatibility.
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (levels, membership)
        levels     : Catalog of 13 factors and 44 ordered levels.
        membership : Pair-to-factor classification mapping.
    """
    root = configured_path("data_root", data_root) / "illumination_factors"
    levels = pd.read_csv(root / "levels.csv").sort_values(["factor_order", "level_order"])
    members = pd.read_csv(root / "membership.csv")
    return levels, members
