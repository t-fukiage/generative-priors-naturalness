"""Participant screening, image means, and paired response matrices."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from data_io import configured_path, load_generative_scores, load_pair_metadata

TASKS = ("thatcher", "swap")


def load_responses(data_root: Path | str | None = None) -> pd.DataFrame:
    """Load individual participant rating responses from responses.csv.

    Parameters
    ----------
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    pd.DataFrame
        DataFrame of all individual participant ratings and presentation metadata.
    """
    root = configured_path("data_root", data_root) / "human_responses"
    path = root / "responses.csv"
    if not path.exists():
        path = root / "responses.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Human responses not found at {path}")
    return pd.read_csv(path)


def first_presentations(responses: pd.DataFrame) -> pd.DataFrame:
    """Retain only the first submission per participant and first presentation per image.

    Parameters
    ----------
    responses : pd.DataFrame
        Raw response table including repeats and duplicates.

    Returns
    -------
    pd.DataFrame
        Filtered response table with repeat presentations and resubmissions dropped.
    """
    ordered = responses.sort_values(["participant", "submission", "trial"])
    first = ordered.submission.eq(ordered.groupby("participant").submission.transform("min"))
    return ordered.loc[first].drop_duplicates(["participant", "pair_id", "version"]).copy()


def gaussian(values: np.ndarray, points: np.ndarray, bandwidth: float, derivative: int = 0) -> np.ndarray:
    """Evaluate Gaussian kernel density and its analytical derivatives.

    Parameters
    ----------
    values : np.ndarray
        Sample observations (1D array of length N).
    points : np.ndarray
        Evaluation points (1D array of length M).
    bandwidth : float
        Kernel bandwidth (h).
    derivative : int
        0 for density, 1 for first derivative, 2 for second derivative.

    Returns
    -------
    np.ndarray
        Kernel density estimate or derivative values of shape [M].
    """
    z = (np.atleast_1d(points)[:, None] - values[None, :]) / bandwidth
    density = np.exp(-z * z / 2.0) / (np.sqrt(2.0 * np.pi) * bandwidth)
    if derivative == 1:
        density *= -z / bandwidth
    elif derivative == 2:
        density *= (z * z - 1.0) / (bandwidth ** 2)
    return density.mean(axis=1)


def screening_cutoff(correlations: np.ndarray) -> tuple[float, float]:
    """Silverman bw.nrd0; choose the deepest trough relative to adjacent modes.

    Finds the automatic cutoff separating attentive observers from guessing/inconsistent
    observers by identifying the deepest trough in the leave-one-out correlation density.

    Parameters
    ----------
    correlations : np.ndarray
        Array of leave-one-out correlation values.

    Returns
    -------
    tuple[float, float]
        (cutoff_threshold, silverman_bandwidth)
    """
    values = np.asarray(correlations, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("Screening needs at least two finite correlations.")

    # 1. Silverman rule-of-thumb bandwidth (bw.nrd0)
    sd = values.std(ddof=1)
    iqr_scaled = np.subtract(*np.percentile(values, [75, 25])) / 1.34
    scale = min(sd, iqr_scaled)
    if scale == 0:
        scale = sd or abs(values[0]) or 1.0
    bandwidth = 0.9 * scale * len(values) ** (-0.2)

    # 2. Grid search for sign changes in 1st derivative (zero-crossings)
    grid = np.linspace(-1, 1, 10001)
    derivative = gaussian(values, grid, bandwidth, derivative=1)
    roots = []
    for i in range(len(grid) - 1):
        if derivative[i] * derivative[i + 1] < 0:
            # Refine root location with high precision using Brent's method
            root = brentq(lambda r: gaussian(values, [r], bandwidth, 1)[0],
                          grid[i], grid[i + 1], xtol=1e-13)
            roots.append(root)
        elif derivative[i] == 0 and i > 0 and derivative[i - 1] * derivative[i + 1] < 0:
            roots.append(grid[i])

    # 3. Classify critical points using 2nd derivative: is_minimum = (2nd derivative > 0)
    extrema = []
    for r in sorted(set(roots)):
        density_at_r = gaussian(values, [r], bandwidth, derivative=0)[0]
        is_minimum = gaussian(values, [r], bandwidth, derivative=2)[0] > 0
        extrema.append((r, density_at_r, is_minimum))

    # 4. Filter for troughs between two modes and score by depth relative to adjacent peaks
    troughs = []
    for i, (r, density, is_minimum) in enumerate(extrema):
        if is_minimum and 0 < i < len(extrema) - 1:
            left_is_mode = not extrema[i - 1][2]
            right_is_mode = not extrema[i + 1][2]
            if left_is_mode and right_is_mode:
                adjacent_min_mode = min(extrema[i - 1][1], extrema[i + 1][1])
                relative_depth = density / adjacent_min_mode
                troughs.append((relative_depth, r))

    if not troughs:
        raise ValueError("No KDE trough between two modes; no automatic cutoff is defined.")

    # Select the trough with the minimal relative density (deepest valley)
    best_trough_r = min(troughs)[1]
    return best_trough_r, bandwidth


@lru_cache(maxsize=4)
def screened_responses(
    data_root: Path | str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute leave-one-out KDE screening across Thatcher and Illumination tasks.

    Parameters
    ----------
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (trials, decisions, summaries)
        trials    : First-presentation trial records.
        decisions : Participant-level screening decisions with loo_r and retention flag.
        summaries : Task-level screening summary counts, cutoffs, and bandwidths.
    """
    trials = first_presentations(load_responses(data_root))
    decisions, summaries = [], []
    for task in TASKS:
        local = trials.loc[trials.task.eq(task)]
        totals = local.groupby(["pair_id", "version"]).rating.agg(["sum", "count"])
        joined = local.join(totals, on=["pair_id", "version"])
        quality = []
        for participant, group in joined.groupby("participant", sort=True):
            x = group.rating.to_numpy(float)
            y = (group["sum"].to_numpy() - x) / (group["count"].to_numpy() - 1)
            valid = np.isfinite(x) & np.isfinite(y)
            r = (np.corrcoef(x[valid], y[valid])[0, 1]
                 if valid.sum() >= 3 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0 else np.nan)
            quality.append({"task": task, "participant": participant,
                            "assignment_group": int(group.assignment_group.iloc[0]), "loo_r": r})
        quality = pd.DataFrame(quality)
        cutoff, bandwidth = screening_cutoff(quality.loo_r)
        quality["retained"] = quality.loo_r.ge(cutoff)
        decisions.append(quality)
        summaries.append({"task": task, "n_participants": len(quality), "retained": int(quality.retained.sum()),
                          "excluded": int((~quality.retained).sum()), "undefined": int(quality.loo_r.isna().sum()),
                          "cutoff": cutoff, "bandwidth": bandwidth})
    return trials, pd.concat(decisions, ignore_index=True), pd.DataFrame(summaries)


def aggregate_pairs(
    pairs: pd.DataFrame,
    cohort: str = "retained",
    data_root: Path | str | None = None,
    screened: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute original-minus-modified image means for a designated participant cohort.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pair table defining pair_id and task labels.
    cohort : str, optional
        Cohort to aggregate: 'retained' (default), 'excluded', or 'all'.
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).
    screened : tuple, optional
        Precomputed output from screened_responses().

    Returns
    -------
    pd.DataFrame
        Pairs table populated with mean naturalness and original-minus-modified scores.
    """
    if cohort not in ("retained", "excluded", "all"):
        raise ValueError(f"Unknown participant cohort: {cohort}")
    if screened is None:
        screened = screened_responses(data_root)
    trials, decisions, _ = screened
    if cohort != "all":
        keys = decisions.loc[decisions.retained.eq(cohort == "retained"), "participant"]
        trials = trials.loc[trials.participant.isin(keys)]
    aggregate = trials.groupby(["pair_id", "version"]).rating.mean()
    result = pairs.copy()
    for version in ("original", "modified"):
        means = aggregate.xs(version, level="version").reindex(pairs.pair_id)
        result[f"{version}_mean_naturalness"] = means.to_numpy()
    result["score"] = result.original_mean_naturalness - result.modified_mean_naturalness
    if not np.isfinite(result.score).all():
        raise ValueError("An image pair has no ratings.")
    return result


def response_matrix(
    pairs: pd.DataFrame,
    cohort: str = "retained",
    data_root: Path | str | None = None,
    screened: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute participant-by-pair rating differences (original - modified) and strata labels.

    Parameters
    ----------
    pairs : pd.DataFrame
        Subset of pairs belonging to a single task.
    cohort : str, optional
        Target cohort: 'retained' (default), 'excluded', or 'all'.
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).
    screened : tuple, optional
        Precomputed output from screened_responses().

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rating_differences, strata)
        rating_differences : Matrix of shape [N_participants, N_pairs].
        strata             : Resampling stratification labels for balanced bootstrap.
    """
    if cohort not in ("retained", "excluded", "all"):
        raise ValueError(f"Unknown participant cohort: {cohort}")
    if screened is None:
        screened = screened_responses(data_root)
    trials, decisions, _ = screened

    task = pairs.task.unique()
    if len(task) != 1:
        raise ValueError("Response matrices are task-specific.")
    task_name = task[0]

    # Select participants belonging to the target task and cohort
    participants = decisions[decisions.task.eq(task_name)]
    if cohort != "all":
        participants = participants[participants.retained.eq(cohort == "retained")]

    # Filter trials for selected participants and specified image pairs
    task_trials = trials[trials.participant.isin(participants.participant) & trials.pair_id.isin(pairs.pair_id)]
    participants = participants[participants.participant.isin(task_trials.participant)].sort_values("participant")

    # Pivot rating matrices for original and modified images [N_participants, N_pairs]
    def extract_matrix(version: str) -> np.ndarray:
        sub = task_trials[task_trials.version.eq(version)]
        pivot = sub.pivot(index="participant", columns="pair_id", values="rating")
        return pivot.reindex(index=participants.participant, columns=pairs.pair_id).to_numpy(float)

    original_ratings = extract_matrix("original")
    modified_ratings = extract_matrix("modified")

    if not np.array_equal(np.isfinite(original_ratings), np.isfinite(modified_ratings)):
        raise ValueError("Unpaired missing ratings.")

    # Construct stratification labels ("<scope>:<assignment_group>") for balanced resampling
    if task_name == "thatcher":
        scopes = np.repeat("thatcher", len(participants))
    else:
        participant_conditions = (
            task_trials.drop_duplicates("participant")
            .set_index("participant")["pair_id"]
            .map(pairs.set_index("pair_id")["condition"])
        )
        scopes = participant_conditions.reindex(participants.participant).to_numpy()

    strata = np.array([f"{s}:{g}" for s, g in zip(scopes, participants.assignment_group)])
    rating_differences = original_ratings - modified_ratings
    return rating_differences, strata


def filtering_robustness(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    screened: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    data_root: Path | str | None = None,
) -> pd.DataFrame:
    """Evaluate sensitivity of model correlations to participant screening rules.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pair table defining pair_id, task, and conditions.
    scores : np.ndarray
        Model score matrix across stimulus pairs.
    screened : tuple, optional
        Precomputed output from screened_responses().
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    pd.DataFrame
        Table comparing screened vs. unfiltered ratings and alignment statistics.
    """
    from analysis.metrics import pearson, sensitivity

    if screened is None:
        screened = screened_responses(data_root)

    retained = aggregate_pairs(pairs, cohort="retained", screened=screened)
    all_participants = aggregate_pairs(pairs, cohort="all", screened=screened)
    excluded = aggregate_pairs(pairs, cohort="excluded", screened=screened)

    rows = []
    for task in TASKS:
        local_pairs = pairs.loc[pairs.task.eq(task)]
        scopes = ["pooled"] + list(local_pairs.condition.unique())
        for scope in scopes:
            if scope == "pooled":
                mask = pairs.task.eq(task).to_numpy()
            else:
                mask = (pairs.task.eq(task) & pairs.condition.eq(scope)).to_numpy()

            x = scores[mask]
            a = retained.loc[mask, "score"].to_numpy()
            b = all_participants.loc[mask, "score"].to_numpy()
            exc = excluded.loc[mask, "score"].to_numpy()

            sens = sensitivity(x)
            r_a = pearson(x, a)
            r_b = pearson(x, b)

            rows.append({
                "task": task,
                "scope": scope,
                "human_r": pearson(a, b),
                "screened_association": pearson(sens, r_a),
                "unfiltered_association": pearson(sens, r_b),
                "screened_median_alignment": float(np.median(r_a)),
                "unfiltered_median_alignment": float(np.median(r_b)),
                "screened_mean_effect": float(a.mean()),
                "excluded_mean_effect": float(exc.mean()),
            })

    return pd.DataFrame(rows)


def response_counts(
    pairs: pd.DataFrame,
    screened: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    data_root: Path | str | None = None,
) -> pd.DataFrame:
    """Compute participant counts and rating counts per condition and task.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pair table defining pair_id, task, and conditions.
    screened : tuple, optional
        Precomputed output from screened_responses().
    data_root : Path | str | None, optional
        Root directory containing data/ (defaults to settings.json).

    Returns
    -------
    pd.DataFrame
        Table summarizing total, retained, and excluded participant counts and ratings per pair.
    """
    if screened is None:
        screened = screened_responses(data_root)
    trials, decisions, _ = screened
    trials = trials.merge(pairs[["pair_id", "condition"]], on="pair_id", validate="many_to_one")
    trials = trials.merge(decisions[["participant", "retained"]], on="participant", validate="many_to_one")
    rows = []
    for task in TASKS:
        local = trials.loc[trials.task.eq(task)]
        scopes = ["pooled"] + list(local.condition.unique())
        for scope in scopes:
            if scope == "pooled":
                group = local
            else:
                group = local.loc[local.condition == scope]
            kept = group.loc[group.retained]
            counts = kept.groupby(["pair_id", "version"]).size()
            rows.append({
                "task": task,
                "scope": scope,
                "participants": group.participant.nunique(),
                "retained": kept.participant.nunique(),
                "excluded": group.loc[~group.retained].participant.nunique(),
                "ratings_min": int(counts.min()),
                "ratings_max": int(counts.max()),
                "ratings_mean": counts.mean(),
            })
    return pd.DataFrame(rows)


def plot_screening(
    decisions: pd.DataFrame, summary: pd.DataFrame, output_dir: Path
) -> None:
    """Render the leave-one-out consistency histogram and fitted KDE curves.

    Parameters
    ----------
    decisions : pd.DataFrame
        Participant screening table with loo_r and retention decisions.
    summary : pd.DataFrame
        Screening summary metrics per task (cutoffs, bandwidths).
    output_dir : Path
        Directory to save response_consistency_histogram.pdf and .png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.6), layout="constrained")
    for ax, row, title in zip(axes, summary.itertuples(), ("Thatcher", "Illumination")):
        values = decisions.loc[decisions.task.eq(row.task)]
        bins = np.arange(-.2, 1.02001, .04)
        ax.hist([values.loc[~values.retained, "loo_r"].dropna(), values.loc[values.retained, "loo_r"]],
                bins=bins, stacked=True, color=["#C65D5D", "#4C78A8"], label=["Excluded", "Retained"])
        grid = np.linspace(-.2, 1, 601)
        finite = values.loo_r.dropna().to_numpy()
        ax.plot(grid, gaussian(finite, grid, row.bandwidth) * len(finite) * .04, color="#39424E", lw=1)
        ax.axvline(row.cutoff, color="black", ls="--", lw=1)
        ax.set(title=f"{title} (n={row.n_participants})", xlabel="Leave-one-out Pearson r", xlim=(-.2, 1))
        ax.text(.02, .97, f"Cutoff {row.cutoff:.4f}\nUndefined: {row.undefined}", transform=ax.transAxes, va="top", fontsize=8)
    axes[0].set_ylabel("Participants")
    axes[1].legend(fontsize=7)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"response_consistency_histogram.{extension}", dpi=200)
    plt.close(fig)


def main() -> None:
    """CLI entry point for participant screening and aggregate computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing data (default: settings.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/analysis/human)")
    args = parser.parse_args()

    data_root = configured_path("data_root", args.data_root)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path("output_root") / "analysis/human"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pair_metadata(data_root=data_root)
    screened = screened_responses(data_root=data_root)
    trials, decisions, summary = screened

    decisions.to_csv(output_dir / "screening.csv", index=False)
    summary.to_csv(output_dir / "screening_summary.csv", index=False)
    aggregate_pairs(pairs, screened=screened).to_csv(output_dir / "human_pairs.csv", index=False)
    response_counts(pairs, screened=screened).to_csv(output_dir / "response_counts.csv", index=False)
    scores, _ = load_generative_scores(pairs, data_root=data_root)
    filtering_robustness(pairs, scores.to_numpy(), screened=screened).to_csv(output_dir / "filtering_robustness.csv", index=False)
    plot_screening(decisions, summary, output_dir)
    print(summary.to_string(index=False))
    print(f"Saved screening summary, aggregate ratings, and histogram to {output_dir}")


if __name__ == "__main__":
    main()
