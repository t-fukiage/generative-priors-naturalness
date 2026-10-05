"""Whole-participant and whole-scene sampling for the paired design."""

from typing import Tuple
import numpy as np


def participant_weights(
    strata: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Multinomial resample of participant indices within each stratum.

    Parameters
    ----------
    strata : np.ndarray
        Array of stratum labels for each participant (length N).
    size : int
        Number of bootstrap draws to generate (B).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Weight matrix of shape [B, N] containing frequency counts per participant.
    """
    weights = np.zeros((size, len(strata)))
    for stratum in np.unique(strata):
        columns = np.flatnonzero(strata == stratum)
        prob = np.full(len(columns), 1.0 / len(columns))
        weights[:, columns] = rng.multinomial(len(columns), prob, size=size)
    return weights


def split_weights(
    strata: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate disjoint split-half partition weights across strata.

    In strata with an odd number of participants, the first half receives
    the single extra participant.

    Parameters
    ----------
    strata : np.ndarray
        Stratum identifier per participant (length N).
    size : int
        Number of split-half draws (B).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Binary indicator matrices (half_a, half_b), each of shape [B, N].
    """
    sorted_unique_strata = sorted(np.unique(strata), key=lambda s: int(s.rsplit(":", 1)[1]))
    groups = [np.flatnonzero(strata == s) for s in sorted_unique_strata]
    a = np.zeros((size, len(strata)))
    for row in a:
        for group in groups:
            order = group.copy()
            rng.shuffle(order)
            half_size = (len(order) + 1) // 2
            row[order[:half_size]] = 1.0
    return a, 1.0 - a


def weighted_means(responses: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted column means ignoring non-finite ratings.

    Parameters
    ----------
    responses : np.ndarray
        Participant response matrix of shape [N, items].
    weights : np.ndarray
        Participant resampling weights of shape [B, N].

    Returns
    -------
    np.ndarray
        Weighted means matrix of shape [B, items].
    """
    counts = weights @ np.isfinite(responses).astype(float)
    totals = weights @ np.nan_to_num(responses)
    if np.any(counts == 0):
        raise ValueError("A resample has an unrated image; no draws or images are dropped.")
    return totals / counts


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute frequency-weighted Pearson correlation along axis 1.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays of shape [B, items].
    weights : np.ndarray
        Frequency weights of shape [B, items] or broadcastable.

    Returns
    -------
    np.ndarray
        Vector of correlations of shape [B].
    """
    n = weights.sum(axis=1, keepdims=True)
    mean_x = (weights * x).sum(axis=1, keepdims=True) / n
    mean_y = (weights * y).sum(axis=1, keepdims=True) / n
    a = x - mean_x
    b = y - mean_y
    sum_a2 = (weights * a * a).sum(axis=1)
    sum_b2 = (weights * b * b).sum(axis=1)
    denominator = np.sqrt(sum_a2 * sum_b2)
    if np.any(denominator <= 0):
        raise ValueError("A resample has zero variance.")
    sum_ab = (weights * a * b).sum(axis=1)
    return np.clip(sum_ab / denominator, -1.0, 1.0)
