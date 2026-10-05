"""Statistics shared by the paper analyses; no file access or random sampling."""

import numpy as np


def sensitivity(task_scores, condition_mask=None):
    """Mean score / task-wide sample SD, with pairs on axis 0.

    A condition mask selects the numerator only. A constant task has undefined
    sensitivity (NaN). Scores are not centered before taking their mean.
    """
    scores = np.asarray(task_scores, dtype=np.float64)
    if scores.ndim not in (1, 2) or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Sensitivity requires finite scores from at least two pairs.")
    selected = scores
    if condition_mask is not None:
        mask = np.asarray(condition_mask)
        if mask.dtype != bool or mask.shape != (len(scores),) or not mask.any():
            raise ValueError("Expected a nonempty condition mask over all task pairs.")
        selected = scores[mask]
    mean, sd = selected.mean(axis=0), scores.std(axis=0, ddof=1)
    valid = (sd > 0) & (np.ptp(scores, axis=0) > 0)
    return np.divide(mean, sd, out=np.full_like(mean, np.nan), where=valid)


def pearson(x, y):
    """Signed Pearson correlation down axis 0; constant columns return NaN.

    Inputs may be vectors or pair-by-column matrices. A vector is broadcast
    over the other input's columns.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim not in (1, 2) or y.ndim not in (1, 2) or len(x) != len(y) or len(x) < 3:
        raise ValueError("Pearson correlation requires matching inputs with at least three pairs.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Correlation inputs must be finite.")
    if x.ndim < y.ndim:
        x = x[:, None]
    if y.ndim < x.ndim:
        y = y[:, None]
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    xx, yy = np.sum(x * x, axis=0), np.sum(y * y, axis=0)
    denominator = np.sqrt(xx * yy)
    valid = (xx > 0) & (yy > 0) & (np.ptp(x, axis=0) > 0) & (np.ptp(y, axis=0) > 0)
    numerator = np.sum(x * y, axis=0)
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=valid)
    return np.clip(result, -1, 1)


def center_within(values, conditions):
    """Subtract each condition's mean, keeping rows and columns in input order."""
    centered = np.asarray(values, dtype=float).copy()
    conditions = np.asarray(conditions)
    if conditions.shape != (len(centered),):
        raise ValueError("Expected one condition label per row.")
    for condition in np.unique(conditions):
        mask = conditions == condition
        centered[mask] -= centered[mask].mean(axis=0)
    return centered


def residualize(values, covariates):
    """OLS residuals after fitting an intercept and the supplied covariates.

    Rows are observations; a vector or a matrix of outcome columns is accepted.
    Residual sums of squares <= 1e-12 of the centered input are set to zero,
    so a subsequent correlation is undefined for a numerically perfect fit.
    """
    values = np.asarray(values, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    if (values.ndim not in (1, 2) or covariates.ndim != 2 or len(values) != len(covariates)
            or len(values) < covariates.shape[1] + 3):
        raise ValueError("Expected matching outcome and covariate rows with positive residual degrees of freedom.")
    if not np.isfinite(values).all() or not np.isfinite(covariates).all():
        raise ValueError("Regression inputs must be finite.")
    design = np.column_stack([np.ones(len(values)), covariates])
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Covariates are constant or linearly dependent.")
    residuals = values - design @ coefficients
    residuals -= residuals.mean(axis=0)
    centered = values - values.mean(axis=0)
    negligible = ((np.ptp(values, axis=0) == 0)
                  | (np.sum(residuals**2, axis=0) <= 1e-12 * np.sum(centered**2, axis=0)))
    return np.where(negligible, 0.0, residuals)


def human_correlations(human, scores):
    """Marginal human correlations and the predictor correlation matrix.

    Normalizing centered columns before multiplication preserves the scale
    invariance of correlation for small encoder distances. Constants give NaN.
    """
    human, scores = np.asarray(human, dtype=float), np.asarray(scores, dtype=float)
    if human.ndim != 1 or scores.ndim != 2 or len(human) != len(scores) or len(human) < 4:
        raise ValueError("Expected a human vector and predictor matrix with at least four rows.")
    values = np.column_stack([human, scores])
    if not np.isfinite(values).all():
        raise ValueError("Correlation inputs must be finite.")
    centered = values - values.mean(axis=0)
    norm = np.sqrt(np.sum(centered**2, axis=0))
    normalized = np.divide(centered, norm, out=np.full_like(centered, np.nan),
                           where=(norm > 0) & (np.ptp(values, axis=0) > 0))
    correlations = np.clip(normalized.T @ normalized, -1, 1)
    return correlations[0, 1:], correlations[1:, 1:]


def bidirectional_partial(human_g, human_x, g_x):
    """Return r(h, G | X) and r(h, X | G) for every G-by-X pair.

    The one-covariate Pearson formula equals correlation of intercept-inclusive
    OLS residuals. A residual variance fraction <= 1e-12 is undefined (NaN).
    """
    hg, hx = np.asarray(human_g)[:, None], np.asarray(human_x)[None, :]
    gx = np.asarray(g_x)

    def partial(target, control):
        human_variance, predictor_variance = 1 - control**2, 1 - gx**2
        valid = (human_variance > 1e-12) & (predictor_variance > 1e-12)
        denominator = np.sqrt(np.maximum(0, human_variance * predictor_variance))
        value = np.divide(target - control * gx, denominator, out=np.full_like(gx, np.nan), where=valid)
        return np.clip(value, -1, 1)

    return partial(hg, hx), partial(hx, hg)
