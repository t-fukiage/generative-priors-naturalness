"""Task-wide encoder-layer and model selection shared by comparison analyses."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


GROUPS = ("generative", "encoder", "fr_iqa", "nr_iqa")


def candidate_sets(metadata: pd.DataFrame) -> Dict[str, List[np.ndarray]]:
    """Group column positions by model, preserving the documented tie order.

    Parameters
    ----------
    metadata : pd.DataFrame
        Table of candidate predictors containing 'group' and 'model' columns.

    Returns
    -------
    Dict[str, List[np.ndarray]]
        Mapping from model group name to a list of candidate column index arrays,
        one array per distinct model sorted alphabetically.
    """
    candidates = {}
    for group in GROUPS:
        group_meta = metadata[metadata.group.eq(group)]
        grouped = group_meta.groupby("model", sort=True)
        candidates[group] = [frame.index.to_numpy() for _, frame in grouped]
    return candidates


def select_model_candidates(
    correlations: np.ndarray,
    candidates: Dict[str, List[np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Choose each model's candidate with the largest signed task-wide r.

    All correlations must describe the entire task, including in bootstrap
    samples. Endpoint/ascending-block order resolves layer ties; alphabetical
    model order resolves model ties. Undefined candidates are excluded.

    Parameters
    ----------
    correlations : np.ndarray
        Vector of task-wide Pearson correlations across all candidates.
    candidates : Dict[str, List[np.ndarray]]
        Candidate sets mapping groups to arrays of column indices.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from group name to selected representative column indices.
    """
    representatives = {}
    for group, model_candidates in candidates.items():
        indices = []
        for columns in model_candidates:
            finite = columns[np.isfinite(correlations[columns])]
            if len(finite) == 0:
                raise ValueError(f"No defined candidate correlation in {group}.")
            # np.argmax selects the first maximum, resolving exact ties by row order
            indices.append(finite[np.argmax(correlations[finite])])
        representatives[group] = np.asarray(indices)
    return representatives


def select_models(
    correlations: np.ndarray,
    candidates: Dict[str, List[np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Select each model's best candidate, then its group's five largest signed r.

    Parameters
    ----------
    correlations : np.ndarray
        Vector of task-wide Pearson correlations.
    candidates : Dict[str, List[np.ndarray]]
        Candidate column indices grouped by model.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        (representatives, top5_selected), each mapped by group name.
    """
    representatives = select_model_candidates(correlations, candidates)
    selected = {}
    for group, indices in representatives.items():
        if len(indices) < 5:
            raise ValueError(f"Fewer than five models in {group}.")
        # Use stable sort so identical correlation ties preserve alphabetical model order
        top5_order = np.argsort(-correlations[indices], kind="stable")[:5]
        selected[group] = indices[top5_order]
    return representatives, selected


def top_generative_models(
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    tasks: Tuple[str, ...] = ("thatcher", "swap"),
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Select top 5 generative models per task based on task-wide Pearson correlation.

    Parameters
    ----------
    pairs : pd.DataFrame
        Human ratings table containing 'task', 'pair_id', and 'score'.
    scores : pd.DataFrame
        Generative model scores indexed by 'pair_id'.
    tasks : Tuple[str, ...]
        Tasks to evaluate.

    Returns
    -------
    Tuple[Dict[str, List[str]], pd.DataFrame]
        models_by_task : Dict mapping each task to list of top-5 model names.
        selected_table : DataFrame of selected models with rank and pooled_r.
    """
    from analysis.metrics import pearson

    models_by_task = {}
    records = []
    names = sorted(scores.columns)
    candidates = {"generative": [np.array([i]) for i in range(len(names))]}

    for task in tasks:
        local = pairs[pairs.task.eq(task)]
        x = scores.loc[local.pair_id, names].to_numpy()
        correlations = pearson(x, local.score.to_numpy())
        _, chosen = select_models(correlations, candidates)
        top_names = [names[i] for i in chosen["generative"]]
        models_by_task[task] = top_names
        for rank, name in enumerate(top_names, 1):
            records.append({
                "task": task,
                "rank": rank,
                "model": name,
                "pooled_r": float(correlations[names.index(name)]),
            })

    return models_by_task, pd.DataFrame(records)


def evaluate_and_select(
    items: pd.DataFrame,
    scores: np.ndarray,
    metadata: pd.DataFrame,
    task: str = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute task-wide correlations and select representative layers and top 5 models.

    Parameters
    ----------
    items : pd.DataFrame
        Evaluation items with 'task' and 'score'.
    scores : np.ndarray
        Predictor scores matrix aligned with items.
    metadata : pd.DataFrame
        Candidate predictor metadata.
    task : str, optional
        Filter by task name if specified, or None for all items.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]
        (correlations, representatives, selected)
    """
    from analysis.metrics import pearson

    if task is not None:
        mask = items.task.eq(task).to_numpy()
        items = items.loc[mask]
        scores = scores[mask]
    correlations = pearson(scores, items.score.to_numpy())
    representatives, selected = select_models(correlations, candidate_sets(metadata))
    return correlations, representatives, selected
