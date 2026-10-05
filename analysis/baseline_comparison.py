"""Figures 3 and S14: encoder and IQA baseline comparisons."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import configured_path, load_human_pairs, load_human_split_half, load_predictor_candidates
from analysis.baseline_plots import plot
from analysis.metrics import bidirectional_partial, human_correlations
from analysis.uncertainty import apply_intervals
from analysis.selection import GROUPS, evaluate_and_select


CONDITIONS = {"thatcher": ("str", "inv"), "swap": ("shadow_swap", "reflection_swap", "lightdir_swap")}
PARTIAL_STATISTICS = ("g_partial", "x_partial", "difference")


def compare(human, scores, groups):
    """Compute marginal and bidirectional partial r for the supplied columns."""
    marginal, correlations = human_correlations(human, scores)
    n_g = len(groups["generative"])
    g_partial, x_partial = bidirectional_partial(marginal[:n_g], marginal[n_g:], correlations[:n_g, n_g:])
    return marginal, g_partial, x_partial


def summarize(marginal, g_partial, x_partial, groups, selected):
    """Four marginal medians, then three statistics for each comparator group.

    Each partial statistic uses all 25 selected model pairs. The difference is
    the median of within-pair differences, not the difference of two medians.
    An undefined pair makes the corresponding median undefined.
    """
    values = [np.median(marginal[selected[group]]) for group in GROUPS]
    n_g = len(groups["generative"])
    for group in GROUPS[1:]:
        positions = np.ix_(selected["generative"], selected[group] - n_g)
        g, x = g_partial[positions], x_partial[positions]
        values.extend([np.median(g), np.median(x), np.median(g - x)])
    return np.asarray(values)


def column_positions(columns):
    """Convert group-wise global columns to positions in their concatenation."""
    sizes = [len(columns[group]) for group in GROUPS]
    boundaries = np.cumsum([0, *sizes])
    return {group: np.arange(boundaries[i], boundaries[i + 1]) for i, group in enumerate(GROUPS)}


def _record_model_selection(tables, task, metadata, pooled_r, representatives, selected, total_columns):
    """Record selected encoder layers, top-5 models per group, and placeholder selection frequencies."""
    for column in representatives["encoder"]:
        tables["selected_layers"].append({
            "task": task,
            **metadata.iloc[column].to_dict(),
            "pooled_r": pooled_r[column],
        })

    for group in GROUPS:
        for rank, column in enumerate(selected[group], start=1):
            tables["selected_models"].append({
                "task": task,
                **metadata.iloc[column].to_dict(),
                "rank": rank,
                "pooled_r": pooled_r[column],
            })

    layer_counts = top5_counts = np.zeros(total_columns, dtype=int)
    for column, record in metadata.iterrows():
        tables["selection_frequency"].append({
            "task": task,
            **record.to_dict(),
            "draws": 0,
            "best_encoder_layer_count": int(layer_counts[column]),
            "top5_count": int(top5_counts[column]),
        })


def _filter_human_references(split_half):
    """Extract and validate the required pooled and condition split-half rows."""
    expected = {(task, scope) for task, conditions in CONDITIONS.items() for scope in ("pooled", *conditions)}
    rows = [tuple(row) in expected for row in split_half[["task", "scope"]].to_numpy()]
    references = split_half.loc[rows].copy()
    found = set(references[["task", "scope"]].itertuples(index=False, name=None))
    if found != expected or len(references) != len(expected):
        raise ValueError("Incomplete pooled/condition split-half references.")
    return references


def analyze(items, scores, metadata, split_half, data_root=None):
    tables = {name: [] for name in (
        "model_alignment", "partial_pairs", "marginal_summary", "partial_summary",
        "selected_layers", "selected_models", "selection_frequency",
    )}

    for task, conditions in CONDITIONS.items():
        task_mask = items.task.eq(task).to_numpy()
        local = items.loc[task_mask]
        human, x = local.score.to_numpy(), scores[task_mask]

        # 1. Representative layer selection and top-5 model evaluation
        pooled_r, representatives, selected = evaluate_and_select(local, x, metadata)
        _record_model_selection(
            tables, task, metadata, pooled_r, representatives, selected,
            total_columns=x.shape[1],
        )

        columns = np.concatenate([representatives[group] for group in GROUPS])
        groups = column_positions(representatives)
        positions = {column: i for i, column in enumerate(columns)}
        selected_positions = {group: np.array([positions[column] for column in selected[group]]) for group in GROUPS}

        # 2. Marginal and bidirectional partial comparisons across scopes (pooled + conditions)
        scopes = ("pooled", *conditions)
        for scope in scopes:
            mask = np.ones(len(local), dtype=bool) if scope == "pooled" else local.condition.eq(scope).to_numpy()
            context = {
                "task": task, "scope": scope,
                "n_pairs": int(mask.sum()),
                "n_base_scenes": local.loc[mask, "base_scene_group"].nunique(),
            }

            marginal, g_partial, x_partial = compare(human[mask], x[mask][:, columns], groups)
            summary = summarize(marginal, g_partial, x_partial, groups, selected_positions)

            # (a) Model alignment and marginal group summaries
            for group_index, group in enumerate(GROUPS):
                for position in groups[group]:
                    column = columns[position]
                    ranks = np.flatnonzero(selected[group] == column)
                    tables["model_alignment"].append({
                        **context,
                        **metadata.iloc[column].to_dict(),
                        "r": marginal[position],
                        "selected": len(ranks) > 0,
                        "task_top5_rank": int(ranks[0] + 1) if len(ranks) else None,
                    })

                tables["marginal_summary"].append({
                    **context,
                    "group": group,
                    "median_r": summary[group_index],
                    "candidate_models": len(groups[group]),
                    "selected_models": 5,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "valid_draws": 0,
                })

            # (b) Pairwise partial correlations and comparator group summaries
            n_g = len(groups["generative"])
            for group_index, group in enumerate(GROUPS[1:]):
                for g in groups["generative"]:
                    for x_position in groups[group]:
                        comparator = x_position - n_g
                        g_value = g_partial[g, comparator]
                        x_value = x_partial[g, comparator]
                        tables["partial_pairs"].append({
                            **context,
                            "comparator_group": group,
                            "g_model": metadata.iloc[columns[g]].model,
                            "x_model": metadata.iloc[columns[x_position]].model,
                            "x_candidate": metadata.iloc[columns[x_position]].candidate,
                            "g_partial": g_value,
                            "x_partial": x_value,
                            "difference": g_value - x_value,
                            "selected": bool(g in selected_positions["generative"] and x_position in selected_positions[group]),
                            "best_pair": bool(g == selected_positions["generative"][0] and x_position == selected_positions[group][0]),
                        })

                # summary layout: [4 marginal medians] + [3 partial statistics * group_index]
                stat_offset = len(GROUPS) + len(PARTIAL_STATISTICS) * group_index
                record = {**context, "comparator_group": group, "selected_pairs": 25}
                for offset, statistic in enumerate(PARTIAL_STATISTICS):
                    record[statistic] = summary[stat_offset + offset]
                    record[f"{statistic}_ci_low"] = np.nan
                    record[f"{statistic}_ci_high"] = np.nan
                    record[f"{statistic}_valid_draws"] = 0
                tables["partial_summary"].append(record)

    result = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    result["human_references"] = _filter_human_references(split_half)
    return apply_intervals(result, "baseline_comparison", data_root)


def save_results(result, output_dir, data_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing data (default: settings.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/analysis/baseline_comparison)")
    args = parser.parse_args()

    data_root = configured_path("data_root", args.data_root)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path("output_root") / "analysis/baseline_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_human_pairs(data_root=data_root)
    scores, metadata = load_predictor_candidates(items, data_root=data_root)
    result = analyze(items, scores, metadata, load_human_split_half(data_root=data_root), data_root=data_root)
    save_results(result, output_dir, data_root)
    plot(result, output_dir)
    print(f"Saved Figures 5 and S14 and numerical tables to {output_dir}")


if __name__ == "__main__":
    main()
