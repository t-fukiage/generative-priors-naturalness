"""Illumination factor-level profiles and model–human MSE (four appendix figures)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import configured_path, load_human_pairs, load_illumination_factors, load_predictor_candidates
from analysis.factor_plots import plot
from analysis.uncertainty import human_profile_intervals
from analysis.selection import GROUPS, evaluate_and_select


def profile_scores(scores, metadata):
    """Convert FR-IQA scores to distances before task-SD scaling.

    The correlation input stores PSNR/SSIM/VIF as negated quality. Recover
    that quality to obtain unit-range MSE, 1-SSIM and 1-VIF, respectively.
    LPIPS/DISTS are already distances. Other groups retain their signed scores.
    """
    values = scores.copy()
    definitions = []
    for column, row in enumerate(metadata.itertuples()):
        definition = "released signed score"
        if row.group == "fr_iqa":
            if row.model == "psnr":
                psnr = -values[:, column]
                values[:, column] = 10.0 ** (-psnr / 10.0)
                definition = "10^(-PSNR/10), unit-range MSE"
            elif row.model in ("ssim", "vif"):
                similarity = -values[:, column]
                values[:, column] = 1.0 - similarity
                definition = f"1-{row.model.upper()}"
            elif row.model in ("lpips_vgg", "dists"):
                definition = "native pair distance"
            else:
                raise ValueError(f"Unknown FR-IQA profile transform: {row.model}")
            if np.any(values[:, column] < -1e-12):
                raise ValueError(f"Negative FR-IQA distance: {row.model}")
            values[:, column] = np.maximum(values[:, column], 0.0)
        definitions.append(definition)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite profile scores.")
    return values, definitions


def summarize_mse(table, key):
    """Summarize individual-model errors; never calculate error from a median profile."""
    return table.groupby([key, "group"], sort=False).mse.agg(
        n_models="size", median="median", minimum="min", maximum="max").reset_index()


def analyze(items, scores, metadata, levels, membership):
    # 1. Select representative encoder layers and top-5 models per group
    mask = items.task.eq("swap").to_numpy()
    local, x = items.loc[mask].reset_index(drop=True), scores[mask]
    human = local.score.to_numpy()

    correlations, representatives, selected = evaluate_and_select(local, x, metadata)
    layers = metadata.iloc[representatives["encoder"]].copy()
    layers["pooled_r"] = correlations[representatives["encoder"]]

    columns = np.concatenate([selected[group] for group in GROUPS])
    panel = metadata.iloc[columns].reset_index(drop=True)
    panel["rank"] = np.tile(np.arange(1, 6), len(GROUPS))
    panel["pooled_r"] = correlations[columns]

    # 2. Transform FR-IQA metrics and standardize by task-wide sample SD
    transformed, definitions = profile_scores(x[:, columns], panel)
    matrix = np.column_stack([human, transformed])
    sds = matrix.std(axis=0, ddof=1)
    if not np.isfinite(sds).all() or np.any(sds <= 0):
        raise ValueError("Each profile requires a finite, positive task-wide sample SD.")
    standardized = matrix / sds

    human_source = pd.DataFrame([{
        "group": "human", "model": "human", "candidate": "score",
        "readout": "aggregate", "rank": 0,
    }])
    sources = pd.concat([human_source, panel.drop(columns="pooled_r")], ignore_index=True)
    scales = sources.assign(
        task_mean=matrix.mean(axis=0),
        task_sample_sd=sds,
        score_definition=["original minus modified mean rating", *definitions],
    )

    # 3. Compute factor-level mean profiles and human-model MSE
    profiles, errors = [], []
    factor_counts = levels.drop_duplicates("factor_id").groupby("operation").size()
    positions = pd.Series(np.arange(len(local)), index=local.pair_id)
    source_records = sources.to_dict("records")
    model_records = panel.to_dict("records")

    for factor, factor_levels in levels.groupby("factor_id", sort=False):
        means = []
        for level in factor_levels.itertuples(index=False):
            ids = membership.loc[
                membership.factor_id.eq(factor) & membership.level_order.eq(level.level_order),
                "pair_id",
            ]
            mean = standardized[positions.loc[ids].to_numpy()].mean(axis=0)
            means.append(mean)
            for source, value in zip(source_records, mean):
                profiles.append({**level._asdict(), **source, "score": value})

        means = np.asarray(means)
        # Column 0 is human reference; columns 1: are the 20 selected models
        human_profile = means[:, :1]
        model_profiles = means[:, 1:]
        mse = np.mean((model_profiles - human_profile) ** 2, axis=0)

        factor_info = factor_levels.iloc[0]
        operation = factor_info.operation
        weight = 1.0 / (3.0 * factor_counts[operation])  # 3 operations weighted equally
        for source, value in zip(model_records, mse):
            errors.append({
                "factor_id": factor, "operation": operation, "factor_weight": weight,
                "n_levels": len(means), **source, "mse": value,
            })

    profiles = pd.DataFrame(profiles)
    errors = pd.DataFrame(errors)

    # 4. Aggregate model errors by operation and overall scope
    identity = ["group", "model", "candidate", "readout", "rank"]
    operations = errors.groupby(["operation", *identity], sort=False).mse.mean().reset_index()
    # Equal factor weight within each operation, then equal weight for the three operations
    overall = operations.groupby(identity, sort=False).mse.mean().reset_index().assign(scope="overall")
    scopes = pd.concat([overall, operations.rename(columns={"operation": "scope"})], ignore_index=True)

    profile_summary = (
        profiles[profiles.group.ne("human")]
        .groupby(["factor_id", "level_order", "group"], sort=False)
        .score.median()
        .reset_index(name="median")
    )

    return {
        "selected_layers": layers,
        "selected_models": panel,
        "standardization": scales,
        "factor_level_profiles": profiles,
        "profile_medians": profile_summary,
        "mse_by_model_factor": errors,
        "mse_by_model_scope": scopes,
        "factor_mse_summary": summarize_mse(errors, "factor_id"),
        "scope_mse_summary": summarize_mse(scopes, "scope"),
    }


def save_results(result, output_dir, data_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory containing data (default: settings.json)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/analysis/factor_profiles)")
    args = parser.parse_args()

    data_root = configured_path("data_root", args.data_root)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path("output_root") / "analysis/factor_profiles"
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_human_pairs(data_root=data_root)
    scores, metadata = load_predictor_candidates(items, data_root=data_root)
    levels, membership = load_illumination_factors(items, data_root=data_root)
    result = analyze(items, scores, metadata, levels, membership)
    intervals = human_profile_intervals(data_root=data_root)
    result["human_intervals"] = intervals[intervals.task.eq("swap")].copy()
    save_results(result, output_dir, data_root)

    plot(result, levels, output_dir)
    print(f"Saved four appendix figures and numerical tables to {output_dir}")


if __name__ == "__main__":
    main()
