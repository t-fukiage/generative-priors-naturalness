"""Recompute shared paper intervals from individual ratings and model scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import configured_path, load_human_pairs, load_predictor_candidates, load_illumination_factors
from analysis.human import response_matrix
from analysis.resampling import participant_weights, split_weights, weighted_means, weighted_correlation
from analysis.selection import candidate_sets

ITERATIONS, BATCH = 10000, 100
TASKS = ('thatcher', 'swap')
BASELINE_METRICS = ('median_g_partial_given_x', 'median_x_partial_given_g', 'median_paired_difference',
                    'median_g_marginal_human_r', 'median_x_marginal_human_r')

PARTICIPANT_SEED = 20260920
FAMILY_SEED = 20260921
SCENE_SEEDS = (4444, 5453)
SPLIT_SEED = 0


def input_contract(data_root, iterations):
    return {
        'iterations': iterations,
        'batch_size': BATCH,
        'participant_seed': PARTICIPANT_SEED,
        'scene_seeds': list(SCENE_SEEDS),
        'split_seed': SPLIT_SEED,
        'family_seed': FAMILY_SEED,
    }



def moments(x, y, weights):
    """Frequency-weighted Pearson correlation, sums, and variances.

    Parameters
    ----------
    x : np.ndarray
        Predictor matrix of shape [N_pairs, N_candidates].
    y : np.ndarray
        Human rating mean vector/matrix of shape [B, N_pairs].
    weights : np.ndarray
        Scene resampling frequency weights of shape [B, N_pairs].

    Returns
    -------
    tuple
        (r, sx, vx, n)
        r  : Pearson correlation matrix of shape [B, N_candidates], clipped to [-1, 1]
        sx : Weighted sum of x along pairs [B, N_candidates]
        vx : Weighted sum of squared deviations for x [B, N_candidates]
        n  : Total resampled pair count per draw [B, 1]
    """
    n = weights.sum(axis=1, keepdims=True)
    sx = weights @ x
    sy = np.sum(weights * y, axis=1, keepdims=True)
    vx = weights @ (x * x) - sx * sx / n
    vy = np.sum(weights * y * y, axis=1, keepdims=True) - sy * sy / n
    if np.any(vx <= 0) or np.any(vy <= 0):
        raise ValueError("A bootstrap statistic has zero variance.")
    covariance = (weights * y) @ x - sx * sy / n
    denominator = np.sqrt(np.maximum(vx * vy, 1e-15))
    r = covariance / denominator
    return np.clip(r, -1.0, 1.0), sx, vx, n


def select(correlations, candidates):
    """Select best candidate layer per model and group's Top 5 models.

    Parameters
    ----------
    correlations : np.ndarray
        Task-wide correlation matrix of shape [B, N_candidates].
    candidates : dict
        Mapping from group to list of column index arrays per model.

    Returns
    -------
    tuple
        (representatives, selected)
        representatives : dict of best layer indices per model [B, N_models]
        selected        : dict of Top 5 model column indices [B, 5]
    """
    representatives, selected = {}, {}
    for group, columns in candidates.items():
        # Choose each model's best candidate layer with maximum task-wide correlation
        representatives[group] = np.column_stack([
            c[np.argmax(correlations[:, c], axis=1)] for c in columns
        ])
        # Select top 5 models within each group using stable sort
        ranks = np.argsort(
            -np.take_along_axis(correlations, representatives[group], axis=1),
            axis=1,
            kind="stable",
        )[:, :5]
        selected[group] = np.take_along_axis(representatives[group], ranks, axis=1)
    return representatives, selected


def pair_partials(x, weights, stats, g, e):
    """Compute bidirectional partial correlations r(h, G | X) and r(h, X | G).

    Evaluates partial correlation between each selected generative model (g)
    and baseline comparator (e) after partialling out the other representation.

    Tensor indices in einsum:
      b : bootstrap draw batch index (size B)
      n : pair index within condition/scope
      i : selected generative model index (size 5)
      j : selected comparator baseline model index (size 5)
    """
    r, sums, variance, n = stats

    def take(a, indices):
        return np.take_along_axis(a, indices, axis=1)

    # 1. Weighted cross-model covariance between G and Baseline E:
    products = np.einsum(
        "bni,bnj,bn->bij",
        x[:, g].transpose(1, 0, 2),
        x[:, e].transpose(1, 0, 2),
        weights,
        optimize=True,
    )
    mean_correction = take(sums, g)[:, :, None] * take(sums, e)[:, None, :] / n[:, :, None]
    cov_ge = products - mean_correction

    # 2. Pairwise correlation between G and Baseline E: r(G, X)
    var_g = take(variance, g)[:, :, None]
    var_e = take(variance, e)[:, None, :]
    r_gx = np.clip(cov_ge / np.sqrt(np.maximum(var_g * var_e, 1e-15)), -1.0, 1.0)

    # 3. Marginal human correlations: r(H, G) and r(H, X)
    r_hg = take(r, g)[:, :, None]
    r_he = take(r, e)[:, None, :]

    # 4. Partial correlations (Appendix E.3, Eq. 16):
    #    gp = r(Human, Generative | Baseline)
    #    ep = r(Human, Baseline | Generative)
    denom_g = np.sqrt(np.maximum((1.0 - r_he * r_he) * (1.0 - r_gx * r_gx), 1e-15))
    denom_e = np.sqrt(np.maximum((1.0 - r_hg * r_hg) * (1.0 - r_gx * r_gx), 1e-15))
    gp = (r_hg - r_he * r_gx) / denom_g
    ep = (r_he - r_hg * r_gx) / denom_e

    batch_len = len(weights)
    return np.clip(gp, -1.0, 1.0).reshape(batch_len, -1), np.clip(ep, -1.0, 1.0).reshape(batch_len, -1)


def evaluate(x, human, weights, masks, candidates, metadata, family_weights):
    """Task-wide layer/Top5 selection is shared by all conditions in a draw."""
    pooled = moments(x, human, weights)
    representatives, selected = select(pooled[0], candidates)
    order = np.concatenate(list(representatives.values()), axis=1)
    names = [metadata.iloc[c[0]].model for group in candidates.values() for c in group]
    generators = representatives['generative']
    task_sd = np.sqrt(pooled[2] / (pooled[3] - 1))
    total = weights.sum(axis=1)
    human_mean = (weights * human).sum(axis=1)
    human_sq_mean = (weights * human * human).sum(axis=1)
    human_sd = np.sqrt((human_sq_mean - human_mean ** 2 / total) / (total - 1))

    columns, values = [], []

    def add(scope, kind, target, metric, value):
        columns.append((scope, kind, target, metric))
        values.append(value)

    for scope, mask in masks.items():
        xs = x[mask]
        ys = human[:, mask]
        ws = weights[:, mask]
        stats = pooled if scope == 'pooled' else moments(xs, ys, ws)
        r, sums, variance, n = stats
        correlations = np.take_along_axis(r, order, axis=1)

        # 1. Model human alignment r
        for j, model in enumerate(names):
            add(scope, 'model', model, 'human_r', correlations[:, j])

        # 2. Directional sensitivity (mean score / task SD)
        sensitivity = (sums / n) / task_sd
        gs = np.take_along_axis(sensitivity, generators, axis=1)
        gr = np.take_along_axis(r, generators, axis=1)
        for j, c in enumerate(candidates['generative']):
            add(scope, 'model', metadata.iloc[c[0]].model, 'sensitivity', gs[:, j])

        if scope == 'pooled':
            es = np.take_along_axis(sensitivity, representatives['encoder'], axis=1)
            for j, c in enumerate(candidates['encoder']):
                add(scope, 'model', metadata.iloc[c[0]].model, 'sensitivity', es[:, j])

        # 3. Human sensitivity reference
        human_sens = (ws * ys).sum(axis=1) / ws.sum(axis=1) / human_sd
        add(scope, 'human', 'human', 'sensitivity', human_sens)

        # 4. Cross-model sensitivity-alignment correlation
        add(scope, 'cross_model', 'generative_25', 'sensitivity_alignment_r',
            weighted_correlation(gs, gr, family_weights))

        # 5. Baseline comparator partial correlations (Top 5 vs Top 5)
        for group in ('encoder', 'fr_iqa', 'nr_iqa'):
            gp, ep = pair_partials(xs, ws, stats, selected['generative'], selected[group])
            stats_values = [
                np.median(gp, axis=1),
                np.median(ep, axis=1),
                np.median(gp - ep, axis=1),
                np.median(np.take_along_axis(r, selected['generative'], axis=1), axis=1),
                np.median(np.take_along_axis(r, selected[group], axis=1), axis=1),
            ]
            for metric, value in zip(BASELINE_METRICS, stats_values):
                add(scope, 'baseline', group, metric, value)

    array = np.column_stack(values)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite bootstrap statistic; no draws are dropped.')
    return array, columns, representatives, selected


def summarize(task, columns, points, draws):
    lo, hi = np.quantile(draws, [.025, .975], axis=0, method='linear')
    return [dict(task=task, condition=s, kind=k, target=t, metric=m, estimate=points[i],
                 ci_low=lo[i], ci_high=hi[i], finite_draws=len(draws))
            for i, (s, k, t, m) in enumerate(columns)]


def factor_intervals(pairs, iterations, seed=PARTICIPANT_SEED):
    """Compute bootstrap confidence intervals for human naturalness across factor levels."""
    levels, members = load_illumination_factors(pairs)
    seeds = np.random.SeedSequence(seed).spawn(2)
    rows = []

    for task, task_seed in zip(TASKS, seeds):
        task_pairs = pairs.loc[pairs.task.eq(task)].sort_values(['condition', 'scene_id'])
        effects, strata = response_matrix(task_pairs)

        # 1. Define factor level specifications and corresponding pair masks
        if task == 'thatcher':
            level_specs = [
                {'factor_id': 'thatcher', 'level_order': 0, 'level_label': 'Upright', 'condition': 'str'},
                {'factor_id': 'thatcher', 'level_order': 1, 'level_label': 'Inverted', 'condition': 'inv'},
            ]
            level_masks = [task_pairs.condition.eq(spec['condition']).to_numpy() for spec in level_specs]
        else:
            level_specs = levels.to_dict('records')
            level_masks = []
            for spec in level_specs:
                matched_pairs = members.loc[
                    members.factor_id.eq(spec['factor_id']) & members.level_order.eq(spec['level_order']),
                    'pair_id',
                ]
                level_masks.append(task_pairs.pair_id.isin(matched_pairs).to_numpy())

        # 2. Construct indicator weight matrix to compute level means via matrix multiplication
        # Shape: [N_pairs, N_levels], each column sums to 1.0
        level_weights = np.column_stack([m / m.sum() for m in level_masks])

        # Point estimates: standardized human mean per level (mean / task SD)
        observed_scores = task_pairs.score.to_numpy()
        point_estimates = (observed_scores @ level_weights) / observed_scores.std(ddof=1)

        # 3. Bootstrap resampling over participants
        rng = np.random.default_rng(task_seed)
        bootstrap_draws = np.empty((iterations, len(level_specs)))

        for start in range(0, iterations, BATCH):
            size = min(BATCH, iterations - start)
            pw = participant_weights(strata, size, rng)
            resampled_human = weighted_means(effects, pw)

            # Standardized level means for each bootstrap draw
            level_means = resampled_human @ level_weights
            task_sd = resampled_human.std(axis=1, ddof=1)[:, None]
            bootstrap_draws[start:start + size] = level_means / task_sd

        # 4. Summarize 95% percentiles (2.5% and 97.5%)
        ci_low, ci_high = np.quantile(bootstrap_draws, [.025, .975], axis=0)

        for j, spec in enumerate(level_specs):
            rows.append({
                'task': task,
                'factor_id': spec['factor_id'],
                'level_order': spec['level_order'],
                'level_label': spec['level_label'],
                'human_mean': point_estimates[j],
                'ci_low': ci_low[j],
                'ci_high': ci_high[j],
                'n_pairs': int(level_masks[j].sum()),
            })

    return pd.DataFrame(rows)


def split_correlations(effects, split_a, split_b, masks, scene_weights=None):
    if min((split_a @ np.isfinite(effects)).min(), (split_b @ np.isfinite(effects)).min()) < 2:
        raise ValueError('Split half has fewer than two raters per image.')
    half_a = weighted_means(effects, split_a)
    half_b = weighted_means(effects, split_b)
    batch_size = len(half_a)

    mask_list = list(masks.values()) if isinstance(masks, dict) else (masks if isinstance(masks, (list, tuple)) else [masks])

    raw = np.column_stack([
        weighted_correlation(half_a[:, m], half_b[:, m], np.ones((batch_size, m.sum())))
        for m in mask_list
    ])
    if scene_weights is None:
        return raw

    weighted = np.column_stack([
        weighted_correlation(half_a[:, m], half_b[:, m], scene_weights[:, m])
        for m in mask_list
    ])
    return raw, weighted


def screening_split_half(pairs, iterations, seed=SPLIT_SEED):
    swap_pairs = pairs.loc[pairs.task.eq('swap')].sort_values('scene_id')
    scopes = ['lightdir_swap', 'reflection_swap', 'shadow_swap', 'pooled']

    rows = []
    for scope in scopes:
        sub = swap_pairs if scope == 'pooled' else swap_pairs.loc[swap_pairs.condition.eq(scope)]
        effects, strata = response_matrix(sub, 'excluded')
        rng = np.random.default_rng(seed)
        mask = np.ones(effects.shape[1], dtype=bool)

        draws = []
        for start in range(0, iterations, BATCH):
            batch_size = min(BATCH, iterations - start)
            split_a, split_b = split_weights(strata, batch_size, rng)
            draws.extend(split_correlations(effects, split_a, split_b, mask).ravel())

        rows.append({
            'task': 'swap',
            'scope': scope,
            'cohort': 'excluded',
            'n_participants': len(effects),
            'n_splits': iterations,
            'r_mean': float(np.mean(draws)),
        })
    return pd.DataFrame(rows)


def run(data_root, output, iterations=ITERATIONS):
    """Run crossed bootstrap resampling for all tasks and write shared paper tables."""
    output.mkdir(parents=True, exist_ok=True)
    contract = input_contract(data_root, iterations)

    # ---------------------------------------------------------
    # 1. Load data, model candidates, and resampling structures
    # ---------------------------------------------------------
    pairs = load_human_pairs(data_root)
    scores, metadata = load_predictor_candidates(pairs, data_root)
    candidates = candidate_sets(metadata)
    n_generators = len(candidates['generative'])

    # Model family clustering weights for cross-model correlation
    extra = data_root / 'resampling_inputs'
    families = pd.read_csv(extra / 'model_families.csv').set_index('model')
    gen_names = [metadata.iloc[c[0]].model for c in candidates['generative']]
    labels, codes = np.unique(families.loc[gen_names, 'family'], return_inverse=True)
    family_rng = np.random.default_rng(contract['family_seed'])
    family_weights = family_rng.multinomial(
        len(labels), np.full(len(labels), 1 / len(labels)), size=iterations
    )[:, codes]

    # Text conditioning control scores (empty vs conditional prompt)
    text_path = extra / 'text_conditioning.csv'
    if not text_path.exists():
        text_path = extra / 'text_conditioning.csv.gz'
    text = pd.read_csv(text_path, float_precision='round_trip')
    text_models = sorted(text.model.unique())
    text_scores = text.pivot(index='pair_id', columns=['mode', 'model'], values='score')
    text_columns = [(mode, model) for mode in ('empty', 'conditional') for model in text_models]

    intervals = []
    split_half_references = []
    selection_frequencies = []
    text_controls = []

    # ---------------------------------------------------------
    # 2. Process each task (Thatcher and Swap)
    # ---------------------------------------------------------
    for ti, task in enumerate(TASKS):
        task_pairs = pairs.loc[pairs.task.eq(task)].sort_values(['scene_id', 'condition'])

        # Standardize predictor scores (sample standard deviation ddof=1)
        model_scores = scores[task_pairs.index]
        model_scores = model_scores / model_scores.std(axis=0, ddof=1)
        observed_human_scores = task_pairs.score.to_numpy()

        # Human participant response matrix and stratification
        effects, strata = response_matrix(task_pairs, data_root=data_root)

        # Align and standardize text conditioning predictor scores
        text_scores_matrix = text_scores.reindex(task_pairs.pair_id).loc[:, text_columns].to_numpy()
        text_scores_matrix = text_scores_matrix / text_scores_matrix.std(axis=0, ddof=1)

        # Condition evaluation masks
        masks = {
            'pooled': np.ones(len(task_pairs), bool),
            **{cond: task_pairs.condition.eq(cond).to_numpy() for cond in task_pairs.condition.unique()},
        }
        scene_groups, scene_indices = np.unique(task_pairs.base_scene_group, return_inverse=True)

        # Point estimates on full dataset (unweighted)
        point_estimates, stat_columns, _, _ = evaluate(
            model_scores,
            observed_human_scores[None],
            np.ones((1, len(observed_human_scores))),
            masks,
            candidates,
            metadata,
            np.ones((1, n_generators)),
        )
        text_point_r = moments(
            text_scores_matrix, observed_human_scores[None], np.ones((1, len(observed_human_scores)))
        )[0].reshape(2, len(text_models))

        # Independent random generators per task
        scene_rng = np.random.default_rng(contract['scene_seeds'][ti])
        participant_rng = np.random.default_rng(contract['participant_seed'] + ti * 1009)
        split_rng = np.random.default_rng(contract['split_seed'])

        # -----------------------------------------------------
        # 3. Batch bootstrap resampling loop (BATCH draws per step)
        # -----------------------------------------------------
        batch_cache = []
        for start in range(0, iterations, BATCH):
            size = min(BATCH, iterations - start)

            # Resample scene clusters and participants
            scene_weights = scene_rng.multinomial(
                len(scene_groups), np.full(len(scene_groups), 1 / len(scene_groups)), size=size
            )[:, scene_indices].astype(float)
            pw = participant_weights(strata, size, participant_rng)
            split_a, split_b = split_weights(strata, size, split_rng)

            # Compute resampled human mean scores and evaluate metrics
            resampled_human = weighted_means(effects, pw)
            batch_draws, batch_cols, batch_reps, batch_selected = evaluate(
                model_scores,
                resampled_human,
                scene_weights,
                masks,
                candidates,
                metadata,
                family_weights[start:start + size],
            )
            if batch_cols != stat_columns:
                raise ValueError('Bootstrap columns changed.')

            # Split-half reliability correlations (unweighted and scene-resampled)
            raw_split, weighted_split = split_correlations(
                effects, split_a, split_b, masks, scene_weights=scene_weights
            )

            # Text conditioning correlation difference (conditional minus empty)
            text_batch_r = moments(text_scores_matrix, resampled_human, scene_weights)[0].reshape(
                size, 2, len(text_models)
            )

            batch_cache.append({
                'draws': batch_draws,
                'raw_split': raw_split,
                'weighted_split': weighted_split,
                'text_diff': text_batch_r[:, 1] - text_batch_r[:, 0],
                'encoder_layers': np.bincount(batch_reps['encoder'].ravel(), minlength=model_scores.shape[1]),
                'top5_selections': np.bincount(
                    np.concatenate(list(batch_selected.values()), axis=1).ravel(),
                    minlength=model_scores.shape[1],
                ),
            })

            if start % 2000 == 0:
                print(f'{task}: {start+size}/{iterations}', flush=True)

        # -----------------------------------------------------
        # 4. Summarize bootstrap distributions and confidence intervals
        # -----------------------------------------------------
        all_draws = np.concatenate([batch['draws'] for batch in batch_cache])
        intervals.extend(summarize(task, stat_columns, point_estimates[0], all_draws))

        all_raw_split = np.concatenate([batch['raw_split'] for batch in batch_cache])
        all_weighted_split = np.concatenate([batch['weighted_split'] for batch in batch_cache])
        all_text_diff = np.concatenate([batch['text_diff'] for batch in batch_cache])

        # Human split-half reliability summaries
        for j, scope in enumerate(masks):
            lo, hi = np.quantile(all_weighted_split[:, j], [.025, .975])
            split_half_references.append(dict(
                task=task, scope=scope, r_mean=all_raw_split[:, j].mean(), ci_low=lo, ci_high=hi,
                n_splits=iterations, n_items=int(masks[scope].sum()), finite_draws=iterations,
            ))

        # Text conditioning control summaries
        for j, model in enumerate(text_models):
            lo, hi = np.quantile(all_text_diff[:, j], [.025, .975])
            empty_r = text_point_r[0, j]
            desc_r = text_point_r[1, j]
            text_controls.append(dict(
                task=task, model=model, empty_r=empty_r, description_r=desc_r,
                difference=desc_r - empty_r, ci_low=lo, ci_high=hi, finite_draws=iterations,
            ))

        # Layer and model selection frequencies across all draws
        layer_counts = sum(batch['encoder_layers'] for batch in batch_cache)
        top5_counts = sum(batch['top5_selections'] for batch in batch_cache)
        for j, row in metadata.iterrows():
            selection_frequencies.append(dict(
                task=task, target=row.model, candidate=row.candidate, readout=row.readout,
                draws=iterations, best_encoder_layer_count=int(layer_counts[j]), top5_count=int(top5_counts[j]),
            ))

    # ---------------------------------------------------------
    # 5. Build output tables and write CSV/JSON artifacts
    # ---------------------------------------------------------
    intervals_df = pd.DataFrame(intervals)
    family_df = intervals_df.loc[intervals_df.kind.eq('cross_model')].copy()
    family_df['n_models'] = n_generators
    family_df['n_families'] = len(labels)

    tables = {
        'intervals': intervals_df.loc[~intervals_df.kind.eq('cross_model')],
        'family_correlations': family_df,
        'human_split_half': pd.DataFrame(split_half_references),
        'selection_frequency': pd.DataFrame(selection_frequencies),
        'text_conditioning': pd.DataFrame(text_controls),
        'human_participant_intervals': factor_intervals(pairs, iterations, seed=contract['participant_seed']),
        'screening_split_half': screening_split_half(pairs, iterations, seed=contract['split_seed']),
    }

    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)

    manifest = dict(
        contract,
        files=[f'{name}.csv' for name in tables],
    )
    (output / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main() -> None:
    """CLI entry point for clustered bootstrap interval computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=None, help='Root directory containing data (default: settings.json)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory (default: results/analysis/bootstrap)')
    parser.add_argument('--iterations', type=int, default=ITERATIONS, help=f'Number of bootstrap iterations (default: {ITERATIONS})')
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error('--iterations must be at least 2')

    data_root = configured_path('data_root', args.data_root)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else configured_path('output_root') / 'analysis/bootstrap'
    output_dir.mkdir(parents=True, exist_ok=True)

    run(data_root, output_dir, args.iterations)
    print(f"Saved {args.iterations}-draw bootstrap intervals and batch records to {output_dir}")


if __name__ == '__main__':
    main()
