"""Read shared bootstrap outputs and join them to figure tables."""

import json
import pandas as pd
from data_io import configured_path


def load_tables(data_root=None, bootstrap_root=None):
    root = configured_path('bootstrap_root', bootstrap_root)
    record = root / 'run.json'
    if not record.exists():
        raise FileNotFoundError('Run python -m analysis.bootstrap before generating figures.')
    manifest = json.loads(record.read_text())
    tables = {}
    for name in manifest.get('files', []):
        path = root / name
        if path.exists():
            tables[path.stem] = pd.read_csv(path, float_precision='round_trip')
    return tables


def provenance(data_root=None, bootstrap_root=None):
    root = configured_path('bootstrap_root', bootstrap_root)
    return json.loads((root / 'run.json').read_text())


def human_profile_intervals(data_root=None, bootstrap_root=None, *args, **kwargs):
    """Load precomputed human participant intervals for factor profiles."""
    return load_tables(data_root=data_root, bootstrap_root=bootstrap_root)['human_participant_intervals']


def apply_intervals(result, analysis, data_root=None, bootstrap_root=None):
    """Join recomputed uncertainty tables to figure-specific point estimates."""
    refs = load_tables(data_root=data_root, bootstrap_root=bootstrap_root)
    table = refs['intervals'].set_index(['task', 'condition', 'kind', 'target', 'metric'], verify_integrity=True)

    def bounds(task, scope, kind, target, metric):
        row = table.loc[(task, scope, kind, target, metric)]
        return dict(ci_low=row.ci_low, ci_high=row.ci_high, valid_draws=int(row.finite_draws))

    for name in ('model_metrics', 'condition_model_metrics', 'model_scores'):
        if name not in result:
            continue
        frame = result[name]
        for i, row in frame.iterrows():
            scope = row.get('condition', 'pooled')
            metrics = ('alignment',) if name == 'model_scores' else ('sensitivity', 'alignment')
            for metric in metrics:
                target_metric = 'human_r' if metric == 'alignment' else metric
                for key, value in bounds(row.task, scope, 'model', row.model, target_metric).items():
                    frame.loc[i, f'{metric}_{key}'] = value

    human = refs['human_split_half'].set_index(['task', 'scope'])
    for name in ('human_references', 'condition_human_references'):
        if name not in result:
            continue
        frame = result[name]
        for i, row in frame.iterrows():
            scope = row.get('scope', row.get('condition', 'pooled'))
            ref = human.loc[(row.task, scope)]
            if 'sensitivity' in frame:
                for key, value in bounds(row.task, scope, 'human', 'human', 'sensitivity').items():
                    frame.loc[i, f'sensitivity_{key}'] = value
                frame.loc[i, 'alignment'] = ref.r_mean
                frame.loc[i, 'alignment_ci_low'] = ref.ci_low
                frame.loc[i, 'alignment_ci_high'] = ref.ci_high
                frame.loc[i, 'n_splits'] = ref.n_splits
            else:
                for key in ('r_mean', 'ci_low', 'ci_high', 'n_splits', 'n_items'):
                    frame.loc[i, key] = ref[key]

    if analysis == 'baseline_comparison':
        for i, row in result['marginal_summary'].iterrows():
            group = 'encoder' if row.group == 'generative' else row.group
            metric = 'median_g_marginal_human_r' if row.group == 'generative' else 'median_x_marginal_human_r'
            for key, value in bounds(row.task, row.scope, 'baseline', group, metric).items():
                result['marginal_summary'].loc[i, key] = value
        mapping = dict(g_partial='median_g_partial_given_x', x_partial='median_x_partial_given_g', difference='median_paired_difference')
        for i, row in result['partial_summary'].iterrows():
            for stat, metric in mapping.items():
                for key, value in bounds(row.task, row.scope, 'baseline', row.comparator_group, metric).items():
                    result['partial_summary'].loc[i, f'{stat}_{key}'] = value

    frequency = refs['selection_frequency']
    frequency['candidate'] = frequency.candidate.astype(str)
    frequency = frequency.set_index(['task', 'target', 'candidate', 'readout'], verify_integrity=True)
    for name in ('selection_frequency', 'layer_selection_frequency'):
        if name not in result:
            continue
        frame = result[name]
        for i, row in frame.iterrows():
            ref = frequency.loc[(row.task, row.model, str(row.candidate), row.readout)]
            if name == 'selection_frequency':
                for key in ('best_encoder_layer_count', 'top5_count', 'draws'):
                    frame.loc[i, key] = ref[key]
            else:
                frame.loc[i, 'selected_count'] = ref.best_encoder_layer_count
                frame.loc[i, 'draws'] = ref.draws

    if analysis == 'sensitivity_alignment':
        result['family_correlations'] = refs['family_correlations'].copy()
    return result
