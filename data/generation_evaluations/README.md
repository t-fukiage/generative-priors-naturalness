# Generation Evaluations

This directory contains external benchmark evaluation scores and model metadata used to examine the relationship between generative prior alignment and standard synthesis quality metrics (Table 1, Table S5, and Figures S22–S24; Appendix L).

---

## Overview

- **Benchmarks Evaluated:** Includes Image Elo (Artificial Analysis text-to-image arena) and VBench Total Quality % (video generation benchmark).
- **Matched Models:** 18 models (11 image models and 7 video models) with published benchmark results matched to our generative model panel.
- **Excluded Models:** Generative models without public benchmark standings or non-matching baseline variants.
- **Usage:** Run `python -m analysis.generation_performance` to compute raw and size-adjusted rank correlations.

---

## File Inventory

| File | Rows / Size | Description |
| --- | --- | --- |
| `matched_models.csv` | 19 rows | 18 matched generative models with external benchmark scores, parameter counts, and benchmark sources |
| `excluded_models.csv` | 8 rows | 7 generative models without matching benchmark standings and exclusion reasons |

### Column Schema (`matched_models.csv`)

| Column | Type | Meaning |
| --- | --- | --- |
| `model` | string | Model identifier matching `data/models.json` |
| `modality` | string | `image` or `video` |
| `display_name` | string | Formatted display name used in tables and figures |
| `parameter_b` | float | Parameter count in billions |
| `benchmark_name` | string | Name of external benchmark (`image_elo` or `vbench_quality`) |
| `benchmark_score` | float | Reported benchmark score (Elo rating or percentage) |
| `source_url` | string | Citation URL for benchmark leaderboard snapshot |
| `snapshot_date` | string | Date when leaderboard score was accessed |

The benchmark values are cited from third-party public leaderboards as indicated by `source_url`. See [LICENSE_DATA.md](../../LICENSE_DATA.md).
