# Data schema

The dataset contains 280 Thatcher and 288 Illumination pairs.
`models.json` records model settings. Statistical definitions are in Appendix E.

## Pair and human tables

| File | Contents |
| --- | --- |
| `pairs.csv` | `pair_id,condition,original,modified`; image paths relative to `stimuli/` |
| `pair_metadata.csv` | `pair_id,task,scene_id,condition,base_scene_group` |
| `human_aggregates.csv` | Original/modified mean naturalness, response counts, SDs, counts for ratings 1–5, and `score` (original minus modified mean) |

`task` is `thatcher` or `swap` (Illumination). Conditions are `str` (upright),
`inv` (inverted), `shadow_swap`, `reflection_swap`, and `lightdir_swap`.
`base_scene_group` identifies the 140 face identities or 48 illumination
scenes used for resampling, retaining orientations, operations, and mirrors
(Appendix E). Pair order preserves model batch membership.
Individual ratings are in [human_responses](human_responses/README.md).
The analyses recompute screening and image means from those ratings;
`human_aggregates.csv` supplies the reference values for validation.

## Generative losses

`losses/<model>.npz` uses the following arrays (`allow_pickle=False`):

| Array | Shape / dtype | Meaning |
| --- | --- | --- |
| `pair_ids` | `[568]`, Unicode | Pair order |
| `roles` | `[2]`, Unicode | `original`, `modified` |
| `timestep_index` | `[T]`, integer | Evaluation-grid position |
| `loss_mean` | `[568,2,T]`, float64 | Mean over noise draws |
| `loss_by_draw` | `[568,2,100,20]`, float32 | Per-draw losses (image models) |
| `timesteps` | `[100]`, float32 | Native evaluation times (image models) |

Image models use `T=100`; video models use `T=50` and supply draw-averaged
losses. A pair score is the timestep mean for the modified image minus that
for the original. Evaluation-grid positions are model-specific (Appendix C).

```python
from data_io import load_losses, paired_scores
scores = paired_scores(load_losses("stable_diffusion_15"))
```

## Encoder, IQA, and reference tables

| File | Contents |
| --- | --- |
| `encoder_distances.csv.gz` | `pair_id,model,candidate,readout,score`; 326 layer/endpoint candidates across 12 encoders |
| `iqa_scores.csv.gz` | `pair_id,model,score`, plus `raw_pair_quality` (FR) or `original_quality,modified_quality` (NR) |
| `reference_pair_scores.csv.gz` | 568 × 47 pair scores; encoder entries use endpoints |
| `selected_layers.csv` | Task-specific best encoder candidates |
| `selected_top5.csv` | Task-specific model ranks |
| `reference_model_metrics.csv` | Sensitivity and human alignment for 25 generators and 12 selected encoders, by task |

Encoder candidates are one-based block numbers or `endpoint`. IQA scores
are signed so larger values indicate greater degradation. Readouts and
score conventions are defined in Appendix D; selection is defined in
Appendix E.3.
