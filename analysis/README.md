# Statistical Analyses & Visualizations

This directory contains the post-hoc statistical modeling, bootstrap resampling, and figure/table generation modules for the paper. All analyses run entirely on CPU using the precomputed scalar datasets in `data/`.

For a high-level summary mapping paper targets to execution commands, see the main [README.md](../README.md#paper-results-to-code-mapping).

---

## Execution & Output Layout

All analysis scripts are executed as Python modules from the repository root:

```bash
python -m analysis.<entry> [--output-dir <path>]
```

- **Output Location:** By default, outputs are saved into `results/analysis/<entry>/`. An alternative path can be specified using `--output-dir`.
- **Pipeline Dependencies:** `human` (participant screening) and `bootstrap` (confidence intervals) must be run prior to running the individual figure/table plotting modules.

---

## Analysis Modules

| Module | Paper Location | Description | Key Output Files |
| --- | --- | --- | --- |
| `human` | Appendix B.4 | Leave-one-out KDE participant screening and clean aggregate ratings | `response_consistency_histogram`, screening decisions, response counts |
| `bootstrap` | Appendix E | Clustered bootstrap resampling across participants, scenes, and model families | Combined interval tables and run metadata (no resume checkpoints) |
| `sensitivity_alignment` | Figure 4; App. F–H | Cross-model sensitivity and alignment, condition breakdowns, and single-image naturalness | `cross_model_sensitivity_alignment`, `single_image_naturalness`, `condition_sensitivity_alignment*` |
| `timestep_plots` | Figure 5; App. N | Schedule dynamics, peak sensitivity positions, and condition-specific timestep profiles | `timestep_profiles_and_peaks`, `condition_timestep_peaks`, `condition_timestep_profiles_*` |
| `baseline_comparison` | Figure 3; App. I | Relative alignment comparison between generative models, vision encoders, and IQA metrics | `baseline_comparison`, `condition_baseline_comparison` |
| `factor_profiles` | Appendix J | Illumination factor disagreement analysis (shadows, reflections, light directions) | `illumination_factor_profiles_*`, `illumination_factor_disagreement_overview` |
| `model_size` | Appendix K | Task-pooled model-size comparison with encoders and condition-wise generator scatterplots | `model_size_sensitivity_alignment_with_encoders`, `model_size_conditions_thatcher`, `model_size_conditions_swap` |
| `generation_performance` | App. L, including Table S5 | Cross-model rank correlations with standard image generation benchmarks | `generation_performance_table`, `sensitivity_size_adjusted_table`, scatter plots |
| `spatial_maps` | Appendix M (and Figure 2) | Spatial error attribution maps across models and conditions | `spatial_maps_all25_*` |

---

## Methodology & Resampling Details

### Participant Screening (`human`)
- Evaluates individual ratings against leave-one-out image means.
- Computes task-specific Silverman KDE troughs to determine inclusion cutoffs.
- Retains only the first submission per participant and the first presentation of each stimulus.

### Clustered Bootstrap Resampling (`bootstrap`)
- **Hierarchical Clustering:** Resamples whole participants within stimulus-assignment groups and whole base scenes, preserving paired stimulus structures.
- **Top-5 & Layer Selection:** Model layers and task-wide Top-5 generative models are reselected within each bootstrap draw and shared across task conditions.
- **Fixed-Generator Sensitivity:** Evaluated by resampling scene clusters while conditioning on fixed models.
- **Cross-Model Correlations:** Additionally resamples across the 11 generative model families.
- **Interval Estimation:** Generates 10,000 draws to construct pointwise 95% percentile confidence intervals (Appendix E).
- **Batching:** Computes 100 draws at a time and retains the summary arrays in memory. This submission implementation does not checkpoint or resume interrupted runs. `--iterations` can override the total draw count (default: 10,000); reduced runs are not the full paper reproduction.

### Parameter Scaling & Condition Profiles (`model_size`)
- **Task-Pooled and Condition Panels:** Evaluates parameter scaling across 25 generative models (16 image, 9 video) and 12 vision encoders for pooled tasks, as well as condition-specific panels for upright/inverted Thatcher faces and shadow, reflection, and light-direction changes in Illumination.
- **Metric Definitions:** Sensitivity is each condition's mean signed pair score divided by that model's task-wide sample standard deviation (`ddof=1`; 280 Thatcher or 288 Illumination pairs). Human alignment is the within-condition Pearson correlation with human unnaturalness ratings.
- **Parameter Scaling Correlations:** Computes Pearson correlations against $\log_{10}$ parameter counts, Spearman rank correlations (average ranks for ties), and linear regression fits for all 25 models, image models ($n=16$), and video models ($n=9$) separately.

Figure mapping and execution details are recorded in [README.md](../README.md#paper-results-to-code-mapping).
