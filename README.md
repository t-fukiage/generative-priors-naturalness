# Supplementary Code: Do Generative Priors Align with Human Naturalness Perception?

This repository contains the reproduction code and scientific data for **"Do Generative Priors Align with Human Naturalness Perception?"**.

The complete CPU-analysis inputs are included: 25 model-loss NPZ files (~138 MB), anonymized participant ratings, vision encoder and IQA scores, factor metadata, and spatial maps. The 576 Illumination stimulus PNGs (~238 MB) are also included.

Thatcher face images and model checkpoint weights are excluded. A standalone [Thatcher generator](stimulus_generation/thatcher/README.md) is provided to recreate the Thatcher stimuli from official public sources.

See [Paper Results to Code Mapping](#paper-results-to-code-mapping) below for the mapping of paper figures and tables to analysis scripts.

---

## Overview & Two Modes of Verification

This package supports two levels of reproducibility:

1. **Mode 1: Reproduce Paper Analyses & Figures (CPU-only)**
   - **Environment:** Standard Python 3.10+ on any typical CPU (macOS / Linux / Windows).
   - **Inputs:** Bundled participant ratings, encoder/IQA scores, factor metadata, and model-loss files under `data/`.
   - **What it does:** Computes participant screening (Appendix B.4), executes 10,000-draw clustered bootstrap resampling (Appendix E), and renders main and appendix figures into `results/analysis/`.

2. **Mode 2: Recompute Raw Model Scores from Weights (GPU-required, Optional)**
   - **Environment:** Dedicated PyTorch/CUDA environments with GPU support.
   - **What it does:** Runs raw model forward passes (25 generative diffusion models, 12 vision encoders, and 10 IQA baselines) on the stimulus pairs (`data/stimuli/`).
   - See [Recomputing Model Scores](#recomputing-model-scores-mode-2-gpu-required) below and [MODEL_INDEX.md](MODEL_INDEX.md).

---

## Repository Structure

```
supplemental_code/
├── README.md               # Main entry point and reproduction guide (this file)
├── MODEL_INDEX.md          # 25 generative models + baselines (checkpoints, configs)
├── settings.json           # Configurable paths, outputs, and figure font settings
├── data/                   # Bundled metadata, encoder/IQA scores, ratings, and loss profiles
│   ├── README.md           # Dataset overview and file descriptions
│   ├── SCHEMA.md           # Column and array schema definitions
│   ├── pairs.csv           # 568 stimulus pair definitions (Thatcher & Illumination)
│   ├── human_responses/    # Anonymized individual participant ratings
│   ├── losses/             # 25 generative-model loss NPZ files (~138 MB)
│   ├── stimuli/            # Illumination PNGs (~238 MB); Thatcher is generated separately
│   └── spatial_maps/       # Spatial error maps and condition metadata
├── stimulus_generation/    # Thatcher stimulus generation code (GPLv3)
├── analysis/               # Statistical analysis, bootstrap resampling, and plotting scripts
├── scoring/                # Model-native score computation for diffusion models (GPU)
├── encoders/               # Feature distance computation for vision encoders
├── iqa/                    # Classical and deep IQA baseline metrics computation
├── configs/                # Per-model JSON configuration files
├── environments/           # Python environment specification files
└── results/                # Created at runtime for generated figures and tables
```

---

## Included Data and External Inputs

| Input | Location / Acquisition | Purpose |
| --- | --- | --- |
| 25 model-loss NPZ files | Included in `data/losses/` (~138 MB) | Complete CPU analyses |
| Ratings, encoder/IQA scores, factors, reference tables | Included in `data/` | Complete CPU analyses |
| 576 Illumination PNGs | Included in `data/stimuli/stimuli_shadow_reflection_lightdir/` (~238 MB) | GPU scoring |
| 560 Thatcher PNGs | Recreate with [Thatcher generator](stimulus_generation/thatcher/README.md) | GPU scoring |
| Generative / encoder / IQA weights | Acquire from official sources (see [MODEL_INDEX.md](MODEL_INDEX.md)) | Optional GPU scoring |

---

## Quickstart: Reproduce Paper Analyses (Mode 1)

Run the following commands sequentially from the package root using Python 3.10+:

### Step 1: Install Dependencies
```bash
python3.10 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r environments/analysis_requirements.txt
```

### Step 2: Human Participant Screening & Aggregation (Appendix B.4)
Apply leave-one-out Silverman KDE screening to raw participant ratings to generate clean aggregates:
```bash
python -m analysis.human
```

### Step 3: Bootstrap Resampling (Appendix E)
Compute 10,000-draw clustered bootstrap intervals (sampling participants and scenes, with cross-model family clustering):
```bash
OPENBLAS_NUM_THREADS=1 python -m analysis.bootstrap
```
> **Note:** `OPENBLAS_NUM_THREADS=1` prevents OpenBLAS thread oversubscription.

### Step 4: Render Figures and Tables
Run the plotting scripts to produce the figures and tables in the paper (see mapping below):
```bash
python -m analysis.overview_profiles
python -m analysis.sensitivity_alignment
python -m analysis.timestep_plots
python -m analysis.baseline_comparison
python -m analysis.factor_profiles
python -m analysis.model_size
python -m analysis.generation_performance
python -m analysis.spatial_maps
```

---

## Paper Results to Code Mapping

Outputs are saved under `results/analysis/<entry>/`.

| Paper Target | Description / Content | Command | Key Output Files |
|---|---|---|---|
| **Figure 1B** | Numerical profile panels | `python -m analysis.overview_profiles` | `factorial_plots_detail.pdf`, `.png` |
| **Figure 3** | Alignment and bidirectional partial correlations with IQA/encoders | `python -m analysis.baseline_comparison` | `baseline_comparison.pdf`, `.png` |
| **Figure 4** | Cross-model sensitivity and alignment scatterplots and correlations | `python -m analysis.sensitivity_alignment` | `cross_model_sensitivity_alignment.pdf`, `.png` |
| **Figure 5** | Timestep profiles and peak positions | `python -m analysis.timestep_plots` | `timestep_profiles_and_peaks.pdf`, `.png` |
| **Appendix B.4** | Human screening distribution & consistency | `python -m analysis.human` | `response_consistency_histogram.pdf`, `.png` |
| **Appendix F.2 (Fig. S11)** | Single-image naturalness score alignment | `python -m analysis.sensitivity_alignment` | `single_image_naturalness.pdf`, `.png` |
| **Appendix G (Fig. S12)** | Condition-wise sensitivity heatmap | `python -m analysis.sensitivity_alignment` | `condition_sensitivity_alignment_heatmap.pdf`, `.png` |
| **Appendix H (Fig. S13)** | Condition-wise sensitivity and human alignment scatterplots | `python -m analysis.sensitivity_alignment` | `condition_sensitivity_alignment.pdf`, `.png` |
| **Appendix I (Fig. S14)** | Condition-wise baseline comparisons | `python -m analysis.baseline_comparison` | `condition_baseline_comparison.pdf`, `.png` |
| **Appendix J (Figs. S15–S18)** | Illumination factor profiles (shadow, reflection, light) | `python -m analysis.factor_profiles` | `illumination_factor_profiles_*.pdf`, `.png` |
| **Appendix K (Figs. S19–S21)** | Model parameter count vs. sensitivity/alignment | `python -m analysis.model_size` | `model_size_sensitivity_alignment_with_encoders.pdf`, etc. |
| **Appendix L (Tab. S5, Figs. S22–S24)** | Size-adjusted alignment and scatter plots | `python -m analysis.generation_performance` | `sensitivity_size_adjusted_table.csv`, scatter plots |
| **Appendix M (Figs. S25–S27)** | Spatial error attribution maps | `python -m analysis.spatial_maps` | `spatial_maps_all25_*.pdf`, `.png` |
| **Appendix N (Figs. S28–S30)** | Condition-specific timestep peak profiles | `python -m analysis.timestep_plots` | `condition_timestep_peaks.pdf`, `condition_timestep_profiles_*.pdf` |

For further details on statistical definitions, covariates, and estimation procedures, see [analysis/README.md](analysis/README.md).

---

## Recomputing Model Scores (Mode 2, GPU Required)

To recompute model loss scores or feature distances from raw weights on GPU:

1. Consult [MODEL_INDEX.md](MODEL_INDEX.md) and [environments/README.md](environments/README.md) to set up the appropriate Python environment (e.g., `diffusers`, `wan`, `cogvideox`, `encoders_iqa`).
2. Run pair evaluations from this package directory:
   ```bash
   # Diffusion model native score evaluation (e.g., Stable Diffusion 1.5)
   python -m scoring.score_pairs --model stable_diffusion_15 --preset paper

   # Vision encoder feature distance (e.g., ResNet-50)
   python -m encoders.score_pairs --model resnet50_imagenet

   # Classical / Deep IQA metric (e.g., NIQE)
   python -m iqa.score_pairs --metric niqe
   ```
See [scoring/README.md](scoring/README.md), [encoders/README.md](encoders/README.md), and [iqa/README.md](iqa/README.md) for detailed configuration options.

---

## License

- **Code:** [MIT License](LICENSE)
- **Scientific Data & Annotations:** [CC BY 4.0](LICENSE_DATA.md)
- **Illumination Stimulus Images:** [CC BY-SA 4.0](LICENSE_DATA.md) (adapted from KuBasic / Kubric project)
- **Thatcher Stimulus Generator:** [GPLv3](stimulus_generation/thatcher/LICENSE)
