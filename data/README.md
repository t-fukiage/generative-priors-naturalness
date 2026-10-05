# Dataset Overview

`losses/` contains all 25 model-loss NPZ files. `stimuli/` contains the 576
Illumination PNGs; Thatcher PNGs are generated separately from official FFHQ
inputs. No external archive is required for CPU analyses.

This directory provides stimulus pairs, precomputed model evaluation scores, participant ratings, and task factor annotations.

---

## Directory Inventory

| Path | Contents | Paper Reference |
| --- | --- | --- |
| `data/` (Root) | 568 image pairs (`pairs.csv`), included model loss profiles (`losses/`), item aggregates (`human_aggregates.csv`), and Illumination images (`stimuli/`) | Appendices A–E |
| [human_responses](human_responses/README.md) | Individual participant ratings (`responses.csv`), including repeat and exclusion trials | Appendix B |
| [resampling_inputs](resampling_inputs/README.md) | Generative model family memberships and paired text-conditioning scores | Appendices E and O |
| [generation_evaluations](generation_evaluations/README.md) | External benchmark evaluation scores and model mappings | Appendix L |
| [illumination_factors](illumination_factors/README.md) | Stimulus-to-factor classification mappings | Appendix J |
| [model_sizes](model_sizes/README.md) | Generative and encoder parameter counts | Appendix K |
| [spatial_maps](spatial_maps/README.md) | Condition-mean spatial error attribution maps (`condition_maps.npz`) | Appendix M |

- **Storage Requirements:** Included loss NPZ files occupy ~138 MB; the 576 Illumination PNGs occupy ~238 MB. Other bundled data occupy about 10.5 MB. Thatcher faces are not bundled.
- **Core bundle location:** Core scalar tables are directly in `data/`, with model losses in `data/losses/` and source images in `data/stimuli/`. Participant ratings and supplementary metadata use the directories listed above.
- **Usage:** Full post-hoc CPU reproduction requires the bundled files only. Raw model scoring (GPU) additionally requires the corresponding source images and external model weights.
- **Field Definitions:** See [SCHEMA.md](SCHEMA.md) for full column schemas and array specifications.

Data licensing terms are specified in [LICENSE_DATA.md](../LICENSE_DATA.md).
