# Resampling Inputs

This directory provides model family assignments and paired text-conditioning control scores, supporting the clustered bootstrap resampling pipeline in `analysis.bootstrap` (Appendices E and O).

---

## Overview

- **Model Family Clustering:** Groups the 25 generative models into 11 distinct architectural families to prevent over-representing heavily sampled families during resampling (Appendix E.2).
- **Text Conditioning Control:** Contains modified-minus-original scores across nine generative models evaluated under empty vs. descriptive prompt conditioning to test text dependency (Appendix O).

---

## File Inventory

| File | Rows / Size | Description |
| --- | --- | --- |
| `model_families.csv` | 26 rows | Assignment mapping 25 generative models to 11 family clusters (Table S1) |
| `text_conditioning.csv` | 10,225 rows | Paired native loss scores across 9 models for empty (`empty`) and description (`desc`) prompts |

### Column Schemas

- **`model_families.csv`:**
  - `model`: Unique model identifier.
  - `family`: Architectural family cluster (e.g., `FLUX.1`, `Wan`, `JiT`, `Stable Diffusion`).
  - `modality`: `image` or `video`.
- **`text_conditioning.csv`:**
  - `pair_id`: Unique stimulus pair identifier.
  - `model`: Model identifier.
  - `mode`: Conditioning prompt mode (`empty` or `desc`).
  - `score`: Timestep-averaged native loss difference (modified minus original).
