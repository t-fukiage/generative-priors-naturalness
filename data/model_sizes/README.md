# Model Parameter Counts

This directory contains parameter counts and component scopes for all 37 generative and discriminative models, used to generate Figures S19–S21 (Appendix K).

---

## Overview

- **Model Coverage:** Includes all 25 generative diffusion models and 12 vision encoders evaluated in the paper.
- **Usage:** Run `python -m analysis.model_size` to estimate scaling relationships against human alignment and feature sensitivity.

---

## File Inventory

| File | Rows / Size | Description |
| --- | --- | --- |
| `models.csv` | 38 rows | Parameter counts in billions (`parameter_b`), counting methodology, and included component scopes |

### Column Schema (`models.csv`)

| Column | Type | Meaning |
| --- | --- | --- |
| `group` | string | Model class: `generative` or `encoder` |
| `model` | string | Unique model identifier matching `data/models.json` |
| `display_name` | string | Formatted display name used in paper figures |
| `modality` | string | Evaluation modality: `image` or `video` |
| `parameter_b` | float | Total parameter count expressed in billions |
| `count_kind` | string | Methodology used (e.g., `stated_architecture`, `checkpoint_state_dict`) |
| `count_scope` | string | Specific sub-networks included (e.g., DiT/UNet backbone, text encoders, VAE) |
| `count_note` | string | Additional context or source citation for the parameter estimate |
