# Illumination Factors

This directory provides stimulus-to-factor classification mappings and factor-level definitions for the Illumination task, used to generate Figures S15–S18 (Appendix J).

---

## Overview

- **Purpose:** Decomposes the 288 illumination stimulus pairs into 13 perceptual/physical factors and 44 ordered levels (covering shadows, reflections, and lighting directions).
- **Usage:** Run `python -m analysis.factor_profiles` from the repository root to evaluate model–human MSE and render the factor profile figures.

---

## File Inventory

| File | Rows / Size | Description |
| --- | --- | --- |
| `levels.csv` | 45 rows | Catalog of 13 illumination factors, their 44 ordered levels, and display labels |
| `membership.csv` | 289 rows | Mapping connecting each of the 288 illumination pairs to its corresponding factor levels |

### Schema & Relationships

- **`levels.csv`:** Contains `factor_id`, `factor_order`, `factor_label`, `level_order`, `level_label`, and `group`.
- **`membership.csv`:** Contains `pair_id`, `factor_id`, and `level_order`.
- **Join Key:** Join `membership.csv` and `levels.csv` on `(factor_id, level_order)`. `pair_id` joins directly with `data/pair_metadata.csv`.
