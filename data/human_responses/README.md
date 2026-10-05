# Human Ratings

This directory provides individual participant ratings and stimulus presentation logs for the Thatcher and Illumination (Swap) perceptual tasks (Section 3 and Appendix B).

---

## Overview

- **Task Coverage:** 48,702 main-task trials from 162 Thatcher submissions (162 participants) and 167 Illumination submissions (165 participants).
- **Inclusions:** Includes all submitted trials, repeat presentations, duplicate submissions, and subsequently excluded participants for complete auditability.
- **Pipeline:** Executing `python -m analysis.human` applies the leave-one-out Silverman KDE screening rule (Appendix B.4), writes screening decisions and summaries, and outputs clean item aggregate ratings.

---

## File Inventory

| File | Rows / Size | Description |
| --- | --- | --- |
| `responses.csv` | 48,702 data rows (~2.4 MB) | Individual single-trial ratings and presentation metadata across all participants |

### Column Schema (`responses.csv`)

| Column | Type | Meaning |
| --- | --- | --- |
| `task` | string | Task identifier: `thatcher` or `swap` (Illumination) |
| `participant` | string | Task-local anonymized participant ID (no cross-task linkage) |
| `submission` | integer | Submission order within participant (1-based) |
| `assignment_group` | string | Stimulus assignment group for stratified resampling |
| `trial` | integer | Presentation order within the main experiment block (1-based) |
| `pair_id` | string | Unique stimulus pair key matching `data/pairs.csv` and `data/pair_metadata.csv` |
| `version` | string | Stimulus version evaluated: `original` or `modified` |
| `rating` | integer | Naturalness rating on a 1–5 Likert scale (higher indicates greater naturalness) |

---
