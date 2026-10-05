# Spatial Attribution Maps

This directory contains data files for generating Figures S25–S27 (Appendix M).

---

## File Inventory

| File | Format / Shape | Description |
| --- | --- | --- |
| `condition_maps.npz` | 125 float32 arrays of shape `[96, 96]` | Condition-mean spatial loss maps, keyed by `task__condition__model` |
| `models.json` | JSON catalog | Model ordering, display labels, native input geometry, and crop bounds |
| `task_scales.csv` | CSV table | Task-wide sample standard deviations used for normalization |

All spatial arrays are precomputed condition means, scaled by task standard deviations, and cropped/resized for comparative visualization. Run `python -m analysis.spatial_maps` from the repository root to render Figures S25–S27.
