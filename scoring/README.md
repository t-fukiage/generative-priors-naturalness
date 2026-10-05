# Generative-Model Scoring

This module evaluates model-native score / loss profiles for the 25 generative models on the stimulus image pairs.

Model-native loss definitions and input settings are described in Section 3 and Appendix C. Model implementations, configurations, and runtime environments are listed in [MODEL_INDEX.md](../MODEL_INDEX.md).

---

## Execution

Run from the repository root in the appropriate model environment:

```bash
python -m scoring.score_pairs --model stable_diffusion_15 --preset paper
```

- **Input Data:** Evaluates pairs from `data/pairs.csv` using images in `data/stimuli/`.
- **Presets:** `--preset paper` uses the exact evaluation grid from the paper (100 timesteps for image models, 50 for video models, with 20 noise draws per timestep).
- **Options:** Use `--output-dir` to customize the destination directory (default: `results/scoring/<model>/`).

---

## Output Format

Each stimulus image produces `<pair_id>_<original|modified>.npz`:

| Array | Shape | Meaning |
| --- | --- | --- |
| `timesteps` | `[T]` | Native evaluation times |
| `per_noise` | `[T, N]` | Scalar losses per timestep and noise draw |
| `per_timestep` | `[T]` | Float64 means over noise draws |
| `spatial` | `[T, H, W]` | Mean spatial loss maps (video frames averaged) |
| `mean_loss` | scalar | Mean of float32 timestep means |

Additionally, `scores.csv` records image losses and modified-minus-original pair scores (computed from unrounded float64 timestep means). `run.json` records configuration parameters, input checksums, and package versions. Completed images are cached, allowing interrupted runs to be resumed by re-running the command.
