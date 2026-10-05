# IQA Metric Evaluations

This module computes perceptual quality and distortion metrics for image pairs using 10 standard Image Quality Assessment (IQA) baselines.

Metric definitions, normalization, and score orientations are detailed in Appendices D.2–D.3 (oriented such that higher scores indicate greater perceptual degradation). Model keys and environment requirements are listed in [MODEL_INDEX.md](../MODEL_INDEX.md).

---

## Execution

Run from the repository root in the `encoders_iqa` environment:

```bash
python -m iqa.score_pairs --metric dists
```

- **Input Data:** Evaluates pairs from `data/pairs.csv` using images in `data/stimuli/`.
- **Options:** Use `--output-dir` to customize the output directory (default: `results/iqa/<metric>/`).

---

## Output Format

`scores.csv` contains:
- `pair_id`: Unique stimulus pair identifier.
- `condition`: Task condition.
- `score`: Pair degradation score (modified minus original, or full-reference distortion distance).
- `raw_pair_quality` or `original_quality,modified_quality`: Unadjusted individual/pair metrics.

`run.json` records run parameters, input hashes, and package versions. Re-running the command automatically resumes incomplete runs.
