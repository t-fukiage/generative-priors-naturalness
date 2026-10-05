# Vision Encoder Feature Distances

This module extracts intermediate and endpoint feature representations from 12 vision encoders and computes cosine distances for stimulus pairs.

Readout architectures are detailed in Appendix D.1; layer and Top-5 selection rules are described in Appendix E.3. Configurations and environment requirements are listed in [MODEL_INDEX.md](../MODEL_INDEX.md).

---

## Execution

Run from the repository root in the `encoders_iqa` (or `vlm` / `diffusers`) environment:

```bash
python -m encoders.score_pairs --model clip_vit_b16
```

- **Input Data:** Evaluates pairs from `data/pairs.csv` using images in `data/stimuli/`.
- **Options:** Use `--output-dir` to customize the output directory (default: `results/encoders/<model>/`).

---

## Output Format

`scores.csv` contains:
- `pair_id`: Unique stimulus pair identifier.
- `condition`: Task condition.
- `candidate`: Intermediate block number (1-based) or `endpoint`.
- `score`: Feature cosine distance between modified and original image representations.

`run.json` records run configurations, input hashes, and dependency versions. Re-running the command automatically resumes incomplete runs.
