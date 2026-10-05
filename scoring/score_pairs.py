"""Score original/modified image pairs with a released generative model.

Run from the release directory: python -m scoring.score_pairs --help
"""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pair_io import load_pairs
from data_io import configured_path

from .models import MODEL_CLASSES, load_model


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def save_loss(path: Path, result) -> None:
    """Commit one complete image, so interruption preserves earlier images."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, timesteps=result.timesteps, per_noise=result.per_noise,
            per_timestep=result.per_timestep, spatial=result.spatial,
            mean_loss=result.mean_loss,
        )
    temporary.replace(path)


def read_loss(path: Path, timesteps: np.ndarray = None, draws: int = None, spatial_shape=(64, 64)) -> dict:
    """Read a saved loss file."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_CLASSES), default="stable_diffusion_15")
    parser.add_argument("--data-root", type=Path, help="Override settings.json data_root")
    parser.add_argument("--pairs", type=Path, help="Defaults to the full release pair table")
    parser.add_argument("--image-root", type=Path, help="Defaults to DATA_ROOT/stimuli")
    parser.add_argument("--output-dir", type=Path, help="Defaults to OUTPUT_ROOT/scoring/MODEL/PRESET")
    parser.add_argument("--preset", choices=("sample", "paper"), default="sample")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    data_root = configured_path("data_root", args.data_root)
    args.pairs = args.pairs or data_root / "pairs.csv"
    args.image_root = args.image_root or data_root / "stimuli"
    args.output_dir = args.output_dir or configured_path("output_root") / "scoring" / args.model / args.preset
    image_root = args.image_root.resolve()
    pairs = load_pairs(args.pairs, image_root)
    config_path = RELEASE_ROOT / "configs" / f"{args.model}.json"
    config = json.loads(config_path.read_text())
    protocol = config["presets"][args.preset]
    run = {
        "model": args.model,
        "preset": args.preset,
        "config": config,
        "device": args.device,
        "pairs": pairs,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "diffusers", "transformers", "numpy", "Pillow",
        )},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run.json"
    if not run_path.exists():
        temporary = run_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, indent=2) + "\n")
        temporary.replace(run_path)

    timestep_path = args.output_dir / "evaluation_timesteps.npy"
    timesteps = np.load(timestep_path) if timestep_path.exists() else None
    if timesteps is not None and (timesteps.shape != (len(protocol["timestep_indices"]),)
                                  or not np.isfinite(timesteps).all()):
        raise ValueError("Invalid saved evaluation timesteps.")
    model = None
    summaries = []
    for pair in pairs:
        results = {}
        for role in ("original", "modified"):
            path = args.output_dir / f"{pair['pair_id']}_{role}.npz"
            if not path.exists():
                if model is None:
                    model = load_model(args.model, config, args.preset, args.device)
                with Image.open(image_root / pair[role]) as image:
                    result = model.score_image(image)
                if timesteps is None:
                    timesteps = result.timesteps
                    temporary = timestep_path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, timesteps)
                    temporary.replace(timestep_path)
                if not np.array_equal(timesteps, result.timesteps):
                    raise ValueError("Evaluation timesteps changed between images.")
                save_loss(path, result)
            if timesteps is None:
                raise ValueError("Completed image exists without evaluation_timesteps.npy.")
            results[role] = read_loss(path, timesteps, protocol["noise_draws"],
                                      config.get("spatial_shape", (64, 64)))

        # Pair scores use the saved, unrounded per-timestep losses.
        paired_score = (
            results["modified"]["per_timestep"].mean()
            - results["original"]["per_timestep"].mean()
        )
        summaries.append({
            "pair_id": pair["pair_id"], "condition": pair["condition"],
            "original_mean_loss": float(results["original"]["mean_loss"]),
            "modified_mean_loss": float(results["modified"]["mean_loss"]),
            "paired_score": float(paired_score),
        })
        temporary = args.output_dir / "scores.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        temporary.replace(args.output_dir / "scores.csv")
        print(f"{pair['pair_id']}: paired score = {paired_score:.8g}", flush=True)


if __name__ == "__main__":
    main()
