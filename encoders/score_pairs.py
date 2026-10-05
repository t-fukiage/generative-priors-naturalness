"""Save every configured layer readout and the official endpoint for image pairs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pair_io import load_pairs, prepare_run, write_csv
from data_io import configured_path
from .common import distances
from .models import load_encoder


RELEASE = Path(__file__).resolve().parents[1]


def main():
    configs = json.loads((RELEASE / "configs/encoders.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(configs), required=True)
    parser.add_argument("--data-root", type=Path, help="Override settings.json data_root")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    data_root = configured_path("data_root", args.data_root)
    args.pairs = args.pairs or data_root / "pairs.csv"
    args.image_root = args.image_root or data_root / "stimuli"
    args.output_dir = args.output_dir or configured_path("output_root") / "encoders" / args.model
    config = {**configs[args.model], "input_resize": "bilinear"}
    pairs = load_pairs(args.pairs, args.image_root)
    prepare_run(args.output_dir, {"model": args.model, "settings": config, "device": args.device}, pairs,
                packages=["torch", "transformers", "numpy", "Pillow"])
    path = args.output_dir / "scores.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    candidates = [str(i) for i in range(1, config["layers"] + 1)] + ["endpoint"]
    per_pair = len(candidates)
    completed = len(rows) // per_pair
    if completed == len(pairs):
        return
    batch = config["batch_size"]
    encoder = load_encoder(args.model, config, args.device)
    for start in range(completed, len(pairs), batch):
        selected = pairs[start:start + batch]
        features = {}
        for role in ("modified", "original"):
            images = []
            for pair in selected:
                with Image.open(args.image_root / pair[role]) as image:
                    images.append(image.convert("RGB"))
            features[role] = encoder.features(images)
        scores = {candidate: distances(features["original"][candidate], features["modified"][candidate])
                  for candidate in candidates}
        for index, pair in enumerate(selected):
            rows.extend({"pair_id": pair["pair_id"], "condition": pair["condition"],
                         "candidate": candidate, "score": float(scores[candidate][index])}
                        for candidate in candidates)
        write_csv(path, rows)
        print(f"{args.model}: {start + len(selected)}/{len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
