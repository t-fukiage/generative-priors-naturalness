"""Compute one IQA metric on the public original/modified pair table."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pair_io import load_pairs, prepare_run, write_csv
from data_io import configured_path
from .quality import QualityMetric


RELEASE = Path(__file__).resolve().parents[1]


def main():
    configs = json.loads((RELEASE / "configs/iqa.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=tuple(configs), required=True)
    parser.add_argument("--data-root", type=Path, help="Override settings.json data_root")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    data_root = configured_path("data_root", args.data_root)
    args.pairs = args.pairs or data_root / "pairs.csv"
    args.image_root = args.image_root or data_root / "stimuli"
    args.output_dir = args.output_dir or configured_path("output_root") / "iqa" / args.metric
    config = {**configs[args.metric], "input_resize": "bilinear"}
    pairs = load_pairs(args.pairs, args.image_root)
    prepare_run(args.output_dir, {"metric": args.metric, "settings": config, "device": args.device}, pairs,
                packages=["torch", "pyiqa", "numpy", "Pillow"])
    path = args.output_dir / "scores.csv"
    rows = list(csv.DictReader(path.open())) if path.exists() else []
    if len(rows) == len(pairs):
        return
    model = QualityMetric(config, args.device)
    for pair in pairs[len(rows):]:
        with Image.open(args.image_root / pair["original"]) as original, \
             Image.open(args.image_root / pair["modified"]) as modified:
            result = model.score_pair(original, modified)
        rows.append({"pair_id": pair["pair_id"], "condition": pair["condition"], **result})
        write_csv(path, rows)
        print(f"{args.metric}: {len(rows)}/{len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
