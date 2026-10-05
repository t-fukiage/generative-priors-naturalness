"""Shared pair-table validation and atomic run records for the public runners."""

import csv
import importlib.metadata
import json
from pathlib import Path


def load_pairs(path, image_root):
    image_root = Path(image_root).resolve()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"pair_id", "condition", "original", "modified"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"The pairs CSV must contain columns: {sorted(required)}")
        pairs = list(reader)
    if not pairs:
        raise ValueError("The pairs CSV is empty.")
    for pair in pairs:
        for role in ("original", "modified"):
            resolved = (image_root / Path(pair[role])).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Missing {role} image for {pair['pair_id']}: {pair[role]}")
    return pairs


def prepare_run(
    output_dir: Path,
    config: dict,
    pairs: list,
    packages: list = None,
    **kwargs,
) -> None:
    """Record execution configuration and environment for reproducibility."""
    record = {
        "config": config,
        "pairs": pairs,
    }
    if packages:
        record["packages"] = {name: importlib.metadata.version(name) for name in packages}

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run.json"
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)


def write_csv(path: Path, rows: list) -> None:
    """Write table rows to CSV atomically using a temporary file."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

