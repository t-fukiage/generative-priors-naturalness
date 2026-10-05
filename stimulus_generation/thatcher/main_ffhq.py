# SPDX-License-Identifier: GPL-3.0-only
# Derived from Thatcher Effect Dataset Generator (https://github.com/Erfaniaa/thatcher-effect-dataset-generator)
"""Generate Thatcher effect stimulus sets from FFHQ face images."""
import argparse
import csv
import json
from math import inf
from pathlib import Path, PurePosixPath
import re
import sys

BLUR_SIZE = 21
cv2 = None
DEFAULT_MANIFEST = Path(__file__).parent / "main_sources.csv"


def validate_relative_image_path(value):
    if not re.fullmatch(r"[0-9]{5}/[0-9]{5}\.png", value):
        raise ValueError("Expected an FFHQ relative path #####/#####.png")
    return value


def read_sources(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    seen_scenes, seen_images = set(), set()
    for row in rows:
        scene = row.get("scene_id", "")
        if not re.fullmatch(r"0|[1-9][0-9]*", scene) or scene in seen_scenes:
            raise ValueError(f"Invalid or duplicate scene ID: {scene}")
        image = validate_relative_image_path(row["img_path"])
        if image in seen_images or row.get("condition") != "thatcher":
            raise ValueError(f"Duplicate image or non-thatcher row: {image}")
        seen_scenes.add(scene)
        seen_images.add(image)
    if not rows:
        raise ValueError("Source manifest is empty")
    return rows


def load_metadata(path):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected FFHQ JSON dictionary")
    return data


def is_public_domain(value):
    return "public domain" in str(value.get("metadata", {}).get("license", "")).lower()


def select_sources(mode, manifest, metadata, practice_offset=145):
    if mode == "main":
        return read_sources(manifest)
    if metadata is None:
        raise ValueError("Practice selection requires --metadata")
    candidates = []
    for value in metadata.values():
        path = value.get("image", {}).get("file_path", "")
        if is_public_domain(value) and path.startswith("images1024x1024/"):
            relative = validate_relative_image_path(path[len("images1024x1024/"):])
            candidates.append({
                "img_path": relative,
                "ffhq_id": PurePosixPath(relative).stem,
                "condition": "thatcher"
            })
    return candidates[practice_offset:]


def get_bounding_rectangle(points):
    top_left = [inf, inf]
    bottom_right = [-inf, -inf]
    for point in points:
        top_left[0] = min(top_left[0], point[1])
        top_left[1] = min(top_left[1], point[0])
        bottom_right[0] = max(bottom_right[0], point[1])
        bottom_right[1] = max(bottom_right[1], point[0])
    return [top_left, bottom_right]


def flip_subimage_ellipse_vertically(image, x1, y1, x2, y2):
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    b = (y2 - y1) / 2.0
    a = (x2 - x1) / 2.0
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            dx = x - mid_x
            dy = y - mid_y
            if (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1 and x1 + x2 - x > x:
                image[x][y], image[x1 + x2 - x][y] = image[x1 + x2 - x][y].copy(), image[x][y].copy()


def blur_ellipse_border(image, x1, y1, x2, y2):
    blurred_image = cv2.GaussianBlur(image, (BLUR_SIZE, BLUR_SIZE), 0)
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    b = (y2 - y1) / 2.0
    a = (x2 - x1) / 2.0
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            dx = x - mid_x
            dy = y - mid_y
            if 0.75 <= (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1.25:
                image[x][y] = blurred_image[x][y]


def flip_subimage_ellipse_vertically_with_border_softening(image, x1, y1, x2, y2):
    flip_subimage_ellipse_vertically(image, x1, y1, x2, y2)
    blur_ellipse_border(image, x1, y1, x2, y2)


def expanded_rectangles(landmarks):
    groups = (landmarks[36:42], landmarks[42:48], landmarks[48:68])
    offsets = ((-5, -6, 7, 3), (-5, -3, 7, 6), (-4, -5, 3, 5))
    rectangles = []
    for points, (dr1, dc1, dr2, dc2) in zip(groups, offsets):
        (r1, c1), (r2, c2) = get_bounding_rectangle(points)
        rectangles.append((int(r1 + dr1), int(c1 + dc1), int(r2 + dr2), int(c2 + dc2)))
    return rectangles


def apply_thatcher_effect(image, landmarks):
    if image.shape[:2] != (1024, 1024):
        raise ValueError("Expected 1024x1024 image")
    rectangles = expanded_rectangles(landmarks)
    for r1, c1, r2, c2 in rectangles:
        if not (0 <= r1 < r2 < image.shape[0] and 0 <= c1 < c2 < image.shape[1]):
            raise ValueError("Feature rectangle is outside the image bounds")
    original = image.copy()
    modified = image.copy()
    for rectangle in rectangles:
        flip_subimage_ellipse_vertically_with_border_softening(modified, *rectangle)
    return {
        "original_str.png": original,
        "original_inv.png": cv2.rotate(original, cv2.ROTATE_180),
        "modified_str.png": modified,
        "modified_inv.png": cv2.rotate(modified, cv2.ROTATE_180)
    }, rectangles


def input_path(root, relative):
    return (root / relative).resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("main", "practice"), required=True, help="Generation mode")
    p.add_argument("--input-root", type=Path, help="Directory containing FFHQ image folders (e.g. 00000/)")
    p.add_argument("--output", type=Path, help="Output directory")
    p.add_argument("--predictor", type=Path, help="Path to shape_predictor_68_face_landmarks.dat")
    p.add_argument("--metadata", type=Path, help="Path to selected_metadata.json or ffhq-dataset-v2.json")
    p.add_argument("--source-manifest", type=Path, help="Custom scene mapping CSV (default: main_sources.csv)")
    p.add_argument("--practice-offset", type=int, default=145, help="Offset for practice face selection")
    p.add_argument("--limit", type=int, help="Limit number of generated scenes")
    p.add_argument("--dry-run", action="store_true", help="Validate configuration without generating images")
    return p


def run(args):
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    manifest = args.source_manifest or DEFAULT_MANIFEST
    metadata = load_metadata(args.metadata) if args.metadata else None
    candidates = select_sources(args.mode, manifest, metadata, args.practice_offset)
    limit = args.limit or (len(candidates) if args.mode == "main" else 50)
    candidates = candidates[:limit]

    if not candidates:
        raise ValueError("No source candidates found")

    if args.dry_run:
        available = sum(input_path(args.input_root, r["img_path"]).is_file() for r in candidates) if args.input_root else None
        print(f"Dry run: {len(candidates)} candidates, {available} available on disk")
        return

    if not all((args.input_root, args.output, args.predictor)):
        raise ValueError("Generating stimuli requires --input-root, --output, and --predictor")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output directory exists and is not empty: {args.output}")

    from facial_landmark_detection import LandmarkDetector
    detector = LandmarkDetector(args.predictor)
    global cv2
    import cv2

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "info.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("scene_id", "img_path", "condition"))
        writer.writeheader()
        generated_count = 0
        for row in candidates:
            relative = row["img_path"]
            path = input_path(args.input_root, relative)
            if not path.is_file():
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not load image: {relative}")
            landmarks = detector(image)
            if len(landmarks) != 68:
                if args.mode == "main":
                    raise ValueError(f"Expected exactly 1 face for scene {row['scene_id']}, found {len(landmarks)}")
                continue
            images, _ = apply_thatcher_effect(image, landmarks)
            scene_id = row["scene_id"] if args.mode == "main" else str(generated_count)
            scene_dir = args.output / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            for name, pixels in images.items():
                target = scene_dir / name
                if not cv2.imwrite(str(target), pixels):
                    raise OSError(f"Failed to write image: {target}")
            writer.writerow({"scene_id": scene_id, "img_path": relative, "condition": "thatcher"})
            f.flush()
            generated_count += 1
            if generated_count >= limit:
                break

    if generated_count != limit:
        raise ValueError(f"Generated {generated_count} scenes, expected {limit}")
    print(f"Successfully generated {generated_count} Thatcher scenes in {args.output}")


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, ImportError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
