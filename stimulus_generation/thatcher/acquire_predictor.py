# SPDX-License-Identifier: GPL-3.0-only
# Derived from Thatcher Effect Dataset Generator (https://github.com/Erfaniaa/thatcher-effect-dataset-generator)
"""Download and extract the official dlib 68-point facial landmark predictor."""
import argparse
import bz2
import hashlib
from pathlib import Path
from acquire_ffhq import download, hashes

URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
ARCHIVE_BYTES = 64040097
ARCHIVE_SHA256 = "7d6637b8f34ddb0c1363e09a4628acb34314019ec3566fd66b80c04dda6980f5"
WEIGHT_BYTES = 99693937
WEIGHT_SHA256 = "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True, help="Directory for the downloaded predictor")
    args = p.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "shape_predictor_68_face_landmarks.dat.bz2"
    weight = root / "shape_predictor_68_face_landmarks.dat"
    if weight.is_symlink():
        p.error("Refusing an existing weight symlink")
    try:
        if weight.exists():
            if weight.stat().st_size == WEIGHT_BYTES and hashes(weight)[1] == WEIGHT_SHA256:
                print(f"Predictor already exists and is verified: {weight}")
                return
            raise ValueError("Existing predictor differs from expected checksum")
        download(URL, archive, ARCHIVE_BYTES, expected_sha256=ARCHIVE_SHA256)
        part = weight.with_suffix(".dat.part")
        if part.exists():
            part.unlink()
        digest = hashlib.sha256()
        size = 0
        with bz2.open(archive, "rb") as source, part.open("xb") as target:
            while True:
                block = source.read(1048576)
                if not block:
                    break
                size += len(block)
                if size > WEIGHT_BYTES:
                    raise ValueError("Predictor exceeds expected decompressed size")
                digest.update(block)
                target.write(block)
        if size != WEIGHT_BYTES or digest.hexdigest() != WEIGHT_SHA256:
            raise ValueError("Predictor checksum mismatch")
        part.rename(weight)
        print(f"Successfully downloaded and extracted predictor: {weight}")
    except (ValueError, OSError) as exc:
        p.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
