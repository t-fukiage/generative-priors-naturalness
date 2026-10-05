# SPDX-License-Identifier: GPL-3.0-only
# Derived from Thatcher Effect Dataset Generator (https://github.com/Erfaniaa/thatcher-effect-dataset-generator)
"""Facial landmark detection using dlib 68-point shape predictor."""
from pathlib import Path


class LandmarkDetector:
    def __init__(self, predictor_path):
        path = Path(predictor_path)
        if not path.is_file():
            raise ValueError(f"Predictor file not found: {predictor_path}")
        with path.open("rb") as f:
            if f.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Predictor is a Git LFS pointer, not model weights")
        import cv2
        import dlib
        self.cv2 = cv2
        self.version = dlib.__version__
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(path))

    def __call__(self, image):
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) != 1:
            return []
        shape = self.predictor(gray, rects[0])
        return [(int(shape.part(i).x), int(shape.part(i).y)) for i in range(shape.num_parts)]
