"""Frozen PyIQA metrics, with explicit pair order and score direction."""

import numpy as np
import pyiqa
import torch

from scoring.common import image_pixels


class QualityMetric:
    def __init__(self, config, device="cuda:0"):
        if config.get("input_resize", "bilinear") != "bilinear":
            raise ValueError("IQA inputs use the fixed bilinear protocol.")
        self.device = device
        self.config = config
        torch.manual_seed(0)
        np.random.seed(0)
        self.model = pyiqa.create_metric(config["backend"], device=device).eval()
        if self.model.metric_mode.upper() != config["mode"]:
            raise ValueError("PyIQA reference mode differs from the configured metric.")
        if bool(self.model.lower_better) != config["lower_better"]:
            raise ValueError("PyIQA quality direction differs from the configured metric.")

    @torch.inference_mode()
    def score_pair(self, original, modified):
        original = image_pixels(original, self.device)
        modified = image_pixels(modified, self.device)
        if self.config["mode"] == "FR":
            value = float(self.model(modified, original).float().item())
            score = value if self.config["lower_better"] else -value
            result = {"raw_pair_quality": value, "score": score}
        else:
            original_quality = float(self.model(original).float().item())
            modified_quality = float(self.model(modified).float().item())
            difference = modified_quality - original_quality
            score = difference if self.config["lower_better"] else -difference
            result = {"original_quality": original_quality, "modified_quality": modified_quality, "score": score}
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Non-finite IQA score.")
        return result
