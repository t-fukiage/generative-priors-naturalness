"""Layer capture and cosine distance; pooling itself is model-specific."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def canvas(image, device="cpu"):
    """RGB float32 tensor, bilinear 512 resize without antialiasing."""
    image = image.convert("RGB")
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(device)
    return F.interpolate(pixels, size=(512, 512), mode="bilinear",
                         align_corners=False, antialias=False)[0].cpu()


def normalize_pixels(images, mean, std):
    """Normalize float RGB tensors without a lossy round trip through PIL."""
    pixels = torch.stack(images)
    mean = pixels.new_tensor(mean).view(1, 3, 1, 1)
    std = pixels.new_tensor(std).view(1, 3, 1, 1)
    return (pixels - mean) / std


def normalized(features):
    features = features.detach().float().cpu()
    if not torch.isfinite(features).all() or torch.any(features.norm(dim=-1) <= 1e-12):
        raise ValueError("Cosine distance requires finite, nonzero features.")
    return F.normalize(features, dim=-1)


def distances(original, modified):
    if original.shape != modified.shape:
        raise ValueError("Paired feature shapes differ.")
    result = 1 - (original * modified).sum(dim=-1)
    # DINO retains corresponding spatial patches rather than pooling the image.
    return result.mean(dim=1) if result.ndim == 2 else result


@dataclass
class LayerEncoder:
    encode: object
    blocks: list
    readout: object
    model: object

    @torch.inference_mode()
    def features(self, images):
        captured = {}

        def capture(layer):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[str(layer)] = normalized(self.readout(hidden))
            return hook

        handles = [block.register_forward_hook(capture(i)) for i, block in enumerate(self.blocks, 1)]
        try:
            device = next(self.model.parameters()).device
            prepared = [canvas(image, device) for image in images]
            captured["endpoint"] = normalized(self.encode(prepared))
            if len(captured) != len(self.blocks) + 1:
                raise ValueError("Some encoder blocks were not captured.")
        finally:
            for handle in handles:
                handle.remove()
        return captured
