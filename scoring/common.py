"""Small shared pieces: images, fixed noise, and loss aggregation.

Model-specific preprocessing, noising paths, and prediction targets live in the
individual model files. No denoising trajectory is simulated when scoring.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch


@dataclass
class ImageLoss:
    timesteps: np.ndarray           # [T], float32, in evaluation order
    per_noise: np.ndarray           # [T, N], float32
    per_timestep: np.ndarray        # [T], float64 noise-draw mean
    spatial: np.ndarray             # [T, H, W], float32; videos average frames

    @property
    def mean_loss(self) -> float:
        return float(self.per_timestep.astype(np.float32).mean())


def rgb_tensor(image: Image.Image, device) -> torch.Tensor:
    pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.uint8))
    return pixels.permute(2, 0, 1).contiguous().float().div(255).unsqueeze(0).to(device)


def image_pixels(image: Image.Image, device) -> torch.Tensor:
    """RGB float32 in [0,1], bilinear 512 resize, without uint8 requantization.

    This uses the same interpolation kernel as the video and encoder inputs.
    Videos retain their checkpoint-specific aspect-preserving fit/pad geometry.
    """
    return torch.nn.functional.interpolate(
        rgb_tensor(image, device), size=(512, 512), mode="bilinear",
        align_corners=False, antialias=False,
    )


def pixel_tensor(image: Image.Image, device, dtype=torch.float32) -> torch.Tensor:
    return ((image_pixels(image, device) - 0.5) / 0.5).to(dtype)


def noise_bank(shape: tuple, draws: int, seed: int, device, dtype) -> list:
    """CPU float32 draws, then conversion; reuse this bank across all images."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(shape, generator=generator, dtype=torch.float32).to(device, dtype)
            for _ in range(draws)]


def collect_losses(timesteps, indices, noise, loss_at) -> ImageLoss:
    """Evaluate selected grid positions; loss_at returns a scalar and [1,H,W].

    Loss residuals and reductions use float32; model forward precision is
    checkpoint-specific. Noise/time averaging follows the saved arrays.
    """
    losses, maps = [], []
    for index in indices:
        draw_losses, draw_maps = [], []
        for draw in noise:
            scalar, spatial = loss_at(index, draw)
            draw_losses.append(float(scalar.float().cpu()))
            draw_maps.append(spatial)
        losses.append(draw_losses)
        maps.append(torch.stack(draw_maps).mean(dim=0)[0].float().cpu().numpy())
    values = np.asarray(losses, dtype=np.float32)
    spatial = np.stack(maps)
    if not np.isfinite(values).all() or not np.isfinite(spatial).all():
        raise ValueError("Non-finite denoising loss; no score should be saved.")
    return ImageLoss(timesteps[indices].float().cpu().numpy(), values,
                     values.mean(axis=1, dtype=np.float64), spatial)


def image_mse(error: torch.Tensor):
    """Float32 squared error, channel mean, then spatial mean."""
    spatial = error.float().square().mean(dim=1)
    return spatial.mean(dim=(1, 2))[0], spatial
