"""JiT B/32, L/32, H/32: null class and pixel-space velocity residuals.

Install the pinned official JiT model definitions as described in environments/.
No training, classifier-free guidance, or alternate proxy is part of this path.
"""

import numpy as np
import torch
from model_jit import JiT_models

from .common import collect_losses, image_mse, noise_bank, pixel_tensor


class JiT:
    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        self.config = config
        self.protocol = config["presets"][preset]
        self.model = JiT_models[config["architecture"]](input_size=512)
        checkpoint = torch.load(config["checkpoint_path"], map_location="cpu", mmap=True, weights_only=True)
        # Load the EMA weights and strip their wrapper prefix.
        state = {key.removeprefix("net."): value for key, value in checkpoint["model_ema1"].items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.labels = torch.full((1,), 1000, device=self.device, dtype=torch.long)
        self.noise = noise_bank((1, 3, 512, 512), self.protocol["noise_draws"],
                                config["noise_seed"], self.device, torch.float32)

    @torch.no_grad()
    def score_image(self, image):
        clean = pixel_tensor(image, self.device)
        grid = torch.linspace(0.0, self.config["t_max"], self.config["training_steps"], device=self.device)
        positions = np.linspace(0, len(grid) - 1, self.config["evaluation_steps"], dtype=int)
        timesteps = grid[positions]

        def loss_at(index, noise):
            timestep = timesteps[index].expand(1)
            t = timestep.reshape(1, 1, 1, 1)
            noisy = t * clean + (1.0 - t) * (self.config["noise_scale"] * noise)
            prediction = self.model(noisy, timestep, self.labels)
            denominator = (1.0 - t).clamp_min(self.config["t_eps"])
            # Subtract and divide each term separately to retain float32 rounding.
            target_velocity = (clean - noisy) / denominator
            predicted_velocity = (prediction - noisy) / denominator
            return image_mse(target_velocity - predicted_velocity)

        return collect_losses(timesteps, self.protocol["timestep_indices"], self.noise, loss_at)
