"""PixelGen 512: native velocity score with the EMA denoiser, including t=0."""

import numpy as np
import torch
from huggingface_hub import snapshot_download
from src.models.transformer.JiT_T2I import JiT_T2I
from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder

from .common import collect_losses, image_mse, noise_bank, pixel_tensor


class PixelGen:
    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.config = config
        self.protocol = config["presets"][preset]
        self.model = JiT_T2I(
            patch_size=16, input_size=512, in_channels=3, hidden_size=1536,
            num_blocks=16, num_groups=24, txt_embed_dim=2048, txt_max_length=128,
            bottleneck_dim=256, num_text_blocks=4,
        )
        checkpoint = torch.load(config["checkpoint_path"], map_location="cpu", mmap=True, weights_only=False)
        state = {key.removeprefix("ema_denoiser."): value
                 for key, value in checkpoint["state_dict"].items() if key.startswith("ema_denoiser.")}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        text = config["text_checkpoint"]
        text_path = snapshot_download(repo_id=text["repo_id"], revision=text["revision"])
        self.conditioner = Qwen3TextEncoder(weight_path=text_path,
                                            embed_dim=2048, max_length=128).to(self.device).eval()
        with torch.no_grad():
            _, uncondition = self.conditioner([""], {"negative_prompt": ""})
        self.prompt = uncondition.to(self.device, next(self.model.parameters()).dtype)
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
            t = timestep.view(1, 1, 1, 1)
            sigma = 1.0 - t
            noisy = t * clean + sigma * noise
            prediction = self.model(noisy, timestep, self.prompt)
            residual = (prediction.float() - clean.float()) / sigma.float().clamp_min(self.config["t_eps"])
            return image_mse(-residual)

        # One draw per forward preserves the configured batch size.
        return collect_losses(timesteps, self.protocol["timestep_indices"], self.noise, loss_at)
