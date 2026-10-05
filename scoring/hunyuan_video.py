"""HunyuanVideo: empty prompt, neutral embedded guidance, native flow loss."""

import numpy as np
import torch
from diffusers import HunyuanVideoPipeline

from .video_common import StaticVideoScore, video_mse


class HunyuanVideo(StaticVideoScore):
    def load_pipeline(self):
        checkpoint = self.config["checkpoint"]
        return HunyuanVideoPipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], torch_dtype=self.dtype,
        )

    def encode_prompt(self):
        self.prompt, self.pooled, self.mask = self.pipe.encode_prompt(
            prompt="", prompt_2="", device=self.device, max_sequence_length=256,
        )
        self.prompt, self.pooled = self.prompt.to(self.dtype), self.pooled.to(self.dtype)
        # Generation-time guidance=6 was not used by the scoring path.
        self.guidance = torch.tensor([1.0], device=self.device, dtype=self.dtype) * 1000.0

    def encode_video(self, frame):
        latent = self.pipe.vae.encode(self.clip(frame)).latent_dist.mode().float()
        return latent * float(self.pipe.vae.config.scaling_factor)

    def evaluation_timesteps(self, clean):
        steps = self.config["evaluation_steps"]
        self.pipe.scheduler.set_timesteps(
            steps, device=self.device, sigmas=np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[:-1],
        )
        return self.pipe.scheduler.timesteps

    def loss(self, clean, noise, timestep):
        noisy = self.add_noise(clean, noise, timestep).to(self.dtype)
        with self.pipe.transformer.cache_context("cond"):
            prediction = self.pipe.transformer(
                hidden_states=noisy, timestep=timestep.expand(1).to(self.dtype),
                encoder_hidden_states=self.prompt, encoder_attention_mask=self.mask,
                pooled_projections=self.pooled, guidance=self.guidance,
                attention_kwargs=None, return_dict=False,
            )[0]
        return video_mse((prediction.float() - (noise.float() - clean.float())).square())
