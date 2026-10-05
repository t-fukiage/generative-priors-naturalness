"""LTX-Video 2B/13B: normalized VAE modes, packed flow targets, static clips."""

import numpy as np
import torch
from diffusers import LTXConditionPipeline, LTXPipeline
from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift

from .video_common import StaticVideoScore


class LTXVideo(StaticVideoScore):
    def load_pipeline(self):
        checkpoint = self.config["checkpoint"]
        pipeline = LTXConditionPipeline if self.config["pipeline"] == "condition" else LTXPipeline
        return pipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], torch_dtype=self.dtype,
        )

    def encode_prompt(self):
        self.prompt, self.mask, _, _ = self.pipe.encode_prompt(
            prompt="", negative_prompt="", do_classifier_free_guidance=False,
            max_sequence_length=self.config["max_sequence_length"],
            device=self.device, dtype=self.dtype,
        )

    def encode_video(self, frame):
        latent = self.pipe.vae.encode(self.clip(frame)).latent_dist.mode().float()
        return self.pipe._normalize_latents(latent, self.pipe.vae.latents_mean,
                                            self.pipe.vae.latents_std,
                                            self.pipe.vae.config.scaling_factor)

    def evaluation_timesteps(self, clean):
        self.geometry = tuple(clean.shape[-3:])
        steps = self.config["evaluation_steps"]
        config = self.pipe.scheduler.config
        shift = calculate_shift(int(np.prod(self.geometry)), config.get("base_image_seq_len", 256),
                                config.get("max_image_seq_len", 4096), config.get("base_shift", 0.5),
                                config.get("max_shift", 1.15))
        self.pipe.scheduler.set_timesteps(
            steps, device=self.device, sigmas=np.linspace(1.0, 1 / steps, steps, dtype=np.float32), mu=shift,
        )
        return self.pipe.scheduler.timesteps

    def loss(self, clean, noise, timestep):
        pipe = self.pipe
        spatial_patch = pipe.transformer_spatial_patch_size
        temporal_patch = pipe.transformer_temporal_patch_size
        noisy = pipe._pack_latents(self.add_noise(clean, noise, timestep), spatial_patch, temporal_patch)
        target = pipe._pack_latents(noise.float() - clean.float(), spatial_patch, temporal_patch)
        frames, height, width = self.geometry
        with pipe.transformer.cache_context("cond_uncond"):
            prediction = pipe.transformer(
                hidden_states=noisy.to(self.dtype), encoder_hidden_states=self.prompt,
                timestep=timestep.expand(1), encoder_attention_mask=self.mask,
                num_frames=frames, height=height, width=width,
                rope_interpolation_scale=(pipe.vae_temporal_compression_ratio / self.config["frame_rate"],
                                          pipe.vae_spatial_compression_ratio,
                                          pipe.vae_spatial_compression_ratio),
                attention_kwargs=None, return_dict=False,
            )[0]
        squared = (prediction.float() - target.float()).square()
        spatial = pipe._unpack_latents(squared, frames, height, width, spatial_patch, temporal_patch)
        spatial = spatial.float().mean(dim=1).mean(dim=1)
        return squared.mean(dim=(1, 2))[0], spatial
