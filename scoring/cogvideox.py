"""CogVideoX 2B, 5B and 1.5-5B: VAE mode and v-prediction MSE."""

import torch
from diffusers import CogVideoXPipeline

from .video_common import StaticVideoScore, video_mse


class CogVideoX(StaticVideoScore):
    def load_pipeline(self):
        checkpoint = self.config["checkpoint"]
        pipe = CogVideoXPipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], torch_dtype=self.dtype,
        )
        if pipe.scheduler.config.prediction_type != "v_prediction":
            raise ValueError("These checkpoints require the v-prediction scheduler.")
        return pipe

    def encode_prompt(self):
        self.prompt, _ = self.pipe.encode_prompt(
            prompt="", negative_prompt="", do_classifier_free_guidance=False,
            max_sequence_length=226, device=self.device,
        )
        self.rotary = None
        if self.pipe.transformer.config.use_rotary_positional_embeddings:
            frames = (self.config["frames"] - 1) // self.pipe.vae_scale_factor_temporal + 1
            self.rotary = self.pipe._prepare_rotary_positional_embeddings(
                self.config["height"], self.config["width"], frames, self.device,
            )

    def encode_video(self, frame):
        # The 1.5 checkpoint expands a requested 17 frames to 21 so latent frame
        # count is divisible by its temporal patch size. The config records 21.
        video = frame.to(self.dtype).unsqueeze(2).expand(-1, -1, self.config["frames"], -1, -1)
        latent = self.pipe.vae.encode(video).latent_dist.mode().float()
        return latent.permute(0, 2, 1, 3, 4) * float(self.pipe.vae_scaling_factor_image)

    def evaluation_timesteps(self, clean):
        self.pipe.scheduler.set_timesteps(self.config["evaluation_steps"], device=self.device)
        return self.pipe.scheduler.timesteps

    def loss(self, clean, noise, timestep):
        pipe = self.pipe
        noisy = self.add_noise(clean, noise, timestep).to(self.dtype)
        model_input = pipe.scheduler.scale_model_input(noisy, timestep).to(self.dtype)
        with pipe.transformer.cache_context("cond_uncond"):
            prediction = pipe.transformer(
                hidden_states=model_input, encoder_hidden_states=self.prompt.to(self.dtype),
                timestep=timestep.expand(1).to(self.dtype), image_rotary_emb=self.rotary,
                attention_kwargs=None, return_dict=False,
            )[0]
        target = pipe.scheduler.get_velocity(clean, noise, timestep.expand(1))
        return video_mse((target.float() - prediction.float()).square(), channel_dim=2)
