"""Wan 2.1 (1.3B/14B) and 2.2 A14B: static-video native flow MSE."""

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

from .video_common import StaticVideoScore, video_mse


class Wan(StaticVideoScore):
    def load_pipeline(self):
        checkpoint = self.config["checkpoint"]
        vae = AutoencoderKLWan.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipe = WanPipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], vae=vae, torch_dtype=self.dtype,
        )
        # Tie UMT5's encoder table to the checkpoint's shared embeddings. This
        # also avoids the separate uninitialized table in Transformers 5.2.
        embeddings = pipe.text_encoder.get_input_embeddings()
        pipe.text_encoder.set_input_embeddings(embeddings)
        if pipe.text_encoder.encoder.get_input_embeddings().weight.data_ptr() != embeddings.weight.data_ptr():
            raise ValueError("Wan input embedding weights are not tied.")
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, prediction_type="flow_prediction",
            use_flow_sigmas=True, flow_shift=self.config["flow_shift"],
        )
        return pipe

    def encode_prompt(self):
        self.prompt, _ = self.pipe.encode_prompt(
            prompt="", negative_prompt="", do_classifier_free_guidance=False,
            max_sequence_length=512, device=self.device,
        )
        self.prompt = self.prompt.to(self.dtype)

    def encode_video(self, frame):
        video = frame.float().unsqueeze(2).expand(-1, -1, self.config["frames"], -1, -1)
        latent = self.pipe.vae.encode(video).latent_dist.mode().float()
        cfg = self.pipe.vae.config
        mean = torch.tensor(cfg.latents_mean, device=self.device).view(1, cfg.z_dim, 1, 1, 1)
        inverse_std = 1.0 / torch.tensor(cfg.latents_std, device=self.device).view(1, cfg.z_dim, 1, 1, 1)
        return (latent - mean) * inverse_std

    def evaluation_timesteps(self, clean):
        self.pipe.scheduler.set_timesteps(self.config["evaluation_steps"], device=self.device)
        return self.pipe.scheduler.timesteps

    def loss(self, clean, noise, timestep):
        pipe = self.pipe
        noisy = self.add_noise(clean, noise, timestep)
        denoiser = pipe.transformer
        boundary = pipe.config.boundary_ratio
        if boundary is not None and timestep.item() < boundary * pipe.scheduler.config.num_train_timesteps:
            denoiser = pipe.transformer_2
        # All three T2V configurations use one time per clip.
        if pipe.config.expand_timesteps:
            raise ValueError("This release requires one timestep per T2V clip.")
        with denoiser.cache_context("cond"):
            prediction = denoiser(
                hidden_states=noisy.to(self.dtype), timestep=timestep.expand(1),
                encoder_hidden_states=self.prompt, attention_kwargs=None, return_dict=False,
            )[0]
        return video_mse((prediction.float() - (noise.float() - clean.float())).square())
