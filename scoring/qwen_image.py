"""Qwen-Image / Qwen-Image-2512: flow loss in packed VAE space, empty by default."""

import numpy as np
import torch
from diffusers import QwenImageImg2ImgPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_img2img import calculate_shift

from .common import collect_losses, image_mse, image_pixels, noise_bank


class QwenImage:
    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        self.config = config
        self.protocol = config["presets"][preset]
        checkpoint = config["checkpoint"]
        self.pipe = QwenImageImg2ImgPipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], torch_dtype=torch.bfloat16,
        )
        self.pipe.text_encoder.to(device="cpu", dtype=torch.float32).eval()
        self.pipe.transformer.to(self.device)
        self.pipe.vae.to(self.device)
        with torch.no_grad():
            prompt, mask = self.pipe.encode_prompt(
                prompt=config.get("prompt", ""), device="cpu", max_sequence_length=512,
            )
        self.prompt = prompt.to(self.device, torch.bfloat16)
        self.mask = mask.to(self.device) if mask is not None else None
        self.text_lengths = self.mask.sum(dim=1).tolist() if mask is not None else None
        self.noise = noise_bank((1, 16, 64, 64), self.protocol["noise_draws"],
                                config["noise_seed"], self.device, torch.bfloat16)

    @torch.no_grad()
    def encode_image(self, image):
        pipe = self.pipe
        pixels = pipe.image_processor.preprocess(image_pixels(image, self.device), height=512, width=512)
        pixels = pixels.float().unsqueeze(2).to(self.device, self.prompt.dtype)
        latent = pipe.vae.encode(pixels).latent_dist.mode()
        mean = torch.tensor(pipe.vae.config.latents_mean, device=self.device, dtype=latent.dtype)
        inverse_std = 1.0 / torch.tensor(pipe.vae.config.latents_std, device=self.device, dtype=latent.dtype)
        clean = (latent - mean.view(1, 16, 1, 1, 1)) * inverse_std.view(1, 16, 1, 1, 1)
        return clean.transpose(1, 2)  # [1, 1, 16, 64, 64]

    @torch.no_grad()
    def score_image(self, image):
        pipe = self.pipe
        clean = self.encode_image(image)
        steps = self.config["evaluation_steps"]
        scheduler_config = pipe.scheduler.config
        shift = calculate_shift(32 * 32, scheduler_config.get("base_image_seq_len", 256),
                                scheduler_config.get("max_image_seq_len", 4096),
                                scheduler_config.get("base_shift", 0.5),
                                scheduler_config.get("max_shift", 1.15))
        pipe.scheduler.set_timesteps(steps, device=self.device,
                                     sigmas=np.linspace(1.0, 1 / steps, steps), mu=shift)
        timesteps = pipe.scheduler.timesteps

        def loss_at(index, noise):
            timestep = timesteps[index].expand(1)
            noisy = pipe.scheduler.scale_noise(clean, timestep, noise.unsqueeze(1))
            packed = pipe._pack_latents(noisy, 1, 16, 64, 64)
            target = pipe._pack_latents(noise.unsqueeze(1).float() - clean.float(), 1, 16, 64, 64)
            with pipe.transformer.cache_context("cond"):
                prediction = pipe.transformer(
                    hidden_states=packed, timestep=timestep.to(self.prompt.dtype) / 1000,
                    guidance=None, encoder_hidden_states_mask=self.mask,
                    encoder_hidden_states=self.prompt, img_shapes=[[(1, 32, 32)]],
                    txt_seq_lens=self.text_lengths, attention_kwargs={}, return_dict=False,
                )[0]
            prediction = pipe._unpack_latents(prediction, 512, 512, pipe.vae_scale_factor)[:, :, 0]
            target = pipe._unpack_latents(target, 512, 512, pipe.vae_scale_factor)[:, :, 0]
            return image_mse(target.float() - prediction.float())

        return collect_losses(timesteps, self.protocol["timestep_indices"], self.noise, loss_at)
