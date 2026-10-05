"""FLUX.1 dev/schnell: VAE mode and native flow MSE."""

import numpy as np
import torch
from diffusers import FluxImg2ImgPipeline
from diffusers.pipelines.flux.pipeline_flux_img2img import calculate_shift

from .common import collect_losses, image_mse, image_pixels, noise_bank


class Flux1:
    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        self.config = config
        self.protocol = config["presets"][preset]
        checkpoint = config["checkpoint"]
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], torch_dtype=torch.bfloat16,
        ).to(self.device)
        torch.set_float32_matmul_precision("high")
        with torch.no_grad():
            self.prompt, self.pooled, self.text_ids = self.pipe.encode_prompt(
                prompt=config.get("prompt", ""), prompt_2=config.get("prompt", ""),
                device=self.device, max_sequence_length=512,
            )
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        self.guidance = (torch.full((1,), config["guidance"], device=self.device)
                         if self.pipe.transformer.config.guidance_embeds else None)
        self.image_ids = self.pipe._prepare_latent_image_ids(1, 32, 32, self.device, torch.bfloat16)
        self.noise = noise_bank((1, 16, 64, 64), self.protocol["noise_draws"],
                                config["noise_seed"], self.device, torch.bfloat16)

    @torch.no_grad()
    def encode_image(self, image):
        pipe = self.pipe
        pixels = pipe.image_processor.preprocess(image_pixels(image, self.device), height=512, width=512)
        latent = pipe.vae.encode(pixels.to(torch.bfloat16)).latent_dist.mode()
        return (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    @torch.no_grad()
    def score_image(self, image):
        pipe = self.pipe
        clean = self.encode_image(image)
        steps = self.config["evaluation_steps"]
        scheduler_config = pipe.scheduler.config
        shift = calculate_shift(32 * 32, scheduler_config.base_image_seq_len,
                                scheduler_config.max_image_seq_len,
                                scheduler_config.base_shift, scheduler_config.max_shift)
        pipe.scheduler.set_timesteps(steps, device=self.device,
                                     sigmas=np.linspace(1.0, 1 / steps, steps), mu=shift)
        timesteps, _ = pipe.get_timesteps(steps, 1.0, self.device)

        def loss_at(index, noise):
            timestep = timesteps[index]
            pipe.scheduler._step_index = index
            noisy = pipe.scheduler.scale_noise(clean, timestep.reshape(1), noise)
            packed = pipe._pack_latents(noisy, 1, 16, 64, 64)
            prediction = pipe.transformer(
                hidden_states=packed, timestep=(timestep / 1000).to(packed.dtype).expand(1),
                guidance=self.guidance, pooled_projections=self.pooled,
                encoder_hidden_states=self.prompt, txt_ids=self.text_ids, img_ids=self.image_ids,
                return_dict=False,
            )[0]
            prediction = pipe._unpack_latents(prediction, 512, 512, pipe.vae_scale_factor)
            return image_mse((noise.float() - clean.float()) - prediction.float())

        return collect_losses(timesteps, self.protocol["timestep_indices"], self.noise, loss_at)
