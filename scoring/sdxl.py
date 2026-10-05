"""SDXL Base: VAE mode and epsilon MSE; negative empty branch by default."""

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline

from .common import collect_losses, image_mse, image_pixels, noise_bank


class SDXL:
    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        self.config = config
        self.protocol = config["presets"][preset]
        checkpoint = config["checkpoint"]
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], subfolder="scheduler",
            solver_order=2, algorithm_type="dpmsolver++",
        )
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"],
            scheduler=scheduler, torch_dtype=torch.float16,
        ).to(self.device)
        with torch.no_grad():
            text = config.get("prompt", "")
            positive, negative, positive_pooled, negative_pooled = self.pipe.encode_prompt(
                prompt=text, prompt_2=text, negative_prompt="", negative_prompt_2="",
                do_classifier_free_guidance=True, device=self.device,
            )
            # Preserve the main empty branch; the text control uses one
            # conditional forward, without classifier-free guidance mixing.
            self.prompt = positive if text else negative
            self.pooled = positive_pooled if text else negative_pooled
        _, negative_time_ids = self.pipe._get_add_time_ids(
            original_size=(512, 512), crops_coords_top_left=(0, 0), target_size=(512, 512),
            aesthetic_score=6.0, negative_aesthetic_score=2.5,
            negative_original_size=(512, 512), negative_crops_coords_top_left=(0, 0),
            negative_target_size=(512, 512), dtype=self.prompt.dtype,
            text_encoder_projection_dim=self.pooled.shape[-1],
        )
        self.time_ids = negative_time_ids.to(self.device)
        self.noise = noise_bank((1, 4, 64, 64), self.protocol["noise_draws"],
                                config["noise_seed"], self.device, torch.float16)

    @torch.no_grad()
    def encode_image(self, image):
        pipe = self.pipe
        pixels = pipe.image_processor.preprocess(image_pixels(image, self.device))
        pixels = pixels.to(self.device, self.prompt.dtype)
        # Keep the native helper's input cast and float32 VAE upcast. The pinned
        # base VAE has no latent mean/std transform, only a scaling factor.
        if pipe.vae.config.latents_mean is not None or pipe.vae.config.latents_std is not None:
            raise ValueError("This scorer expects the pinned SDXL Base VAE.")
        original_dtype = pipe.vae.dtype
        if pipe.vae.config.force_upcast:
            pipe.vae.to(dtype=torch.float32)
            pixels = pixels.float()
        try:
            latent = pipe.vae.encode(pixels).latent_dist.mode()
        finally:
            pipe.vae.to(dtype=original_dtype)
        return pipe.vae.config.scaling_factor * latent.to(self.prompt.dtype)

    @torch.no_grad()
    def score_image(self, image):
        pipe = self.pipe
        pipe.scheduler.set_timesteps(self.config["evaluation_steps"], device=self.device)
        timesteps, _ = pipe.get_timesteps(self.config["evaluation_steps"], 1.0, self.device)
        clean = self.encode_image(image)

        def loss_at(index, noise):
            timestep = timesteps[index]
            pipe.scheduler._step_index = index
            noisy = pipe.scheduler.add_noise(clean, noise.to(clean.dtype), timestep.reshape(1))
            prediction = pipe.unet(
                pipe.scheduler.scale_model_input(noisy, timestep), timestep,
                encoder_hidden_states=self.prompt,
                added_cond_kwargs={"text_embeds": self.pooled, "time_ids": self.time_ids},
                return_dict=False,
            )[0]
            return image_mse(noise.float() - prediction.float())

        return collect_losses(timesteps, self.protocol["timestep_indices"], self.noise, loss_at)
