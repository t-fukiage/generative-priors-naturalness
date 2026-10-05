"""SD3 Medium: shifted/scaled VAE mode and native flow-prediction MSE."""

import torch
from diffusers import StableDiffusion3Img2ImgPipeline

from .common import collect_losses, image_mse, image_pixels, noise_bank


class StableDiffusion3:
    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        self.config = config
        self.protocol = config["presets"][preset]
        checkpoint = config["checkpoint"]
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"],
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        if self.pipe.text_encoder_3 is None or self.pipe.tokenizer_3 is None:
            raise ValueError("SD3 scoring requires both CLIP encoders and T5.")
        torch.set_float32_matmul_precision("high")
        with torch.no_grad():
            self.prompt, _, self.pooled, _ = self.pipe.encode_prompt(
                prompt=config.get("prompt", ""), prompt_2=None, prompt_3=None,
                do_classifier_free_guidance=False,
                device=self.device, max_sequence_length=256,
            )
        # All three encoders contributed; only their embeddings are needed now.
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            getattr(self.pipe, name).to("cpu")
        torch.cuda.empty_cache()
        self.noise = noise_bank((1, 16, 64, 64), self.protocol["noise_draws"],
                                config["noise_seed"], self.device, torch.bfloat16)

    @torch.no_grad()
    def encode_image(self, image):
        pipe = self.pipe
        pixels = pipe.image_processor.preprocess(image_pixels(image, self.device)).to(torch.bfloat16)
        latent = pipe.vae.encode(pixels).latent_dist.mode()
        return (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    @torch.no_grad()
    def score_image(self, image):
        pipe = self.pipe
        clean = self.encode_image(image)
        pipe.scheduler.set_timesteps(self.config["evaluation_steps"], device=self.device)
        timesteps, _ = pipe.get_timesteps(self.config["evaluation_steps"], 1.0, self.device)

        def loss_at(index, noise):
            timestep = timesteps[index]
            # Use the position on the full scheduler grid.
            pipe.scheduler._step_index = index
            noisy = pipe.scheduler.scale_noise(clean, timestep.reshape(1), noise)
            prediction = pipe.transformer(
                hidden_states=noisy, timestep=timestep.expand(1),
                encoder_hidden_states=self.prompt, pooled_projections=self.pooled,
                return_dict=False,
            )[0]
            return image_mse((noise.float() - clean.float()) - prediction.float())

        return collect_losses(timesteps, self.protocol["timestep_indices"], self.noise, loss_at)
