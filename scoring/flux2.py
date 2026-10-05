"""The four FLUX.2 klein checkpoints share this single-pass flow calculation."""

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

from .common import collect_losses, image_pixels, noise_bank


class Flux2:
    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        self.config = config
        self.protocol = config["presets"][preset]
        checkpoint = config["checkpoint"]
        self.pipe = Flux2KleinPipeline.from_pretrained(
            checkpoint["repo_id"], revision=checkpoint["revision"], torch_dtype=torch.bfloat16,
        )
        # Encode text on CPU in float32.
        self.pipe.text_encoder.to(device="cpu", dtype=torch.float32).eval()
        self.pipe.transformer.to(self.device)
        self.pipe.vae.to(self.device)
        torch.set_float32_matmul_precision("high")
        with torch.no_grad():
            prompt, text_ids = self.pipe.encode_prompt(prompt="", device="cpu", max_sequence_length=512)
        self.prompt = prompt.to(self.device, torch.bfloat16)
        self.text_ids = text_ids.to(self.device)
        self.noise = noise_bank((1, 128, 32, 32), self.protocol["noise_draws"],
                                config["noise_seed"], self.device, torch.bfloat16)

    @torch.no_grad()
    def score_image(self, image):
        pipe = self.pipe
        pixels = pipe.image_processor.preprocess(image_pixels(image, self.device), height=512, width=512)
        # The helper uses the VAE mode, patchifies 2x2, and normalizes using VAE BN statistics.
        clean = pipe._encode_vae_image(pixels.to(self.device, torch.bfloat16), generator=None)
        image_ids = pipe._prepare_latent_ids(clean).to(self.device)
        steps = self.config["evaluation_steps"]
        sigmas = np.linspace(1.0, 1 / steps, steps, dtype=np.float32)
        if pipe.scheduler.config.get("use_flow_sigmas", False):
            sigmas = None
        shift = compute_empirical_mu(image_seq_len=pipe._pack_latents(clean).shape[1], num_steps=steps)
        pipe.scheduler.set_timesteps(steps, device=self.device, sigmas=sigmas, mu=shift)
        # There is no img2img begin index: each requested time determines its own sigma.
        pipe.scheduler._begin_index = None
        pipe.scheduler._step_index = None
        timesteps = pipe.scheduler.timesteps

        def loss_at(index, noise):
            timestep = timesteps[index].expand(1)
            noisy = pipe.scheduler.scale_noise(clean, timestep, noise)
            packed = pipe._pack_latents(noisy)
            target = pipe._pack_latents(noise.float() - clean.float())
            prediction = pipe.transformer(
                hidden_states=packed, timestep=(timestep.float() / 1000).to(packed.dtype),
                encoder_hidden_states=self.prompt, txt_ids=self.text_ids, img_ids=image_ids,
                return_dict=False,
            )[0]
            squared = (prediction.float() - target.float()).square()
            spatial = pipe._unpack_latents_with_ids(squared, image_ids).mean(dim=1)
            return spatial.mean(dim=(1, 2))[0], spatial

        return collect_losses(timesteps, self.protocol["timestep_indices"], self.noise, loss_at)
