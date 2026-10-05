"""Stable Diffusion v1.5: image -> latent -> noise-prediction loss.

This is an input-scoring procedure, not an image-generation sampler. Each
timestep starts from the same clean latent; no scheduler step is taken.
"""

from PIL import Image
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .common import ImageLoss, collect_losses, image_pixels


def image_tensor(image: Image.Image) -> torch.Tensor:
    """RGB float32, bilinear 512 resize, [1,3,512,512] in [0,1]."""
    return image_pixels(image, "cpu")


class StableDiffusion15:
    """Epsilon MSE with shared noise and the paper's latent path.

    Checkpoint revisions and the evaluation grid come from the supplied config.
    A single noise bank is reused across every image and every timestep.
    The default prompt is empty; text controls can supply config["prompt"].
    """

    def __init__(self, config: dict, preset: str, device: str = "cuda"):
        self.device = torch.device(device)
        protocol = config["presets"][preset]
        indices = protocol["timestep_indices"]
        if not indices or indices != sorted(set(indices)):
            raise ValueError("Timestep indices must be nonempty, unique, and increasing.")
        if any(type(index) is not int or not 0 <= index < 999 for index in indices):
            raise ValueError("Indices must be integers in [0, 998]; the old path omits 999.")
        draws = protocol["noise_draws"]
        if type(draws) is not int or draws < 1:
            raise ValueError("noise_draws must be a positive integer.")
        self.indices = indices

        model = config["diffusion_checkpoint"]
        text = config["text_checkpoint"]
        self.vae = AutoencoderKL.from_pretrained(
            model["repo_id"], subfolder="vae", revision=model["revision"],
            torch_dtype=torch.float32,
        ).to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(
            model["repo_id"], subfolder="unet", revision=model["revision"],
            torch_dtype=torch.float32,
        ).to(self.device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(
            text["repo_id"], revision=text["revision"],
        )
        text_encoder = CLIPTextModel.from_pretrained(
            text["repo_id"], revision=text["revision"],
        ).to(device=self.device, dtype=torch.float32).eval()
        tokens = tokenizer(
            [config.get("prompt", "")], padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            self.prompt = text_encoder(tokens.input_ids.to(self.device))[0]
        # Only the embedding is needed during scoring.
        del text_encoder

        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        )
        self.scheduler.set_timesteps(1000)

        generator = torch.Generator(device="cpu").manual_seed(config["noise_seed"])
        self.noise = [
            torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32)
            .to(self.device)
            for _ in range(draws)
        ]

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Use the posterior mode, scaled by 0.18215; do not sample the VAE."""
        pixels = image_tensor(image).to(self.device) * 2 - 1
        return 0.18215 * self.vae.encode(pixels).latent_dist.mode()

    @torch.no_grad()
    def score_image(self, image: Image.Image) -> ImageLoss:
        clean_latent = self.encode_image(image)

        def loss_at(index, noise):
            timestep = self.scheduler.timesteps[index]
            alpha_bar = self.scheduler.alphas_cumprod[timestep.round().long()]
            # Convert LMS sigma to the noise standard deviation of the VP path.
            noise_std = self.scheduler.sigmas[index] * alpha_bar.sqrt()
            noise_variance = noise_std.square()
            noisy_latent = (
                (1.0 - noise_variance).sqrt() * clean_latent
                + noise_variance.sqrt() * noise
            )
            prediction = self.unet(
                noisy_latent, timestep, encoder_hidden_states=self.prompt,
            ).sample
            loss_map = (noise - prediction).square().mean(dim=1)
            return loss_map.mean(dim=(1, 2))[0], loss_map

        return collect_losses(
            self.scheduler.timesteps, self.indices, self.noise, loss_at,
        )
