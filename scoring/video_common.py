"""The shared static-clip protocol for the nine video checkpoints."""

import torch
import torch.nn.functional as F

from .common import collect_losses, noise_bank, rgb_tensor


def fit_pad_frame(image, height, width, device):
    """Bilinear aspect-preserving resize, centered gray padding, then [-1,1]."""
    frame = rgb_tensor(image, device)
    source_h, source_w = frame.shape[-2:]
    scale = min(height / source_h, width / source_w)
    resized_h = max(1, min(height, round(source_h * scale)))
    resized_w = max(1, min(width, round(source_w * scale)))
    if (source_h, source_w) != (resized_h, resized_w):
        frame = F.interpolate(frame, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    pad_h, pad_w = height - resized_h, width - resized_w
    frame = F.pad(frame, (pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2), value=0.5)
    return (frame - 0.5) / 0.5


def video_mse(squared, channel_dim=1):
    """Scalar reduces all non-batch axes; maps average channels and then frames."""
    scalar = squared.mean(dim=tuple(range(1, squared.ndim)))[0]
    spatial = squared.mean(dim=channel_dim).mean(dim=1)
    return scalar, spatial


class StaticVideoScore:
    """Common execution order; each family defines its own encoders and loss.

    VAE tiling and slicing are fixed per checkpoint because they can change
    encoded latents. CPU offload changes placement, never precision or geometry.
    """

    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.config = config
        self.protocol = config["presets"][preset]
        self.pipe = self.load_pipeline()
        if config["vae_tiling"]:
            self.pipe.vae.enable_tiling(**config.get("vae_tile_settings", {}))
        if config["vae_slicing"]:
            self.pipe.vae.enable_slicing()
        if config["model_cpu_offload"]:
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)
        with torch.inference_mode():
            self.encode_prompt()
        if not config["model_cpu_offload"]:
            for name in ("text_encoder", "text_encoder_2"):
                encoder = getattr(self.pipe, name, None)
                if encoder is not None:
                    encoder.to("cpu")
            torch.cuda.empty_cache()

    def clip(self, frame):
        return frame.to(self.dtype).unsqueeze(2).repeat(1, 1, self.config["frames"], 1, 1)

    def add_noise(self, clean, noise, timestep):
        scheduler = self.pipe.scheduler
        if hasattr(scheduler, "add_noise"):
            return scheduler.add_noise(clean, noise, timestep.expand(1))
        return scheduler.scale_noise(clean, timestep.expand(1), noise)

    @torch.inference_mode()
    def score_image(self, image):
        config = self.config
        frame = fit_pad_frame(image, config["height"], config["width"], self.device)
        if not config["model_cpu_offload"]:
            self.pipe.vae.to(self.device)
        clean = self.encode_video(frame)
        if not config["model_cpu_offload"]:
            self.pipe.vae.to("cpu")
            torch.cuda.empty_cache()
        # Video noise remains float32 even when the denoiser uses bfloat16.
        noise = noise_bank(tuple(clean.shape), self.protocol["noise_draws"],
                           config["noise_seed"], "cpu", torch.float32)
        timesteps = self.evaluation_timesteps(clean)

        def loss_at(index, draw):
            return self.loss(clean, draw.to(self.device), timesteps[index])

        return collect_losses(timesteps, self.protocol["timestep_indices"], noise, loss_at)
