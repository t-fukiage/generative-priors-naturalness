"""HiDream-O1-Image Full: pixel-patch x0 predictions converted to flow error."""

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from models.utils import get_rope_index_fix_point

from .common import collect_losses, image_mse, noise_bank, pixel_tensor


def patchify(tensor, patch_size=32):
    batch, channels, height, width = tensor.shape
    return (tensor.reshape(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, (height // patch_size) * (width // patch_size), channels * patch_size**2))


def unpatchify(patches, patch_size=32):
    batch, _, patch_dim = patches.shape
    channels = patch_dim // patch_size**2
    return (patches.reshape(batch, 512 // patch_size, 512 // patch_size, channels, patch_size, patch_size)
            .permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, 512, 512))


class HiDreamO1:
    def __init__(self, config, preset, device="cuda"):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.config = config
        self.protocol = config["presets"][preset]
        checkpoint = config["checkpoint"]
        checkpoint_path = snapshot_download(repo_id=checkpoint["repo_id"], revision=checkpoint["revision"])
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16, device_map=str(self.device),
        ).eval()
        self.sample = self.prepare_prompt()
        self.scheduler = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=config["shift"])
        self.scheduler.set_timesteps(config["evaluation_steps"], device=self.device)
        self.noise = noise_bank((1, 3, 512, 512), self.protocol["noise_draws"],
                                config["noise_seed"], self.device, torch.bfloat16)

    def prepare_prompt(self):
        """Official image/text token layout for the single blank-space prompt."""
        processor, model_config = self.processor, self.model.config
        tokenizer = processor.tokenizer
        messages = [{"role": "user", "content": self.config["prompt"]}]
        caption = (processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                   + "<|boi_token|>" + "<|tms_token|>")
        input_ids = tokenizer.encode(caption, return_tensors="pt", add_special_tokens=False)
        grid = 512 // self.config["patch_size"]
        image_length = grid * grid
        image_grid = torch.tensor([[1, grid, grid]], dtype=torch.int64)
        vision_tokens = torch.zeros((1, image_length), dtype=input_ids.dtype) + model_config.image_token_id
        vision_tokens[0, 0] = model_config.vision_start_token_id
        padded_ids = torch.cat([input_ids, vision_tokens], dim=-1)
        positions, _ = get_rope_index_fix_point(
            1, model_config.image_token_id, model_config.video_token_id, model_config.vision_start_token_id,
            input_ids=padded_ids, image_grid_thw=image_grid, video_grid_thw=None,
            attention_mask=None, skip_vision_start_token=[1],
        )
        text_length = input_ids.shape[-1]
        token_types = torch.zeros((1, positions.shape[-1]), dtype=input_ids.dtype)
        token_types[0, text_length - 1:text_length + image_length] = 1
        token_types[0, text_length - 1:text_length] = 3
        return {"input_ids": input_ids.to(self.device), "position_ids": positions.to(self.device),
                "token_types": (token_types > 0).to(input_ids.dtype).to(self.device),
                "vinput_mask": (token_types == 1).to(self.device)}

    @torch.no_grad()
    def score_image(self, image):
        clean = pixel_tensor(image, self.device, torch.bfloat16)
        timesteps = self.scheduler.timesteps

        def loss_at(index, noise):
            timestep = timesteps[index].float()
            sigma = (timestep / 1000.0).clamp_min(self.config["t_eps"])
            sigma_image = sigma.to(torch.bfloat16).view(1, 1, 1, 1)
            scaled_noise = noise * self.config["noise_scale"]
            noisy = sigma_image * scaled_noise + (1.0 - sigma_image) * clean
            patches = patchify(noisy, self.config["patch_size"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model(
                    input_ids=self.sample["input_ids"], position_ids=self.sample["position_ids"],
                    vinputs=patches, timestep=(1.0 - timestep / 1000.0).reshape(-1),
                    token_types=self.sample["token_types"], use_flash_attn=False,
                )
            predicted_x = output.x_pred[0, self.sample["vinput_mask"][0]].unsqueeze(0).to(torch.bfloat16)
            predicted_flow = (patches.float() - predicted_x.float()) / sigma.float().clamp_min(self.config["t_eps"])
            predicted_flow = unpatchify(predicted_flow, self.config["patch_size"])
            target_flow = scaled_noise.float() - clean.float()
            return image_mse(target_flow - predicted_flow)

        return collect_losses(timesteps, self.protocol["timestep_indices"], self.noise, loss_at)
