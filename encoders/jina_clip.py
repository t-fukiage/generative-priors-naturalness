"""Jina CLIP v2: the pinned official vision tower and its full 1024-D output."""

import importlib.util
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .common import LayerEncoder, normalize_pixels, normalized


def load(config, device):
    checkpoint = Path(snapshot_download(config["backend_name"], revision=config["revision"]))
    implementation = snapshot_download(config["implementation_repo"], revision=config["implementation_revision"])
    config_class = get_class_from_dynamic_module("configuration_clip.JinaCLIPVisionConfig", implementation)
    model_class = get_class_from_dynamic_module("modeling_clip.JinaCLIPVisionModel", implementation)
    processor_class = get_class_from_dynamic_module("processing_clip.JinaCLIPImageProcessor", implementation)
    settings = json.loads((checkpoint / "config.json").read_text())
    if settings["add_projections"]:
        raise ValueError("The Jina checkpoint has no extra visual projection.")
    vision_config = config_class(**settings["vision_config"])
    if vision_config.x_attention and importlib.util.find_spec("xformers") is None:
        vision_config.x_attention = False  # Official model's standard-attention fallback.
    model = model_class(vision_config).to(dtype=torch.bfloat16)
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as weights:
        state = {key: weights.get_tensor(key) for key in weights.keys() if key.startswith("vision_model.")}
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    processor = processor_class.from_pretrained(checkpoint)
    tower = model.vision_model
    def encode(images):
        if processor.size not in (512, [512, 512], (512, 512)):
            raise ValueError("The Jina processor must retain 512-pixel inputs.")
        pixels = normalize_pixels(images, processor.mean, processor.std)
        return normalized(model(pixel_values=pixels.to(device, torch.bfloat16)).image_embeds)
    def readout(hidden):
        hidden = tower.norm(hidden)
        pooled = tower.fc_norm(hidden.mean(1)) if tower.fc_norm is not None else hidden[:, 0]
        return tower.head(pooled)
    return LayerEncoder(encode, list(tower.blocks), readout, model)
