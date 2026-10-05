"""ResNet, CLIP, DINO, SigLIP and PE at 512-pixel input size."""

import torch
import torch.nn.functional as F

from .common import LayerEncoder, normalize_pixels, normalized


def resnet(config, device):
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval().to(device)
    model.fc = torch.nn.Identity()
    preprocess = weights.transforms(resize_size=512, crop_size=512)
    blocks = [block for stage in range(1, 5) for block in getattr(model, f"layer{stage}")]
    def encode(images):
        return model(torch.stack([preprocess(image) for image in images]).to(device))
    return LayerEncoder(encode, blocks, lambda x: x.mean(dim=(-2, -1)), model)


def clip(config, device):
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

    settings = {"revision": config["revision"]}
    preprocess = AutoImageProcessor.from_pretrained(config["backend_name"], **settings)
    model = CLIPVisionModelWithProjection.from_pretrained(config["backend_name"], **settings).eval().to(device)
    def encode(images):
        inputs = preprocess(images=images, size={"height": 512, "width": 512},
                            do_center_crop=False, do_resize=False, do_rescale=False,
                            return_tensors="pt").to(device)
        return model(**inputs, interpolate_pos_encoding=True).image_embeds
    # Raw CLS is the intermediate readout; the endpoint keeps projection.
    return LayerEncoder(encode, list(model.vision_model.encoder.layers), lambda x: x[:, 0], model)


def dino(config, device):
    from transformers import AutoImageProcessor, AutoModel

    dtype = getattr(torch, config["dtype"])
    settings = {"revision": config["revision"]}
    preprocess = AutoImageProcessor.from_pretrained(config["backend_name"], **settings)
    model = AutoModel.from_pretrained(config["backend_name"], torch_dtype=dtype, **settings).eval().to(device)
    dino3 = "dinov3" in config["backend_name"]
    blocks = list(model.layer if dino3 else model.encoder.layer)
    norm = model.norm if dino3 else model.layernorm
    first_patch = 1 + config["expected_register_tokens"]
    def encode(images):
        inputs = preprocess(images=images, size={"height": 512, "width": 512},
                            do_center_crop=False, do_resize=False, do_rescale=False,
                            return_tensors="pt")
        inputs = {key: value.to(device, dtype) for key, value in inputs.items()}
        # Match PE-G's dynamic patch padding: zero after normalization, on the
        # right/bottom only. All 512 pixels survive DINOv2's 14-pixel projection.
        pixels = inputs["pixel_values"]
        patch = int(model.config.patch_size)
        height, width = pixels.shape[-2:]
        inputs["pixel_values"] = F.pad(pixels, (0, -width % patch, 0, -height % patch))
        output = model(**inputs).last_hidden_state[:, first_patch:]
        # DINOv2 has 37x37 patches; DINOv3 needs no padding and keeps 32x32.
        expected = ((height + patch - 1) // patch) * ((width + patch - 1) // patch)
        if output.shape[1] != expected:
            raise ValueError("Unexpected DINO spatial patch grid.")
        return output
    return LayerEncoder(encode, blocks, lambda x: norm(x)[:, first_patch:], model)


def siglip(config, device):
    from transformers import AutoModel, AutoProcessor

    settings = {"revision": config["revision"]}
    preprocess = AutoProcessor.from_pretrained(config["backend_name"], use_fast=False, **settings)
    model = AutoModel.from_pretrained(config["backend_name"], torch_dtype=torch.float32, **settings).eval().to(device)
    tower = model.vision_model
    def encode(images):
        inputs = preprocess(images=images, size={"height": 512, "width": 512},
                            do_resize=False, do_rescale=False, return_tensors="pt").to(device)
        output = model.get_image_features(**inputs, interpolate_pos_encoding=True)
        return normalized(output.pooler_output if hasattr(output, "pooler_output") else output)
    # Apply the usual terminal pooling head separately to each intermediate block.
    return LayerEncoder(encode, list(tower.encoder.layers), lambda x: tower.head(tower.post_layernorm(x)), model)


def pe(config, device):
    import timm
    from huggingface_hub import hf_hub_download

    checkpoint = hf_hub_download(config["backend_name"], "model.safetensors", revision=config["revision"])
    model = timm.create_model(config["timm_model_name"], pretrained=False, checkpoint_path=checkpoint,
                             dynamic_img_size=True, dynamic_img_pad=True)
    dtype = getattr(torch, config["dtype"])
    model = model.eval().to(device, dtype)
    settings = timm.data.resolve_model_data_config(model)
    def encode(images):
        pixels = normalize_pixels(images, settings["mean"], settings["std"])
        return model(pixels.to(device, dtype))
    # Intermediate readout is plain patch mean; the endpoint keeps learned MAP pooling.
    return LayerEncoder(encode, list(model.blocks), lambda x: x[:, model.num_prefix_tokens:].mean(1), model)
