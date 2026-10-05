"""Select a model family without importing the other families' dependencies."""


def load_encoder(name, config, device="cuda"):
    if config.get("input_resize", "bilinear") != "bilinear":
        raise ValueError("Encoder inputs use the fixed bilinear protocol.")
    if name.startswith("qwen"):
        from .qwen_vl import load
    elif name == "jina_clip_v2":
        from .jina_clip import load
    else:
        from . import vision
        if name.startswith("resnet"):
            load = vision.resnet
        elif name.startswith("clip"):
            load = vision.clip
        elif name.startswith("dino"):
            load = vision.dino
        elif name.startswith("siglip"):
            load = vision.siglip
        elif name.startswith("pe_core"):
            load = vision.pe
        else:
            raise ValueError(f"Unknown encoder: {name}")
    return load(config, device)
