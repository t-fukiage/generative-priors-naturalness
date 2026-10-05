"""Public model names and their readable implementation files.

Imports are lazy so one model does not require every other model's dependencies.
"""

from importlib import import_module

MODEL_CLASSES = {
    'flux2_klein_4b': ('flux2', 'Flux2'),
    'flux2_klein_9b': ('flux2', 'Flux2'),
    'flux2_klein_base_4b': ('flux2', 'Flux2'),
    'flux2_klein_base_9b': ('flux2', 'Flux2'),
    'flux1_dev': ('flux1', 'Flux1'),
    'flux1_schnell': ('flux1', 'Flux1'),
    'jit_b32': ('jit', 'JiT'),
    'jit_l32': ('jit', 'JiT'),
    'jit_h32': ('jit', 'JiT'),
    'pixelgen_512': ('pixelgen', 'PixelGen'),
    'qwen_image_2512': ('qwen_image', 'QwenImage'),
    'qwen_image': ('qwen_image', 'QwenImage'),
    'stable_diffusion_3': ('stable_diffusion_3', 'StableDiffusion3'),
    'stable_diffusion_15': ('stable_diffusion_15', 'StableDiffusion15'),
    'sdxl': ('sdxl', 'SDXL'),
    'cogvideox_2b': ('cogvideox', 'CogVideoX'),
    'ltx_2b': ('ltx_video', 'LTXVideo'),
    'ltx_13b': ('ltx_video', 'LTXVideo'),
    'wan21_1_3b': ('wan', 'Wan'),
    'cogvideox_5b': ('cogvideox', 'CogVideoX'),
    'cogvideox15_5b': ('cogvideox', 'CogVideoX'),
    'hunyuan_video': ('hunyuan_video', 'HunyuanVideo'),
    'wan21_14b': ('wan', 'Wan'),
    'wan22_a14b': ('wan', 'Wan'),
    'hidream_o1': ('hidream_o1', 'HiDreamO1'),
}


def load_model(name, config, preset, device):
    module, class_name = MODEL_CLASSES[name]
    model_class = getattr(import_module(f"scoring.{module}"), class_name)
    return model_class(config, preset, device)
