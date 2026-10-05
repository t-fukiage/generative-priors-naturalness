"""Small plotting helpers shared by the paper figures."""

import numpy as np
from matplotlib.colors import to_rgb, to_rgba


MODEL_TYPE_COLORS = {"Pixel": "#D55E00", "Latent": "#0072B2", "Video": "#009E73"}
INK = "#263241"


def marker_area(parameter_b):
    """Area in points squared on a logarithmic scale matching the main cross-model figure."""
    return 8 + 15 * np.log10(np.asarray(parameter_b) / 0.1)


def load_generative_model_metadata(data_root=None):
    """Load parameter counts and model categories (Pixel, Latent, Video) for generative models."""
    import json
    import pandas as pd
    from data_io import configured_path
    root = configured_path("data_root", data_root)
    sizes = pd.read_csv(root / "model_sizes/models.csv", float_precision="round_trip")
    metadata = sizes.query("group == 'generative'").drop(columns="group").copy()
    inventory = json.loads((root / "models.json").read_text())["generative"]
    pixel_models = {"jit_b32", "jit_l32", "jit_h32", "pixelgen_512", "hidream_o1"}
    def model_type(name):
        return "Video" if inventory[name]["modality"] == "video" else "Pixel" if name in pixel_models else "Latent"
    metadata["model_type"] = metadata.model.map(model_type)
    metadata["marker_area_pt2"] = marker_area(metadata.parameter_b)
    return metadata


MODEL_FAMILIES = {
    "JiT": ("#D55E00", "o", ("jit_b32", "jit_l32", "jit_h32")),
    "PixelGen": ("#8C564B", "o", ("pixelgen_512",)),
    "HiDream": ("#E83E8C", "o", ("hidream_o1",)),
    "Stable Diffusion": ("#5B6470", "s", ("stable_diffusion_15", "sdxl", "stable_diffusion_3")),
    "FLUX.1": ("#0072B2", "s", ("flux1_dev", "flux1_schnell")),
    "Qwen-Image": ("#CC79A7", "s", ("qwen_image", "qwen_image_2512")),
    "FLUX.2": ("#009E73", "s", ("flux2_klein_base_4b", "flux2_klein_4b", "flux2_klein_base_9b", "flux2_klein_9b")),
    "CogVideoX": ("#56B4E9", "^", ("cogvideox_2b", "cogvideox_5b", "cogvideox15_5b")),
    "LTX-Video": ("#7B61A8", "^", ("ltx_2b", "ltx_13b")),
    "HunyuanVideo": ("#2A7F9E", "^", ("hunyuan_video",)),
    "Wan": ("#A79A00", "^", ("wan21_1_3b", "wan21_14b", "wan22_a14b")),
}

CHECKPOINT_LABELS = {
    "jit_b32": "JiT-B/32", "jit_l32": "JiT-L/32", "jit_h32": "JiT-H/32",
    "pixelgen_512": "PixelGen 512", "hidream_o1": "HiDream-O1-Image",
    "stable_diffusion_15": "SD v1.5", "sdxl": "SDXL Base 1.0", "stable_diffusion_3": "SD3 Medium",
    "flux1_dev": "FLUX.1 dev", "flux1_schnell": "FLUX.1 schnell",
    "qwen_image": "Qwen-Image", "qwen_image_2512": "Qwen-Image-2512",
    "flux2_klein_base_4b": "FLUX.2 klein base 4B", "flux2_klein_4b": "FLUX.2 klein 4B",
    "flux2_klein_base_9b": "FLUX.2 klein base 9B", "flux2_klein_9b": "FLUX.2 klein 9B",
    "cogvideox_2b": "CogVideoX-2B", "cogvideox_5b": "CogVideoX-5B", "cogvideox15_5b": "CogVideoX1.5-5B",
    "ltx_2b": "LTX-Video 2B v0.9.0", "ltx_13b": "LTX-Video 13B dev", "hunyuan_video": "HunyuanVideo",
    "wan21_1_3b": "Wan2.1 T2V 1.3B", "wan21_14b": "Wan2.1 T2V 14B", "wan22_a14b": "Wan2.2 T2V A14B",
}


def interval_point(axis, x, y, *, xlow, xhigh, ylow, yhigh, **style):
    """Draw percentile endpoints independently of the observed point.

    Selection-aware percentile intervals need not contain the point estimate.
    Midpoint-centered line artists preserve the actual endpoints in that case.
    """
    bars = dict(fmt="none", ecolor=style.get("ecolor", style.get("color")),
                elinewidth=style.get("elinewidth", 1), capsize=style.get("capsize", 0),
                zorder=style.get("zorder", 3))
    axis.errorbar((xlow+xhigh)/2, y, xerr=(xhigh-xlow)/2, **bars)
    axis.errorbar(x, (ylow+yhigh)/2, yerr=(yhigh-ylow)/2, **bars)
    axis.errorbar(x, y, **style)


def model_styles(*, distinct=False):
    """Family colors and modality markers in the manuscript's visual order."""
    styles = {}
    for family, (color, marker, models) in MODEL_FAMILIES.items():
        bounds = (-.30, .38) if distinct else (-.12, .22)
        offsets = np.linspace(*bounds, len(models)) if len(models) > 1 else [0]
        for model, offset in zip(models, offsets):
            target = np.ones(3) if offset > 0 else np.zeros(3)
            shade = (1 - abs(offset)) * np.asarray(to_rgb(color)) + abs(offset) * target
            styles[model] = {"family": family, "color": shade, "marker": marker}
    return styles


def checkpoint_legend(figure, styles, *, column_x, top, row_step, fontsize, marker_size, text_pad):
    """Draw all 25 individually named markers in two columns, in inch coordinates."""
    width, height = figure.get_size_inches()
    axis = figure.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    axis.set_axis_off()
    for index, (model, style) in enumerate(styles.items()):
        column, row = (0, index) if index < 12 else (1, index - 12)
        x, y = column_x[column], top - row_step * row
        axis.plot([x], [y], marker=style["marker"], color=style["color"], markersize=marker_size,
                  linestyle="none", markeredgewidth=.3, markeredgecolor="white", clip_on=False)
        label = CHECKPOINT_LABELS.get(model, model)
        axis.text(x + text_pad, y, label, fontsize=fontsize,
                  fontweight="normal", color="#182536", ha="left", va="center")
    return axis


def figure_style(fontsize=7, **overrides):
    """Common rcParams style for publication figures using top-level configured fonts."""
    from data_io import configured_fonts

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": configured_fonts(),
        "font.size": fontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.5,
        "axes.unicode_minus": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
    if overrides:
        style.update(overrides)
    return style
