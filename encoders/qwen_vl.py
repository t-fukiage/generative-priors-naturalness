"""Image-only Qwen3-VL embedding, with the official EOS and normalization order."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLPreTrainedModel
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from .common import LayerEncoder, normalized


class QwenEmbeddingModel(Qwen3VLPreTrainedModel):
    """The embedding checkpoint wraps Qwen3VLModel without a language-model head."""

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(self, **inputs):
        return self.model(**inputs).last_hidden_state


def load(config, device):
    model = QwenEmbeddingModel.from_pretrained(
        config["backend_name"], revision=config["revision"],
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval().to(device)
    processor = Qwen3VLProcessor.from_pretrained(
        config["backend_name"], revision=config["revision"], padding_side="right",
    )
    tower = model.model.language_model
    last_position = None

    def eos(hidden):
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_position]

    def encode(images):
        nonlocal last_position
        conversations = [[
            {"role": "system", "content": [{"type": "text", "text": config["instruction"]}]},
            {"role": "user", "content": [{"type": "image", "image": image,
                                          "min_pixels": 4096, "max_pixels": 512 * 512}]},
        ] for image in images]
        text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=images, truncation=True, max_length=8192,
                           padding=True, do_resize=False, do_rescale=False, return_tensors="pt").to(device)
        mask = inputs["attention_mask"]
        last_position = mask.shape[1] - mask.flip(1).argmax(1) - 1
        # Normalize in bfloat16, then float32; LayerEncoder applies a second float32 norm.
        return normalized(F.normalize(eos(model(**inputs)), dim=-1))

    def readout(hidden):
        return F.normalize(eos(tower.norm(hidden)), dim=-1)

    return LayerEncoder(encode, list(tower.layers), readout, model)
