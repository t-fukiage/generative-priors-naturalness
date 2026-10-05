# Runtime Environments

This guide specifies Python environments for CPU-based statistical analysis and GPU-based model scoring.

---

## 1. CPU Analysis Environment (Mode 1)

All statistical analyses, bootstrap resampling, and figure generation run on standard Python 3.10+:

```bash
python -m pip install -r environments/analysis_requirements.txt
```

---

## 2. GPU Model Scoring Profiles (Mode 2)

Running model inference from raw weights requires dedicated PyTorch and CUDA environments. Package versions are pinned in the requirements files and [runtime_profiles.json](runtime_profiles.json).

| Profile | Models | Python | PyTorch / CUDA Wheels | Requirements |
| --- | --- | --- | --- | --- |
| `diffusers` | Image models except HiDream; LTX; Hunyuan; Jina | 3.10 | 2.7.1 / cu126 | [diffusers_requirements.txt](diffusers_requirements.txt) |
| `cogvideox` | All three CogVideoX checkpoints | 3.11 | 2.5.1 / cu121 | [cogvideox_requirements.txt](cogvideox_requirements.txt) |
| `wan` | All three Wan checkpoints | 3.11 | 2.5.1 / cu121 | [wan_requirements.txt](wan_requirements.txt) |
| `encoders_iqa` | ResNet, CLIP, DINO, PE-G; all IQA metrics | 3.12 | 2.10.0 / cu128 | [encoders_iqa_requirements.txt](encoders_iqa_requirements.txt) |
| `vlm` | SigLIP 2, PE-B, Qwen3-VL-Embedding 2B/8B | 3.11 | 2.5.1 / cu124 | [vlm_requirements.txt](vlm_requirements.txt) |
| `hidream` | HiDream-O1 Full | 3.12 | 2.10.0 / cu128 | [hidream_requirements.txt](hidream_requirements.txt) |

### Example Setup (`diffusers` Profile)
```bash
python3.10 -m venv .venv-diffusers
source .venv-diffusers/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r environments/diffusers_requirements.txt
```

For other profiles, install the corresponding Python and CUDA wheel versions listed above. Checkpoints download automatically at the revisions specified in `configs/`. Use `HF_HUB_OFFLINE=1` to load from an existing cache.

---

## 3. Additional External Model Dependencies

Certain model families require cloning their official repositories and applying minor compatibility patches:

| Family | Official Source | Commit Hash | Required Local Patch |
| --- | --- | --- | --- |
| JiT | [LTH14/JiT](https://github.com/LTH14/JiT) | `cbc743a2ada5e9762697da2c83f8c4f8379e8c17` | [Device placement patch](patches/jit_local_fixes.patch) |
| PixelGen | [Zehong-Ma/PixelGen](https://github.com/Zehong-Ma/PixelGen) | `0ccec6029f4c011590331be82f7e4eca8fb9ac4a` | [Qwen3 device/compile helper](patches/prepare_pixelgen.py); keep compilation disabled |
| HiDream | [HiDream-ai/HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image) | `2c2d29ff729e48f33e41f49edfdbd81d5ac103b4` | None |

### Setup Example (JiT)
```bash
git clone https://github.com/LTH14/JiT.git /path/to/JiT
git -C /path/to/JiT checkout cbc743a2ada5e9762697da2c83f8c4f8379e8c17
git -C /path/to/JiT apply "$PWD/environments/patches/jit_local_fixes.patch"
PYTHONPATH=/path/to/JiT python -m scoring.score_pairs --model jit_b32 --preset paper
```

For a separately obtained PixelGen checkout at the pinned commit, use the independent helper:

```bash
# Preview and check the exact source hash; do not execute model code.
python environments/patches/prepare_pixelgen.py --pixelgen-root /path/to/PixelGen
# Apply once, preserving an adjacent backup of the original file.
python environments/patches/prepare_pixelgen.py --pixelgen-root /path/to/PixelGen --apply
```

Set `PIXELGEN_QWEN3_COMPILE=0`. Official checkpoint weights should be placed at:
```text
checkpoints/jit-b-32/checkpoint-last.pth
checkpoints/jit-l-32/checkpoint-last.pth
checkpoints/jit-h-32/checkpoint-last.pth
checkpoints/PixelGen_XXL_T2I.ckpt
```

> **Note:** Wan2.2 checkpoint loading requires approximately 172 GiB of host RAM in addition to GPU resources. Run one model evaluation per process.
