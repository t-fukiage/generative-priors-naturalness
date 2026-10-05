# Model index

Models are described in Appendices C–D. Runtime profiles are in
[environments/README.md](environments/README.md).

## Generative models (25)

The entry point is `python -m scoring.score_pairs --model MODEL ...`.

| Model key | Implementation | Fixed settings | Runtime |
| --- | --- | --- | --- |
| `flux2_klein_4b` | [flux2.py](scoring/flux2.py) | [config](configs/flux2_klein_4b.json) | diffusers |
| `flux2_klein_9b` | [flux2.py](scoring/flux2.py) | [config](configs/flux2_klein_9b.json) | diffusers |
| `flux2_klein_base_4b` | [flux2.py](scoring/flux2.py) | [config](configs/flux2_klein_base_4b.json) | diffusers |
| `flux2_klein_base_9b` | [flux2.py](scoring/flux2.py) | [config](configs/flux2_klein_base_9b.json) | diffusers |
| `flux1_dev` | [flux1.py](scoring/flux1.py) | [config](configs/flux1_dev.json) | diffusers |
| `flux1_schnell` | [flux1.py](scoring/flux1.py) | [config](configs/flux1_schnell.json) | diffusers |
| `jit_b32` | [jit.py](scoring/jit.py) | [config](configs/jit_b32.json) | diffusers |
| `jit_l32` | [jit.py](scoring/jit.py) | [config](configs/jit_l32.json) | diffusers |
| `jit_h32` | [jit.py](scoring/jit.py) | [config](configs/jit_h32.json) | diffusers |
| `pixelgen_512` | [pixelgen.py](scoring/pixelgen.py) | [config](configs/pixelgen_512.json) | diffusers |
| `qwen_image_2512` | [qwen_image.py](scoring/qwen_image.py) | [config](configs/qwen_image_2512.json) | diffusers |
| `qwen_image` | [qwen_image.py](scoring/qwen_image.py) | [config](configs/qwen_image.json) | diffusers |
| `stable_diffusion_3` | [stable_diffusion_3.py](scoring/stable_diffusion_3.py) | [config](configs/stable_diffusion_3.json) | diffusers |
| `stable_diffusion_15` | [stable_diffusion_15.py](scoring/stable_diffusion_15.py) | [config](configs/stable_diffusion_15.json) | diffusers |
| `sdxl` | [sdxl.py](scoring/sdxl.py) | [config](configs/sdxl.json) | diffusers |
| `cogvideox_2b` | [cogvideox.py](scoring/cogvideox.py) | [config](configs/cogvideox_2b.json) | cogvideox |
| `ltx_2b` | [ltx_video.py](scoring/ltx_video.py) | [config](configs/ltx_2b.json) | diffusers |
| `ltx_13b` | [ltx_video.py](scoring/ltx_video.py) | [config](configs/ltx_13b.json) | diffusers |
| `wan21_1_3b` | [wan.py](scoring/wan.py) | [config](configs/wan21_1_3b.json) | wan |
| `cogvideox_5b` | [cogvideox.py](scoring/cogvideox.py) | [config](configs/cogvideox_5b.json) | cogvideox |
| `cogvideox15_5b` | [cogvideox.py](scoring/cogvideox.py) | [config](configs/cogvideox15_5b.json) | cogvideox |
| `hunyuan_video` | [hunyuan_video.py](scoring/hunyuan_video.py) | [config](configs/hunyuan_video.json) | diffusers |
| `wan21_14b` | [wan.py](scoring/wan.py) | [config](configs/wan21_14b.json) | wan |
| `wan22_a14b` | [wan.py](scoring/wan.py) | [config](configs/wan22_a14b.json) | wan |
| `hidream_o1` | [hidream_o1.py](scoring/hidream_o1.py) | [config](configs/hidream_o1.json) | hidream |

## Encoders (12)

The entry point is `python -m encoders.score_pairs --model MODEL ...`.
All settings are in [encoders.json](configs/encoders.json); pooling definitions
are defined in Appendix D.1.

| Model key | Implementation | Layers + endpoint | Readout | Runtime |
| --- | --- | --- | --- | --- |
| `resnet50_imagenet` | [vision.py](encoders/vision.py) | 16 + 1 | native_head | encoders_iqa |
| `clip_vit_b16` | [vision.py](encoders/vision.py) | 12 + 1 | raw | encoders_iqa |
| `dinov2_base_patch_spatial` | [vision.py](encoders/vision.py) | 12 + 1 | native_head | encoders_iqa |
| `siglip2_base_patch16_224` | [vision.py](encoders/vision.py) | 12 + 1 | native_head | vlm |
| `pe_core_b16_224` | [vision.py](encoders/vision.py) | 12 + 1 | raw | vlm |
| `dinov2_giant_patch_spatial` | [vision.py](encoders/vision.py) | 40 + 1 | native_head | encoders_iqa |
| `pe_core_g14_448` | [vision.py](encoders/vision.py) | 50 + 1 | raw | encoders_iqa |
| `dinov3_vith16plus_patch_spatial` | [vision.py](encoders/vision.py) | 32 + 1 | native_head | encoders_iqa |
| `dinov3_vit7b16_patch_spatial` | [vision.py](encoders/vision.py) | 40 + 1 | native_head | encoders_iqa |
| `qwen3_vl_embedding_2b` | [qwen_vl.py](encoders/qwen_vl.py) | 28 + 1 | native_head | vlm |
| `qwen3_vl_embedding_8b` | [qwen_vl.py](encoders/qwen_vl.py) | 36 + 1 | native_head | vlm |
| `jina_clip_v2` | [jina_clip.py](encoders/jina_clip.py) | 24 + 1 | native_head | diffusers |

## IQA metrics (10)

The entry point is `python -m iqa.score_pairs --metric METRIC ...`.
The implementation is [quality.py](iqa/quality.py); fixed backend names are in
[iqa.json](configs/iqa.json). All use the `encoders_iqa` runtime.

| Metric key | Reference mode | Native quality direction |
| --- | --- | --- |
| `psnr` | FR | Higher is better |
| `ssim` | FR | Higher is better |
| `vif` | FR | Higher is better |
| `lpips_vgg` | FR | Lower is better |
| `dists` | FR | Lower is better |
| `niqe` | NR | Lower is better |
| `brisque` | NR | Lower is better |
| `hyperiqa` | NR | Higher is better |
| `musiq` | NR | Higher is better |
| `liqe` | NR | Higher is better |
