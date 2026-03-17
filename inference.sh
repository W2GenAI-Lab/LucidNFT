#!/usr/bin/env bash
set -e

# Minimal, readable, zero-logic wrapper. All paths are relative.
# Ensure you've prepared weights beforehand (e.g., source weights/env.sh for FLUX base).

export CUDA_VISIBLE_DEVICES=1

source weights/env.sh

# python inference.py \
#   --checkpoint weights/lucidflux/lucidflux.pth \
#   --control_image /home/notebook/data/personal/S9062836/Real-World-Samples/LucidSR/SR-Benchmark/SR-Benchmark/Bench/RealLQ250/lq_256 \
#   --output_dir outputs-RealLQ250-50 \
#   --width 1024 \
#   --height 1024 \
#   --num_steps 50 \
#   --swinir_pretrained weights/swinir.pth \
#   --siglip_ckpt weights/siglip \
#   --offload

python inference.py \
  --checkpoint weights/lucidflux/lucidflux.pth \
  --control_image assets/3.png \
  --output_dir outputs \
  --width 1024 \
  --height 1024 \
  --num_steps 20 \
  --swinir_pretrained weights/swinir.pth \
  --siglip_ckpt weights/siglip \
  --lora_path weights/lucidflux/LucidFlux+LucidNFT_lora \
  --offload