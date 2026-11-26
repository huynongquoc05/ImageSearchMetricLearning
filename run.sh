#!/usr/bin/env bash
# Giai đoạn 1: 4 run trên CUB. Dùng: bash run_cub.sh <root CUB_200_2011> <out_dir> [seed]
# Kaggle: bash run_cub.sh /kaggle/input/.../CUB_200_2011 /kaggle/working 0
ROOT=${1:-CUB_200_2011}
OUT=${2:-.}
SEED=${3:-0}
S="--seed $SEED"

python train.py --root $ROOT --out_dir $OUT $S --name r50_ms_v2_s$SEED --backbone resnet50 --loss ms           --freeze_bn --dim 512 --lr_bb 3e-5 --lr_head 1e-3                    --p 32 --k 4 --epochs 60
python train.py --root $ROOT --out_dir $OUT $S --name r50_pa_v2_s$SEED --backbone resnet50 --loss proxy_anchor --freeze_bn --dim 512 --lr_bb 3e-5 --lr_head 1e-3 --lr_proxy 1e-2 --p 32 --k 4 --epochs 60
# Run 3: DINOv2-S + Multi-Similarity (giữ scale_min=0.5 an toàn cho ViT)
python train.py --root /kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011 \
  --out_dir /kaggle/working --backbone dinov2_vits14 --loss ms \
  --dim 256 --lr_bb 5e-6 --epochs 12 --scale_min 0.5 --color_jitter 0.1 \
  --name dino_ms

# Run 4: DINOv2-S + Proxy-Anchor (so sánh trực tiếp với Run 3)
python train.py --root /kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011 \
  --out_dir /kaggle/working --backbone dinov2_vits14 --loss proxy_anchor \
  --dim 256 --lr_bb 5e-6 --lr_proxy 5e-3 --epochs 12 --scale_min 0.5 --color_jitter 0.1 \
  --name dino_pa