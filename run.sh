#!/usr/bin/env bash
# Giai đoạn 1: 4 run trên CUB. Dùng: bash run_cub.sh <root CUB_200_2011> <out_dir> [seed]
# Kaggle: bash run_cub.sh /kaggle/input/.../CUB_200_2011 /kaggle/working 0
ROOT=${1:-CUB_200_2011}
OUT=${2:-.}
SEED=${3:-0}
S="--seed $SEED"

python train.py --root $ROOT --out_dir $OUT $S --name r50_ms_s$SEED  --backbone resnet50      --loss ms           --dim 256 --lr_bb 1e-4 --epochs 25
python train.py --root $ROOT --out_dir $OUT $S --name r50_pa_s$SEED  --backbone resnet50      --loss proxy_anchor --dim 256 --lr_bb 1e-4 --lr_proxy 1e-2 --epochs 25
python train.py --root $ROOT --out_dir $OUT $S --name dino_ms_s$SEED --backbone dinov2_vits14 --loss ms           --dim 256 --lr_bb 5e-6 --epochs 12
python train.py --root $ROOT --out_dir $OUT $S --name dino_pa_s$SEED --backbone dinov2_vits14 --loss proxy_anchor --dim 256 --lr_bb 5e-6 --lr_proxy 5e-3 --epochs 12