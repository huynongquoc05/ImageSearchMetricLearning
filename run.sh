R=/kaggle/input/datasets/wenewone/cub2002011/CUB_200_2011
O=/kaggle/working

# Run 1: R50 + MS
python train.py --root $R --out_dir $O --name r50_ms_f1 --backbone resnet50 --loss ms \
  --freeze_bn --pool avgmax --dim 512 --lr_bb 3e-5 --lr_head 1e-3 \
  --p 32 --k 4 --epochs 15 --size 320 --multi_gpu

# Run 2: R50 + Proxy-Anchor
python train.py --root $R --out_dir $O --name r50_pa_f1 --backbone resnet50 --loss proxy_anchor \
  --freeze_bn --pool avgmax --dim 512 --lr_bb 3e-5 --lr_head 1e-3 --lr_proxy 1e-2 \
  --p 32 --k 4 --epochs 15 --size 320 --multi_gpu

# Run 3: DINOv2-S + MS
python train.py --root $R --out_dir $O --name dino_ms_f1 --backbone dinov2_vits14 --loss ms \
  --dim 256 --lr_bb 5e-6 --lr_head 1e-3 --p 32 --k 4 --epochs 12 --size 322 \
  --scale_min 0.5 --color_jitter 0.1 --multi_gpu

# Run 4: DINOv2-S + Proxy-Anchor
python train.py --root $R --out_dir $O --name dino_pa_f1 --backbone dinov2_vits14 --loss proxy_anchor \
  --dim 256 --lr_bb 5e-6 --lr_head 1e-3 --lr_proxy 5e-3 --p 32 --k 4 --epochs 12 --size 322 \
  --scale_min 0.5 --color_jitter 0.1 --multi_gpu