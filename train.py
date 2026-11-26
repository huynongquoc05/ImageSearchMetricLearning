"""Fine-tune metric learning trên CUB-200-2011.

Protocol:
  - Train: 100 lớp đầu của CUB, trong đó `--val_classes` lớp được tách làm validation
    (chọn cấu hình / checkpoint tốt nhất theo val mAP@R, ổn định hơn R@1).
  - Test: 100 lớp cuối, chỉ chạy MỘT lần ở cuối bằng checkpoint tốt nhất.

Ví dụ:
  # baseline cũ (augmentation cũ, BN không đóng băng)
  python train.py --root CUB_200_2011 --backbone resnet50 --loss ms \
      --dim 256 --lr_bb 1e-4 --epochs 25 --scale_min 0.5 --color_jitter 0 --name r50_ms_old

  # recipe đề xuất cho ResNet-50
  python train.py --root CUB_200_2011 --backbone resnet50 --loss ms --freeze_bn \
      --dim 512 --lr_bb 3e-5 --epochs 60 --p 32 --k 4 --name r50_ms_fbn
  python train.py --root CUB_200_2011 --backbone resnet50 --loss proxy_anchor --freeze_bn \
      --dim 512 --lr_bb 3e-5 --lr_proxy 1e-2 --epochs 60 --p 32 --k 4 --name r50_pa_fbn

  python train.py --root CUB_200_2011 --backbone dinov2_vits14 --loss proxy_anchor \
      --dim 256 --lr_bb 5e-6 --lr_proxy 5e-3 --epochs 12 --name dino_pa
  python train.py --root CUB_200_2011 --smoke      # chạy thử vài bước để bắt lỗi
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from pytorch_metric_learning import losses, miners

from data import (CUBDataset, build_transform, make_train_loader, make_test_loader,
                  subset_by_classes)
from evaluate import evaluate
from models import Embedder

BACKBONES = {
    "resnet50": "resnet50",
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
}


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", default=".", help="chứa results/ và checkpoints/")
    ap.add_argument("--name", default=None)
    ap.add_argument("--backbone", default="resnet50", help=f"{list(BACKBONES)} hoặc tên timm")
    ap.add_argument("--loss", choices=["ms", "proxy_anchor"], default="ms")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--iters_per_epoch", type=int, default=None,
                    help="số batch mỗi epoch; mặc định = số ảnh train // (p*k) (~70 với 80 lớp)")
    ap.add_argument("--p", type=int, default=16, help="số lớp / batch")
    ap.add_argument("--k", type=int, default=4, help="số ảnh / lớp")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr_bb", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=None, help="mặc định = 10 x lr_bb")
    ap.add_argument("--lr_proxy", type=float, default=1e-2)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=1.0, help="số epoch warmup")
    ap.add_argument("--val_classes", type=int, default=20, help="0 = không dùng val, lấy epoch cuối")
    ap.add_argument("--select_by", choices=["mAP@R", "R@1"], default="mAP@R",
                    help="metric val để chọn checkpoint")
    # ---- augmentation / tiền xử lý (data.py) ----
    ap.add_argument("--scale_min", type=float, default=0.25,
                    help="cận dưới scale của RandomResizedCrop (cũ: 0.5)")
    ap.add_argument("--color_jitter", type=float, default=0.2, help="0 = tắt ColorJitter")
    ap.add_argument("--test_resize", choices=["crop", "full"], default="crop",
                    help="crop = Resize+CenterCrop; full = Resize thẳng về size x size")
    ap.add_argument("--pool", default=None,
                    help="global pooling cho CNN, vd avgmax (= avg+max); bỏ qua với ViT")
    ap.add_argument("--freeze_bn", action="store_true", help="đóng băng BatchNorm (ResNet)")
    ap.add_argument("--multi_gpu", action="store_true",
                    help="dùng tất cả GPU qua DataParallel (Kaggle T4 x2); loss vẫn tính trên cả batch")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="chạy thử rất ngắn, bỏ qua test")
    a = ap.parse_args()
    a.backbone_timm = BACKBONES.get(a.backbone, a.backbone)
    a.lr_head = a.lr_head if a.lr_head is not None else 10 * a.lr_bb
    a.name = a.name or f"{a.backbone}_{a.loss}_d{a.dim}"
    if a.smoke:
        a.epochs, a.val_classes, a.name = 1, 10, a.name + "_smoke"
    return a


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_loss(a, num_classes):
    # Lưu ý: MS loss và miner của pytorch_metric_learning mặc định dùng CosineSimilarity,
    # tức là đã tự L2-normalize embedding bên trong; ProxyAnchorLoss cũng normalize embedding
    # và proxy. Vì vậy không cần normalize thêm trước khi đưa vào loss.
    if a.loss == "ms":
        return (losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5),
                miners.MultiSimilarityMiner(epsilon=0.1))
    return losses.ProxyAnchorLoss(num_classes, a.dim, margin=0.1, alpha=32), None


def freeze_bn_params(model):
    """Không cập nhật gamma/beta của BN (running stats do set_train giữ ở eval)."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for p in m.parameters():
                p.requires_grad_(False)


def set_train(model, freeze_bn):
    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def fmt(m):
    return " ".join(f"{k} {v:.2f}" for k, v in m.items() if k in ("R@1", "R@4", "mAP@R"))


def main():
    a = get_args()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not a.no_amp
    out = Path(a.out_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoints" / f"{a.name}.pt"

    # ---- dữ liệu: tách val theo LỚP (cố định seed 0 để mọi run dùng cùng split) ----
    train_tf = build_transform(True, a.size, scale_min=a.scale_min, color_jitter=a.color_jitter)
    eval_tf = build_transform(False, a.size, test_resize=a.test_resize)
    full_train = CUBDataset(a.root, "train")
    classes = sorted(set(full_train.labels))
    val_c = random.Random(0).sample(classes, a.val_classes) if a.val_classes > 0 else []
    tr_c = [c for c in classes if c not in val_c]
    train_ds = subset_by_classes(full_train, tr_c, train_tf)
    val_ds = subset_by_classes(full_train, val_c, eval_tf) if val_c else None
    test_ds = CUBDataset(a.root, "test", eval_tf)
    print(f"train {len(train_ds)} ảnh/{len(tr_c)} lớp | val {len(val_ds) if val_ds else 0} ảnh/"
          f"{len(val_c)} lớp | test {len(test_ds)} ảnh | device {device} | amp {use_amp}")

    num_batches = 3 if a.smoke else a.iters_per_epoch
    train_loader = make_train_loader(train_ds, a.p, a.k, a.workers,
                                     num_batches=num_batches, seed=a.seed)
    val_loader = make_test_loader(val_ds, 64, a.workers) if val_ds else None
    test_loader = make_test_loader(test_ds, 64, a.workers)
    print(f"{len(train_loader)} batch/epoch x {a.epochs} epoch = "
          f"{len(train_loader) * a.epochs} iteration | batch {a.p * a.k}")

    # ---- model, loss, optimizer ----
    model = Embedder(a.backbone_timm, dim=a.dim, pretrained=True, img_size=a.size,
                     pool=a.pool).to(device)
    model.amp = use_amp
    net = model
    if a.multi_gpu and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model)  # batch chia đôi cho 2 GPU, embedding gom về GPU 0
        print(f"DataParallel trên {torch.cuda.device_count()} GPU")
    if a.freeze_bn:
        freeze_bn_params(model)
    loss_fn, miner = build_loss(a, len(tr_c))
    loss_fn = loss_fn.to(device)

    groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": a.lr_bb},
              {"params": list(model.head.parameters()), "lr": a.lr_head}]
    proxy_params = list(loss_fn.parameters())
    if proxy_params:  # ProxyAnchor: proxy là tham số học được
        groups.append({"params": proxy_params, "lr": a.lr_proxy, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)

    total = a.epochs * len(train_loader)
    warm = int(a.warmup * len(train_loader))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = make_scaler(use_amp)

    # ---- vòng train ----
    history, best_score, best_epoch = [], -1.0, 0
    if val_loader:
        m0 = evaluate(net, val_loader, device)  # head ngẫu nhiên: mốc trước khi train
        history.append({"epoch": 0, "loss": None, "val": m0})
        print(f"ep 00 | val {fmt(m0)}")

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        set_train(model, a.freeze_bn)
        run_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            emb = net(x)  # autocast nằm trong Embedder.forward
            emb = emb.float()  # tính loss ở fp32
            loss = loss_fn(emb, y, miner(emb, y)) if miner else loss_fn(emb, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            n += 1
        rec = {"epoch": ep, "loss": run_loss / max(1, n)}
        msg = f"ep {ep:02d}/{a.epochs} | loss {rec['loss']:.4f}"
        if val_loader:
            rec["val"] = evaluate(net, val_loader, device)
            msg += f" | val {fmt(rec['val'])}"
            if rec["val"][a.select_by] > best_score:
                best_score, best_epoch = rec["val"][a.select_by], ep
                torch.save({"model": model.state_dict(), "args": vars(a), "epoch": ep}, ckpt_path)
                msg += " *"
        history.append(rec)
        print(msg + f" | {(time.time() - t0) / 60:.1f} phút")

    if not val_loader:  # không có val -> dùng epoch cuối
        best_epoch = a.epochs
        torch.save({"model": model.state_dict(), "args": vars(a), "epoch": a.epochs}, ckpt_path)

    # ---- test: đúng một lần, bằng checkpoint tốt nhất ----
    result = {"args": vars(a), "best_epoch": best_epoch, "history": history,
              "train_minutes": (time.time() - t0) / 60}
    res_path = out / "results" / f"{a.name}.json"
    res_path.write_text(json.dumps(result, indent=2))  # ghi trước, phòng khi bước test lỗi
    if not a.smoke:
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        result["test"] = evaluate(net, test_loader, device)
        result["test_flip_tta"] = evaluate(net, test_loader, device, flip=True)
        print(f"TEST (epoch {best_epoch}): " + " ".join(f"{k} {v:.2f}" for k, v in result["test"].items()))
        print(f"TEST + flip TTA     : " + " ".join(f"{k} {v:.2f}" for k, v in result["test_flip_tta"].items()))
    (out / "results" / f"{a.name}.json").write_text(json.dumps(result, indent=2))
    print("saved ->", out / "results" / f"{a.name}.json")


if __name__ == "__main__":
    main()