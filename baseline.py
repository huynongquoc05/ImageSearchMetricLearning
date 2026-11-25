"""Baseline: embedding pretrained, KHÔNG fine-tune.

python baseline.py --root /path/to/CUB_200_2011
python baseline.py --root ... --backbones resnet50 vit_small_patch14_dinov2.lvd142m
"""
import argparse
import json
from pathlib import Path

import torch

from data import CUBDataset, build_transform, make_test_loader
from evaluate import evaluate
from models import Embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--backbones", nargs="+",
                    default=["resnet50", "vit_small_patch14_dinov2.lvd142m"])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="results/baseline.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CUBDataset(a.root, "test", build_transform(False, a.size))
    print(f"test: {len(ds)} ảnh, {len(set(ds.labels))} lớp | device: {device}")
    loader = make_test_loader(ds, a.bs, a.workers)

    results = {}
    for name in a.backbones:
        model = Embedder(name, dim=None, pretrained=True, img_size=a.size).to(device)
        r = evaluate(model, loader, device)
        results[name] = r
        print(name, {k: round(v, 2) for k, v in r.items()})
        del model
        torch.cuda.empty_cache()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
