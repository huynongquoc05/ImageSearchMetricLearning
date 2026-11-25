"""Kiểm tra dataset + sampler: python sanity_check.py --root /path/to/CUB_200_2011"""
import argparse
from collections import Counter

from data import CUBDataset, build_transform, make_train_loader

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()

    tr = CUBDataset(a.root, "train", build_transform(True))
    te = CUBDataset(a.root, "test", build_transform(False))
    print(f"train: {len(tr)} ảnh / {len(set(tr.labels))} lớp  (kỳ vọng ~5864 / 100)")
    print(f"test : {len(te)} ảnh / {len(set(te.labels))} lớp  (kỳ vọng ~5924 / 100)")

    loader = make_train_loader(tr, a.p, a.k, workers=0)
    x, y = next(iter(loader))
    c = Counter(y.tolist())
    print("batch:", tuple(x.shape), "| số lớp:", len(c), "| ảnh/lớp:", set(c.values()))
    assert len(c) == a.p and set(c.values()) == {a.k}, "Sampler P x K sai!"
    print("OK")
