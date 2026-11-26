"""CUB-200-2011 dataset + P x K sampler.

Protocol chuẩn cho metric learning: lớp 1-100 để train, lớp 101-200 để test
(test dùng các lớp chưa từng thấy lúc train).
"""
import copy
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transform(train: bool, size: int = 224, scale_min: float = 0.25,
                    color_jitter: float = 0.2, test_resize: str = "crop"):
    """
    train:
      scale_min    : cận dưới của RandomResizedCrop (cũ: 0.5). 0.25 -> augmentation mạnh hơn,
                     giảm overfit vì tập train chỉ ~5.9k ảnh.
      color_jitter : độ mạnh brightness/contrast/saturation (0 = tắt). Không đổi hue,
                     vì màu lông là đặc trưng phân biệt loài chim.
    test:
      test_resize  : "crop" = Resize(size*256/224) + CenterCrop (như cũ, có thể cắt mất rìa)
                     "full" = Resize thẳng về (size, size), giữ toàn bộ ảnh (méo nhẹ tỉ lệ)
    """
    assert test_resize in ("crop", "full")
    if train:
        tf = [T.RandomResizedCrop(size, scale=(scale_min, 1.0)),
              T.RandomHorizontalFlip()]
        if color_jitter > 0:
            tf.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                    saturation=color_jitter))
        tf += [T.ToTensor(), T.Normalize(MEAN, STD)]
        return T.Compose(tf)
    if test_resize == "full":
        resize = [T.Resize((size, size))]
    else:
        resize = [T.Resize(int(size * 256 / 224)), T.CenterCrop(size)]
    return T.Compose(resize + [T.ToTensor(), T.Normalize(MEAN, STD)])


class CUBDataset(Dataset):
    """root: thư mục CUB_200_2011 (chứa images/, images.txt, image_class_labels.txt)."""

    def __init__(self, root, split: str, transform=None):
        assert split in ("train", "test")
        root = Path(root)
        self.img_dir = root / "images"
        id2path = dict(line.split() for line in (root / "images.txt").read_text().splitlines())
        id2cls = dict(line.split() for line in (root / "image_class_labels.txt").read_text().splitlines())

        self.paths, self.labels = [], []
        for img_id, rel in id2path.items():
            cid = int(id2cls[img_id])  # 1..200
            if split == "train" and cid <= 100:
                self.labels.append(cid - 1)      # 0..99
            elif split == "test" and cid > 100:
                self.labels.append(cid - 101)    # 0..99
            else:
                continue
            self.paths.append(self.img_dir / rel)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")  # một số ảnh CUB là grayscale
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


class PKBatchSampler(Sampler):
    """Mỗi batch: p lớp ngẫu nhiên, mỗi lớp k ảnh (lấy lặp nếu lớp có ít hơn k ảnh).

    seed: nếu có, dùng bộ sinh số ngẫu nhiên riêng -> tái lập được mà không phụ thuộc
          trạng thái `random` toàn cục. None = dùng `random` toàn cục như cũ.
    """

    def __init__(self, labels, p: int, k: int, num_batches: int = None, seed: int = None):
        self.p, self.k = p, k
        self.rng = random.Random(seed) if seed is not None else random
        self.by_class = defaultdict(list)
        for i, y in enumerate(labels):
            self.by_class[y].append(i)
        self.classes = list(self.by_class)
        assert len(self.classes) >= p, "p lớn hơn số lớp trong dataset"
        self.num_batches = num_batches or len(labels) // (p * k)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for c in self.rng.sample(self.classes, self.p):
                idxs = self.by_class[c]
                batch += self.rng.sample(idxs, self.k) if len(idxs) >= self.k \
                    else self.rng.choices(idxs, k=self.k)
            yield batch


def make_train_loader(ds, p: int = 16, k: int = 4, workers: int = 4,
                      num_batches: int = None, seed: int = None):
    """Mỗi batch gồm p lớp, mỗi lớp k ảnh -> batch_size = p * k."""
    return DataLoader(ds, batch_sampler=PKBatchSampler(ds.labels, p, k, num_batches, seed),
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def subset_by_classes(ds, classes, transform=None):
    """Dataset con chỉ gồm các lớp trong `classes`; nhãn được đánh lại 0..n-1."""
    remap = {c: i for i, c in enumerate(sorted(classes))}
    sub = copy.copy(ds)
    keep = [i for i, y in enumerate(ds.labels) if y in remap]
    sub.paths = [ds.paths[i] for i in keep]
    sub.labels = [remap[ds.labels[i]] for i in keep]
    if transform is not None:
        sub.transform = transform
    return sub


def make_test_loader(ds, bs: int = 64, workers: int = 4):
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)