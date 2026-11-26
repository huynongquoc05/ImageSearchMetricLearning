import torch
import torch.nn.functional as F
from tqdm import tqdm


def _embed(model, x, device, amp):
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=amp and device.type == "cuda"):
        f = model(x)
    return F.normalize(f.float(), dim=1)


@torch.no_grad()
def extract(model, loader, device, amp: bool = True, flip: bool = False):
    """flip=True: TTA, lấy trung bình embedding (đã normalize) của ảnh gốc và ảnh lật ngang."""
    model.eval()
    feats, labels = [], []
    for x, y in tqdm(loader, desc="extract", leave=False):
        x = x.to(device, non_blocking=True)
        f = _embed(model, x, device, amp)
        if flip:
            f = F.normalize(f + _embed(model, x.flip(3), device, amp), dim=1)
        feats.append(f.cpu())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels)


def recall_at_k(feats, labels, ks=(1, 2, 4, 8)):
    """Query = gallery = toàn bộ tập, loại chính ảnh query ra khỏi kết quả."""
    sims = feats @ feats.T
    sims.fill_diagonal_(-float("inf"))
    idx = sims.topk(max(ks), dim=1).indices          # (N, maxK)
    match = labels[idx] == labels[:, None]           # (N, maxK)
    return {f"R@{k}": match[:, :k].any(dim=1).float().mean().item() * 100 for k in ks}


def map_at_r(feats, labels):
    """mAP@R (Musgrave et al. 2020): với mỗi query, R = số ảnh cùng lớp (trừ chính nó);
    lấy top-R kết quả, AP = tổng precision tại các vị trí đúng / R."""
    sims = feats @ feats.T
    sims.fill_diagonal_(-float("inf"))
    R = torch.bincount(labels)[labels] - 1           # (N,)
    max_r = int(R.max())
    idx = sims.topk(max_r, dim=1).indices
    match = (labels[idx] == labels[:, None]).float()  # (N, max_r)
    ranks = torch.arange(1, max_r + 1)
    prec = match.cumsum(dim=1) / ranks
    within_r = (ranks[None, :] <= R[:, None]).float()
    ap = (prec * match * within_r).sum(dim=1) / R.clamp(min=1)
    return ap.mean().item() * 100


def evaluate(model, loader, device, flip: bool = False):
    feats, labels = extract(model, loader, device, flip=flip)
    out = recall_at_k(feats, labels)
    out["mAP@R"] = map_at_r(feats, labels)
    return out