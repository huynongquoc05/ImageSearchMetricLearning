import timm
import torch
import torch.nn as nn


class Embedder(nn.Module):
    """Backbone pretrained (timm) + linear head tuỳ chọn.

    dim=None  -> dùng thẳng feature của backbone (dùng cho baseline chưa fine-tune).
    dim=128.. -> thêm Linear để giảm chiều embedding (dùng khi fine-tune).
    pool      -> kiểu global pooling cho backbone CNN (ResNet...). Ví dụ "avgmax" =
                 0.5 * (avg + max), giữ nguyên số chiều feature (2048 với ResNet-50).
                 None = mặc định của timm (avg). Bỏ qua với ViT/DINOv2 (dùng CLS token).
    """

    def __init__(self, backbone: str = "resnet50", dim=None, pretrained: bool = True,
                 img_size: int = 224, pool: str = None):
        super().__init__()
        is_vit = "vit" in backbone or "dinov2" in backbone
        kw = {"img_size": img_size} if is_vit else {}
        if pool:
            if is_vit:
                print(f"[Embedder] bỏ qua --pool {pool} vì {backbone} là ViT")
            else:
                kw["global_pool"] = pool
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.head = nn.Linear(self.backbone.num_features, dim) if dim else nn.Identity()
        self.amp = False  # train.py bật; autocast đặt trong forward để DataParallel cũng dùng fp16

    def forward(self, x):
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=self.amp and x.is_cuda):
            return self.head(self.backbone(x))