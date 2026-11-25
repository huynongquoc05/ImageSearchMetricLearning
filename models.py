import timm
import torch.nn as nn


class Embedder(nn.Module):
    """Backbone pretrained (timm) + linear head tuỳ chọn.

    dim=None  -> dùng thẳng feature của backbone (dùng cho baseline chưa fine-tune).
    dim=128.. -> thêm Linear để giảm chiều embedding (dùng khi fine-tune).
    """

    def __init__(self, backbone: str = "resnet50", dim=None, pretrained: bool = True, img_size: int = 224):
        super().__init__()
        kw = {"img_size": img_size} if ("vit" in backbone or "dinov2" in backbone) else {}
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        self.head = nn.Linear(self.backbone.num_features, dim) if dim else nn.Identity()

    def forward(self, x):
        return self.head(self.backbone(x))
