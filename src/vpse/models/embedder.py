import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class Embedder(nn.Module):
    """ResNet50 backbone -> GAP -> linear head -> L2-normalized embedding.

    With freeze_backbone=True and use_head=False this is the Phase 1 baseline:
    raw pretrained pool5 features, no training at all.
    """

    def __init__(self, embedding_dim: int = 512, freeze_backbone: bool = False,
                 use_head: bool = True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # drop fc
        self.use_head = use_head
        self.head = nn.Linear(2048, embedding_dim) if use_head else nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x).flatten(1)
        emb = self.head(feats)
        return F.normalize(emb, dim=1)
