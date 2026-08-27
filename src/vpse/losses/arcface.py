"""ArcFace (Deng et al. 2019): additive angular margin on a cosine classifier.

Used only at training time — at inference the class weights are discarded and
retrieval runs on the embeddings.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, n_classes: int,
                 scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # emb is already L2-normalized by the embedder
        cos = F.linear(emb, F.normalize(self.weight, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1.0 - cos ** 2)
        cos_with_margin = cos * self.cos_m - sin * self.sin_m  # cos(theta + m)
        # margin only makes sense while theta + m < pi; fall back otherwise
        cos_target = torch.where(cos > math.cos(math.pi - self.margin),
                                 cos_with_margin,
                                 cos - self.margin * self.sin_m)
        onehot = F.one_hot(labels, num_classes=self.weight.shape[0]).bool()
        logits = torch.where(onehot, cos_target, cos) * self.scale
        return F.cross_entropy(logits, labels)
