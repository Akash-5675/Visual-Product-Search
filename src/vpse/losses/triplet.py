"""Triplet losses over L2-normalized embeddings.

Both operate on a PK batch (see data/samplers.py). Distances are Euclidean;
since embeddings are unit-norm, d^2 = 2 - 2*cosine, so ranking is equivalent.
"""
import torch
import torch.nn.functional as F


def _pairwise_dist(emb: torch.Tensor) -> torch.Tensor:
    return torch.cdist(emb, emb, p=2)


def triplet_random(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2):
    """Anchor-positive pairs from the batch, one *random* negative each."""
    dist = _pairwise_dist(emb)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    losses = []
    for a in range(len(labels)):
        pos_idx = pos_mask[a].nonzero(as_tuple=True)[0]
        neg_idx = neg_mask[a].nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        p = pos_idx[torch.randint(len(pos_idx), (1,), device=emb.device)]
        n = neg_idx[torch.randint(len(neg_idx), (1,), device=emb.device)]
        losses.append(F.relu(dist[a, p] - dist[a, n] + margin))
    return torch.cat(losses).mean() if losses else emb.sum() * 0.0


def triplet_batch_hard(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2):
    """Batch-hard mining (Hermans et al. 2017): for each anchor take the
    hardest (farthest) positive and hardest (closest) negative in the batch."""
    dist = _pairwise_dist(emb)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)

    pos_dist = dist.masked_fill(~(same & ~eye), float("-inf")).max(dim=1).values
    neg_dist = dist.masked_fill(same, float("inf")).min(dim=1).values

    valid = torch.isfinite(pos_dist) & torch.isfinite(neg_dist)
    return F.relu(pos_dist[valid] - neg_dist[valid] + margin).mean()
