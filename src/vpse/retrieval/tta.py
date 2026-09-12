"""Test-time augmentation for embeddings.

Averaging embeddings over several views of the same image smooths out crop and
orientation sensitivity. Views are produced on the batch tensor, so the dataset
and its transform stay untouched.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

VIEWS = ("identity", "hflip", "zoom")


def _apply_view(x: torch.Tensor, view: str) -> torch.Tensor:
    if view == "identity":
        return x
    if view == "hflip":
        return torch.flip(x, dims=[3])
    if view == "zoom":
        # centre 80% re-scaled back to full size
        h, w = x.shape[-2:]
        dh, dw = int(h * 0.1), int(w * 0.1)
        return F.interpolate(x[:, :, dh:h - dh, dw:w - dw], size=(h, w),
                             mode="bilinear", align_corners=False)
    raise ValueError(view)


@torch.no_grad()
def embed_dataset_tta(model, dataset, device: str, views=VIEWS,
                      batch_size: int = 128, num_workers: int = 2):
    """Returns (embeddings [N, d] float32, labels [N]), averaged over views."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embs, labels = [], []
    for x, y in tqdm(loader, desc=f"embedding (TTA x{len(views)})"):
        x = x.to(device)
        acc = None
        for v in views:
            e = model(_apply_view(x, v))
            acc = e if acc is None else acc + e
        embs.append(F.normalize(acc, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs).astype("float32"), np.concatenate(labels)
