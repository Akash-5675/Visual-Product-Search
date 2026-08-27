"""Embed a dataset and build/query a FAISS index.

Embeddings are L2-normalized, so inner product == cosine similarity.
"""
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def embed_dataset(model, dataset, device: str, batch_size: int = 128,
                  num_workers: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Returns (embeddings [N, d] float32, labels [N])."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embs, labels = [], []
    for x, y in tqdm(loader, desc="embedding"):
        embs.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs).astype("float32"), np.concatenate(labels)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(index: faiss.Index, queries: np.ndarray, k: int):
    """Returns (similarities [Nq, k], indices [Nq, k])."""
    return index.search(queries.astype("float32"), k)


def save_embeddings(path: Path, embeddings: np.ndarray, labels: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings, labels=labels)


def load_embeddings(path: Path):
    d = np.load(path)
    return d["embeddings"], d["labels"]
