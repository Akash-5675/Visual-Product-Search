"""Retrieval metrics for SOP-style evaluation.

Standard SOP protocol: every test image is a query against all *other* test
images (leave-one-out gallery). A hit at k means any of the top-k neighbors
shares the query's product class.

The neighbor search is brute-force. FAISS handles it on CPU; when a GPU is
available a chunked matmul is roughly two orders of magnitude faster, which
matters because training evaluates repeatedly.
"""
import faiss
import numpy as np
import torch


def _neighbors_faiss(embeddings: np.ndarray, k_max: int) -> np.ndarray:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    _, idx = index.search(embeddings.astype("float32"), k_max + 1)
    out = np.empty((len(idx), k_max), dtype=np.int64)
    for i, row in enumerate(idx):
        out[i] = row[row != i][:k_max]  # drop self-match
    return out


def _neighbors_torch(embeddings: np.ndarray, k_max: int, device: str,
                     chunk: int = 2048) -> np.ndarray:
    gallery = torch.from_numpy(embeddings.astype("float32")).to(device)
    out = torch.empty((len(gallery), k_max), dtype=torch.int64, device=device)
    for start in range(0, len(gallery), chunk):
        stop = min(start + chunk, len(gallery))
        sims = gallery[start:stop] @ gallery.T
        # mask each query's self-match rather than over-fetching and filtering
        rows = torch.arange(stop - start, device=device)
        sims[rows, torch.arange(start, stop, device=device)] = float("-inf")
        out[start:stop] = sims.topk(k_max, dim=1).indices
    return out.cpu().numpy()


def leave_one_out_neighbors(embeddings: np.ndarray, k_max: int,
                            device: str | None = None) -> np.ndarray:
    """Top-k_max neighbor indices for each row, self excluded. [N, k_max]"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return _neighbors_faiss(embeddings, k_max)
    return _neighbors_torch(embeddings, k_max, device)


def recall_at_k(neighbors: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    neighbor_labels = labels[neighbors]
    hits = neighbor_labels == labels[:, None]
    return {f"R@{k}": float(hits[:, :k].any(axis=1).mean()) for k in ks}


def mean_average_precision(neighbors: np.ndarray, labels: np.ndarray,
                           k: int = 100) -> float:
    """mAP@k over queries that have at least one relevant item."""
    neighbor_labels = labels[neighbors[:, :k]]
    rel = (neighbor_labels == labels[:, None]).astype(np.float32)
    # per-class counts (minus the query itself) for the normalizer
    counts = np.bincount(labels)
    n_rel = np.minimum(counts[labels] - 1, k)

    cum_hits = np.cumsum(rel, axis=1)
    ranks = np.arange(1, k + 1)
    precision_at_hit = (cum_hits / ranks) * rel
    valid = n_rel > 0
    ap = precision_at_hit[valid].sum(axis=1) / n_rel[valid]
    return float(ap.mean())


def evaluate(embeddings: np.ndarray, labels: np.ndarray, ks=(1, 5, 10),
             device: str | None = None) -> dict:
    neighbors = leave_one_out_neighbors(embeddings, k_max=max(max(ks), 100),
                                        device=device)
    metrics = recall_at_k(neighbors, labels, ks)
    metrics["mAP@100"] = mean_average_precision(neighbors, labels)
    return metrics
