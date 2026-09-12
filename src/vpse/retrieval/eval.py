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


def _neighbors_faiss(queries: np.ndarray, gallery: np.ndarray, k_max: int,
                     exclude_self: bool) -> np.ndarray:
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery.astype("float32"))
    fetch = k_max + 1 if exclude_self else k_max
    _, idx = index.search(queries.astype("float32"), fetch)
    if not exclude_self:
        return idx.astype(np.int64)
    out = np.empty((len(idx), k_max), dtype=np.int64)
    for i, row in enumerate(idx):
        out[i] = row[row != i][:k_max]
    return out


def _neighbors_torch(queries: np.ndarray, gallery: np.ndarray, k_max: int,
                     device: str, exclude_self: bool, chunk: int = 2048) -> np.ndarray:
    q = torch.from_numpy(queries.astype("float32")).to(device)
    g = torch.from_numpy(gallery.astype("float32")).to(device)
    out = torch.empty((len(q), k_max), dtype=torch.int64, device=device)
    for start in range(0, len(q), chunk):
        stop = min(start + chunk, len(q))
        sims = q[start:stop] @ g.T
        if exclude_self:
            # query i corresponds to gallery row i
            rows = torch.arange(stop - start, device=device)
            sims[rows, torch.arange(start, stop, device=device)] = float("-inf")
        out[start:stop] = sims.topk(k_max, dim=1).indices
    return out.cpu().numpy()


def neighbors(queries: np.ndarray, gallery: np.ndarray, k_max: int,
              exclude_self: bool = False, device: str | None = None) -> np.ndarray:
    """Top-k_max gallery indices per query. [Nq, k_max]

    exclude_self assumes query i is gallery row i (leave-one-out).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return _neighbors_faiss(queries, gallery, k_max, exclude_self)
    return _neighbors_torch(queries, gallery, k_max, device, exclude_self)


def leave_one_out_neighbors(embeddings: np.ndarray, k_max: int,
                            device: str | None = None) -> np.ndarray:
    """Top-k_max neighbor indices for each row, self excluded. [N, k_max]"""
    return neighbors(embeddings, embeddings, k_max, exclude_self=True, device=device)


def recall_at_k(nbrs: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    hits = labels[nbrs] == labels[:, None]
    return {f"R@{k}": float(hits[:, :k].any(axis=1).mean()) for k in ks}


def mean_average_precision(nbrs: np.ndarray, labels: np.ndarray, k: int = 100) -> float:
    """mAP@k over queries that have at least one relevant item."""
    rel = (labels[nbrs[:, :k]] == labels[:, None]).astype(np.float32)
    # per-class counts (minus the query itself) for the normalizer
    counts = np.bincount(labels)
    n_rel = np.minimum(counts[labels] - 1, k)

    cum_hits = np.cumsum(rel, axis=1)
    ranks = np.arange(1, k + 1)
    precision_at_hit = (cum_hits / ranks) * rel
    valid = n_rel > 0
    ap = precision_at_hit[valid].sum(axis=1) / n_rel[valid]
    return float(ap.mean())


def metrics_from_neighbors(nbrs: np.ndarray, labels: np.ndarray,
                           ks=(1, 5, 10)) -> dict:
    m = recall_at_k(nbrs, labels, ks)
    m["mAP@100"] = mean_average_precision(nbrs, labels)
    return m


def evaluate(embeddings: np.ndarray, labels: np.ndarray, ks=(1, 5, 10),
             device: str | None = None) -> dict:
    nbrs = leave_one_out_neighbors(embeddings, k_max=max(max(ks), 100), device=device)
    return metrics_from_neighbors(nbrs, labels, ks)
