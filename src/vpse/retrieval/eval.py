"""Retrieval metrics for SOP-style evaluation.

Standard SOP protocol: every test image is a query against all *other* test
images (leave-one-out gallery). A hit at k means any of the top-k neighbors
shares the query's product class.
"""
import numpy as np
import faiss


def leave_one_out_neighbors(embeddings: np.ndarray, k_max: int) -> np.ndarray:
    """Top-k_max neighbor indices for each row, self excluded. [N, k_max]"""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    _, idx = index.search(embeddings.astype("float32"), k_max + 1)
    # drop self-match (guard against ties putting self not-first)
    out = np.empty((len(idx), k_max), dtype=np.int64)
    for i, row in enumerate(idx):
        out[i] = row[row != i][:k_max]
    return out


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


def evaluate(embeddings: np.ndarray, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    neighbors = leave_one_out_neighbors(embeddings, k_max=max(max(ks), 100))
    metrics = recall_at_k(neighbors, labels, ks)
    metrics["mAP@100"] = mean_average_precision(neighbors, labels)
    return metrics
