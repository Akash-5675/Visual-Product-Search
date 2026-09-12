"""Re-ranking by alpha-weighted query expansion.

Each query is replaced by a similarity-weighted blend of itself and its top-k
neighbours, then the search is repeated. Neighbours that the model is already
confident about pull the query toward the right region of the space; the alpha
exponent suppresses weakly-matching ones. Cheap (one extra search) and needs no
training, which is why it is the usual first re-ranking step to try.

Default k is deliberately small: SOP products carry roughly five images each, so
a query has at most ~4 true positives. Expanding over a larger neighbourhood
mixes in other products and measurably hurts -- on a 1108-image probe, k=10
dropped R@1 from 0.754 to 0.689 while k<=2 left it unchanged.
"""
import numpy as np

from vpse.retrieval.eval import metrics_from_neighbors, neighbors


def expand_queries(queries: np.ndarray, gallery: np.ndarray, nbrs: np.ndarray,
                   k: int = 2, alpha: float = 3.0) -> np.ndarray:
    """alpha-QE: q' = normalize(q + sum_i sim_i^alpha * g_i) over top-k."""
    top = nbrs[:, :k]                                  # [Nq, k]
    picked = gallery[top]                              # [Nq, k, d]
    sims = np.einsum("qd,qkd->qk", queries, picked)
    weights = np.clip(sims, 0.0, None) ** alpha
    expanded = queries + (weights[..., None] * picked).sum(axis=1)
    norms = np.linalg.norm(expanded, axis=1, keepdims=True)
    return (expanded / np.maximum(norms, 1e-12)).astype("float32")


def rerank_leave_one_out(embeddings: np.ndarray, labels: np.ndarray,
                         k: int = 2, alpha: float = 3.0, ks=(1, 5, 10),
                         device: str | None = None):
    """Returns (metrics, neighbours) after one round of query expansion."""
    k_max = max(max(ks), 100)
    base = neighbors(embeddings, embeddings, k_max, exclude_self=True, device=device)
    expanded = expand_queries(embeddings, embeddings, base, k=k, alpha=alpha)
    new = neighbors(expanded, embeddings, k_max, exclude_self=True, device=device)
    return metrics_from_neighbors(new, labels, ks), new
