"""Phase 3 error analysis: where does top-1 go wrong, and how?

SOP labels each image with a product (class_id) and a category
(super_class_id). Splitting top-1 errors by whether the retrieved item shares
the query's category separates the two failure modes:

  look-alike  - right kind of thing, wrong product (a different black saddle)
  off-target  - wrong kind of thing entirely

The look-alike share is the number Phase 3 is trying to move.
"""
import numpy as np


def top1_errors(nbrs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Boolean mask of queries whose nearest neighbour is the wrong product."""
    return labels[nbrs[:, 0]] != labels


def error_breakdown(nbrs: np.ndarray, labels: np.ndarray,
                    super_labels: np.ndarray) -> dict:
    wrong = top1_errors(nbrs, labels)
    same_cat = super_labels[nbrs[:, 0]] == super_labels
    lookalike = wrong & same_cat
    n_err = int(wrong.sum())
    return {
        "n_queries": int(len(labels)),
        "R@1": float(1.0 - wrong.mean()),
        "n_top1_errors": n_err,
        "lookalike_errors": int(lookalike.sum()),
        "offtarget_errors": int((wrong & ~same_cat).sum()),
        "lookalike_share_of_errors": float(lookalike.sum() / n_err) if n_err else 0.0,
    }


def hard_queries(nbrs: np.ndarray, labels: np.ndarray,
                 super_labels: np.ndarray) -> np.ndarray:
    """Indices of look-alike failures: top-1 is the wrong product, same category."""
    wrong = top1_errors(nbrs, labels)
    same_cat = super_labels[nbrs[:, 0]] == super_labels
    return np.flatnonzero(wrong & same_cat)


def recall_on_subset(nbrs: np.ndarray, labels: np.ndarray, subset: np.ndarray,
                     ks=(1, 5, 10)) -> dict:
    """Recall@k restricted to the given query indices."""
    if len(subset) == 0:
        return {f"R@{k}": float("nan") for k in ks}
    hits = labels[nbrs[subset]] == labels[subset][:, None]
    return {f"R@{k}": float(hits[:, :k].any(axis=1).mean()) for k in ks}
