"""OOD refusal gate (Phase 4).

A retrieval system always returns *something*. The gate looks at how close the
nearest gallery item actually is and refuses to answer when that similarity
falls below a threshold. The threshold is chosen from the operating curve over
two query sets: in-catalog (should be answered) and out-of-catalog (should be
refused).

Score = cosine similarity to the nearest gallery item (top-1). Optionally the
mean over the top-k, which is a little more robust to a single lucky neighbour.
"""
from dataclasses import dataclass

import numpy as np

from vpse.retrieval.eval import neighbors


def similarity_scores(queries: np.ndarray, gallery: np.ndarray, k: int = 1,
                      exclude_self: bool = False, device: str | None = None) -> np.ndarray:
    """Per-query confidence: mean cosine similarity to the top-k gallery items."""
    nbrs = neighbors(queries, gallery, k, exclude_self=exclude_self, device=device)
    sims = np.einsum("qd,qkd->qk", queries, gallery[nbrs])
    return sims.mean(axis=1)


@dataclass
class OODGate:
    threshold: float

    def accept(self, score: np.ndarray) -> np.ndarray:
        """True where the query should be answered, False where refused."""
        return score >= self.threshold


def refusal_curve(in_scores: np.ndarray, ood_scores: np.ndarray,
                  n_points: int = 400) -> dict:
    """Sweep thresholds over both distributions.

    ood_refused   -- fraction of out-of-catalog queries correctly refused (TPR)
    valid_refused -- fraction of in-catalog queries wrongly refused (FPR)
    """
    lo = min(in_scores.min(), ood_scores.min())
    hi = max(in_scores.max(), ood_scores.max())
    thresholds = np.linspace(lo, hi, n_points)
    ood_refused = np.array([(ood_scores < t).mean() for t in thresholds])
    valid_refused = np.array([(in_scores < t).mean() for t in thresholds])
    return {"threshold": thresholds, "ood_refused": ood_refused,
            "valid_refused": valid_refused}


def auroc(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """P(in-catalog score > out-of-catalog score); 0.5 = useless, 1.0 = perfect.

    Threshold-free, so it summarises the whole curve in one number.
    """
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(in_scores)), np.zeros(len(ood_scores))]
    return float(roc_auc_score(y, np.r_[in_scores, ood_scores]))


def pick_threshold(curve: dict, max_valid_refused: float = 0.05) -> float:
    """Highest threshold that keeps false refusals under the budget."""
    ok = curve["valid_refused"] <= max_valid_refused
    return float(curve["threshold"][ok][-1]) if ok.any() else float(curve["threshold"][0])


def operating_points(in_scores: np.ndarray, ood_scores: np.ndarray,
                     budgets=(0.01, 0.02, 0.05, 0.10)) -> list[dict]:
    """For each false-refusal budget: threshold and the OOD refusal it buys."""
    curve = refusal_curve(in_scores, ood_scores)
    rows = []
    for b in budgets:
        t = pick_threshold(curve, b)
        rows.append({"valid_refused_budget": b, "threshold": t,
                     "valid_refused": float((in_scores < t).mean()),
                     "ood_refused": float((ood_scores < t).mean())})
    return rows
