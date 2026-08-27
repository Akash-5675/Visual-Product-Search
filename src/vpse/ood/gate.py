"""OOD refusal gate (Phase 4).

Score a query by its similarity to the nearest gallery item; refuse when it
falls below a threshold. The threshold is chosen from the operating curve over
in-catalog vs. out-of-catalog (e.g. random ImageNet) query sets.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class OODGate:
    threshold: float

    def accept(self, top1_similarity: np.ndarray) -> np.ndarray:
        """True where the query should be answered, False where refused."""
        return top1_similarity >= self.threshold


def refusal_curve(in_catalog_sims: np.ndarray, ood_sims: np.ndarray,
                  n_points: int = 200) -> dict:
    """Sweep thresholds over both distributions.

    Returns arrays: threshold, ood_refused (TPR of refusal, higher=better),
    valid_refused (false-refusal rate, lower=better).
    """
    lo = min(in_catalog_sims.min(), ood_sims.min())
    hi = max(in_catalog_sims.max(), ood_sims.max())
    thresholds = np.linspace(lo, hi, n_points)
    ood_refused = np.array([(ood_sims < t).mean() for t in thresholds])
    valid_refused = np.array([(in_catalog_sims < t).mean() for t in thresholds])
    return {"threshold": thresholds, "ood_refused": ood_refused,
            "valid_refused": valid_refused}


def pick_threshold(curve: dict, max_valid_refused: float = 0.05) -> float:
    """Highest threshold that keeps false refusals under the budget."""
    ok = curve["valid_refused"] <= max_valid_refused
    return float(curve["threshold"][ok][-1]) if ok.any() else float(curve["threshold"][0])
