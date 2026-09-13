"""Figures for the refusal layer."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vpse.ood.gate import auroc, refusal_curve


def plot_refusal(in_scores: np.ndarray, ood_scores: np.ndarray, out_path: Path,
                 title: str, marks=(0.01, 0.05, 0.10)) -> Path:
    """Two panels: score distributions, and the refusal operating curve."""
    curve = refusal_curve(in_scores, ood_scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.linspace(min(in_scores.min(), ood_scores.min()), 1.0, 60)
    ax1.hist(in_scores, bins=bins, alpha=0.6, density=True, label="in-catalog")
    ax1.hist(ood_scores, bins=bins, alpha=0.6, density=True, label="out-of-catalog")
    ax1.set_xlabel("similarity to nearest gallery item")
    ax1.set_ylabel("density")
    ax1.set_title("score distributions")
    ax1.legend()

    ax2.plot(curve["valid_refused"] * 100, curve["ood_refused"] * 100, lw=2)
    for b in marks:
        ok = curve["valid_refused"] <= b
        if ok.any():
            i = np.flatnonzero(ok)[-1]
            x, y = curve["valid_refused"][i] * 100, curve["ood_refused"][i] * 100
            ax2.plot(x, y, "o", color="black")
            ax2.annotate(f"{y:.0f}% refused @ {b:.0%} budget", (x, y),
                         textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax2.set_xlabel("valid queries wrongly refused (%)")
    ax2.set_ylabel("garbage queries correctly refused (%)")
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    ax2.set_title(f"refusal operating curve  (AUROC {auroc(in_scores, ood_scores):.3f})")

    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
