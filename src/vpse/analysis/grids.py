"""Query -> top-k result grids."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def retrieval_grid(ds, nbrs: np.ndarray, labels: np.ndarray, queries,
                   out_path: Path, k: int = 5, title: str | None = None):
    queries = list(queries)
    fig, axes = plt.subplots(len(queries), k + 1,
                             figsize=(2.2 * (k + 1), 2.4 * len(queries)),
                             squeeze=False)
    for r, q in enumerate(queries):
        for c, idx in enumerate([q] + nbrs[q, :k].tolist()):
            ax = axes[r][c]
            img = Image.open(ds.data_root / ds.df.iloc[idx]["path"]).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            if c == 0:
                ax.set_title("query", fontsize=9)
            else:
                hit = labels[idx] == labels[q]
                ax.set_title("match" if hit else "wrong", fontsize=9,
                             color="green" if hit else "red")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
