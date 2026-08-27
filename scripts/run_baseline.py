"""Phase 1 baseline: frozen pretrained ResNet50, no training.

Embeds the SOP test split, runs leave-one-out retrieval, prints Recall@k / mAP,
and saves a grid of query -> top-5 results.

    python scripts/run_baseline.py --data-root data/Stanford_Online_Products
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vpse.data.sop import SOPDataset, eval_transform
from vpse.models.embedder import Embedder
from vpse.retrieval.eval import evaluate, leave_one_out_neighbors
from vpse.retrieval.index import embed_dataset, load_embeddings, save_embeddings


def result_grid(ds: SOPDataset, neighbors: np.ndarray, labels: np.ndarray,
                out_path: Path, n_queries: int = 6, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(ds), n_queries, replace=False)
    fig, axes = plt.subplots(n_queries, k + 1, figsize=(2.2 * (k + 1), 2.4 * n_queries))
    for r, q in enumerate(queries):
        row_imgs = [q] + neighbors[q, :k].tolist()
        for c, idx in enumerate(row_imgs):
            ax = axes[r, c]
            img = Image.open(ds.data_root / ds.df.iloc[idx]["path"]).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            if c == 0:
                ax.set_title("query", fontsize=9)
            else:
                hit = labels[idx] == labels[q]
                ax.set_title("match" if hit else "wrong", fontsize=9,
                             color="green" if hit else "red")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"saved result grid -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/Stanford_Online_Products")
    ap.add_argument("--cache", default="cache/baseline_test.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SOPDataset(Path(args.data_root), "test", eval_transform())

    cache = Path(args.cache)
    if cache.exists():
        embs, labels = load_embeddings(cache)
        print(f"loaded cached embeddings {embs.shape} from {cache}")
    else:
        model = Embedder(freeze_backbone=True, use_head=False)  # raw 2048-d pool5
        embs, labels = embed_dataset(model, ds, device, batch_size=args.batch_size)
        save_embeddings(cache, embs, labels)

    metrics = evaluate(embs, labels)
    print(json.dumps(metrics, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/baseline.json").write_text(json.dumps(metrics, indent=2))

    neighbors = leave_one_out_neighbors(embs, k_max=5)
    result_grid(ds, neighbors, labels, Path("results/baseline_grid.png"))


if __name__ == "__main__":
    main()
