"""Drop degenerate (near-blank) gallery images from an existing bundle in place.

Bundles built by the current export notebook already exclude them; this is for
bundles exported before catalog hygiene was added.

    python scripts/clean_bundle.py serve/bundle_full --data data/Stanford_Online_Products
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpse.serve.bundle import degenerate_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--data", required=True, help="dataset root the catalog paths are relative to")
    ap.add_argument("--min-std", type=float, default=3.0)
    ap.add_argument("--std-cache", help="optional: .npy of per-image pixel std, aligned to --std-catalog")
    ap.add_argument("--std-catalog", help="catalog.csv the cache is aligned to")
    args = ap.parse_args()

    b = Path(args.bundle)
    cat = pd.read_csv(b / "catalog.csv")
    embs = np.load(b / "gallery.npz")["embeddings"]
    assert len(cat) == len(embs)

    if args.std_cache:
        # reuse a previous full-catalog scan (~14 min for 60k images) instead of rescanning
        ref = pd.read_csv(args.std_catalog)
        std_by_path = dict(zip(ref.path, np.load(args.std_cache)))
        drop = np.array([i for i, rel in enumerate(cat.path) if std_by_path[rel] < args.min_std],
                        dtype=np.int64)
    else:
        drop = degenerate_images(cat, Path(args.data), min_std=args.min_std)
    if len(drop) == 0:
        print(f"{b}: nothing to drop")
        return
    print(f"{b}: dropping {len(drop)} of {len(cat):,} images:")
    for i in drop:
        print("  ", cat.path[i])

    keep = np.setdiff1d(np.arange(len(cat)), drop)
    cat.iloc[keep].reset_index(drop=True).to_csv(b / "catalog.csv", index=False)
    np.savez_compressed(b / "gallery.npz", embeddings=embs[keep])

    thumbs = b / "thumbs"
    if thumbs.is_dir():
        # thumbnails are named by row index; rebuild the numbering after the drop
        tmp = b / "thumbs_tmp"
        tmp.mkdir()
        for new, old in enumerate(keep):
            shutil.move(thumbs / f"{old}.jpg", tmp / f"{new}.jpg")
        shutil.rmtree(thumbs)
        tmp.rename(thumbs)
    print(f"done: {len(keep):,} images remain")


if __name__ == "__main__":
    main()
