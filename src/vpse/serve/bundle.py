"""Build a self-contained serving bundle.

    bundle/
      model.onnx        embedding network
      gallery.npz       embeddings float16 [N, d]
      catalog.csv       one row per gallery image: product id, category, path
      gate.json         refusal threshold + how it was chosen
      thumbs/<i>.jpg    optional thumbnails (demo bundles)

The engine (serve/engine.py) needs only numpy, PIL and onnxruntime -- no torch.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def degenerate_images(df: pd.DataFrame, data_root: Path, min_std: float = 3.0) -> np.ndarray:
    """Row indices of near-blank images (SOP has at least one all-black listing photo).

    Left in the gallery they act as universal attractors for blank queries -- a
    solid-black upload scored a perfect 1.000 against one -- so drop them.
    """
    bad = []
    for i, rel in enumerate(df["path"]):
        g = Image.open(Path(data_root) / rel).convert("L")
        g.thumbnail((64, 64))
        if float(np.asarray(g, dtype=np.float32).std()) < min_std:
            bad.append(i)
    return np.asarray(bad, dtype=np.int64)


def build_bundle(out_dir: Path, onnx_path: Path, embeddings: np.ndarray,
                 df: pd.DataFrame, data_root: Path, gate: dict,
                 keep_idx: np.ndarray | None = None, thumbs: bool = False,
                 thumb_size: int = 112, thumb_quality: int = 72,
                 drop_idx: np.ndarray | None = None) -> Path:
    """df must have columns class_id, super_class_id, path (relative to data_root).

    drop_idx: rows to exclude (e.g. from degenerate_images). Computed once by the
    caller so both the full and demo bundles share it.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if keep_idx is None:
        keep_idx = np.arange(len(df))
    keep_idx = np.asarray(keep_idx)
    if drop_idx is not None and len(drop_idx):
        keep_idx = keep_idx[~np.isin(keep_idx, drop_idx)]
    sub = df.iloc[keep_idx].reset_index(drop=True)
    embs = embeddings[keep_idx].astype("float16")

    shutil.copy(onnx_path, out_dir / "model.onnx")
    np.savez_compressed(out_dir / "gallery.npz", embeddings=embs)
    catalog = pd.DataFrame({
        "product_id": sub["class_id"].astype(int),
        "category": sub["path"].str.split("/").str[0].str.replace("_final", ""),
        "path": sub["path"],
    })
    catalog.to_csv(out_dir / "catalog.csv", index=False)
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2))

    if thumbs:
        tdir = out_dir / "thumbs"
        tdir.mkdir()
        for i, rel in enumerate(sub["path"]):
            img = Image.open(Path(data_root) / rel).convert("RGB")
            img.thumbnail((thumb_size, thumb_size))
            img.save(tdir / f"{i}.jpg", quality=thumb_quality, optimize=True)
    return out_dir


def subset_by_products(df: pd.DataFrame, max_products: int, seed: int = 0) -> np.ndarray:
    """Row indices covering a random sample of products (all their images)."""
    rng = np.random.default_rng(seed)
    products = df["class_id"].unique()
    chosen = set(rng.choice(products, min(max_products, len(products)), replace=False))
    return np.flatnonzero(df["class_id"].isin(chosen).to_numpy())
