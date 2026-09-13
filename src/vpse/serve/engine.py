"""Search engine over a bundle. Dependencies: numpy, PIL, onnxruntime only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# A solid-colour or near-blank query has no content to match on, yet embeds close
# to the plain-background component shared by many product photos (a solid white
# query scores ~0.87 against the catalog). The similarity gate cannot reject it
# because nothing about it is dissimilar. Refuse such inputs on pixel statistics
# before embedding. Threshold in 0-255 grayscale units; real photos sit far above.
MIN_PIXEL_STD = 5.0


def pixel_std(img: Image.Image) -> float:
    g = img.convert("L")
    g.thumbnail((64, 64))
    return float(np.asarray(g, dtype=np.float32).std())


def preprocess(img: Image.Image, image_size: int = 224) -> np.ndarray:
    """Resize(256) -> CenterCrop(224) -> normalize, matching torchvision's
    integer arithmetic exactly so ONNX serving sees the same pixels as eval."""
    img = img.convert("RGB")
    size = int(image_size * 256 / 224)
    w, h = img.size
    # torchvision Resize(int): short side -> size, long side floor-scaled
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_h, new_w = size, int(size * w / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    # torchvision CenterCrop: offsets rounded, not floored
    left = int(round((new_w - image_size) / 2.0))
    top = int(round((new_h - image_size) / 2.0))
    img = img.crop((left, top, left + image_size, top + image_size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None]  # [1, 3, H, W]


def thumb_rel(i: int) -> str:
    """Thumbnails are sharded 1,000 per folder: hosts such as HuggingFace cap a
    directory at 10,000 files, and a 2,000-product demo catalog exceeds that."""
    return f"thumbs/{i // 1000:03d}/{i}.jpg"


class SearchEngine:
    def __init__(self, bundle_dir: Path):
        import onnxruntime as ort
        self.dir = Path(bundle_dir)
        self.sess = ort.InferenceSession(str(self.dir / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.gallery = np.load(self.dir / "gallery.npz")["embeddings"].astype("float32")
        self.catalog = pd.read_csv(self.dir / "catalog.csv")
        self.gate = json.loads((self.dir / "gate.json").read_text())
        self.threshold = float(self.gate["threshold"])
        self.views = tuple(self.gate.get("views", ("identity",)))
        self.has_thumbs = (self.dir / "thumbs").is_dir()
        assert len(self.gallery) == len(self.catalog)

    def embed(self, img: Image.Image) -> np.ndarray:
        x = preprocess(img)
        acc = None
        for v in self.views:
            xv = x[..., ::-1] if v == "hflip" else x
            e = self.sess.run(["embedding"], {"image": np.ascontiguousarray(xv)})[0][0]
            acc = e if acc is None else acc + e
        return acc / np.linalg.norm(acc)

    def search(self, img: Image.Image, k: int = 5) -> dict:
        std = pixel_std(img)
        blank = std < MIN_PIXEL_STD
        q = self.embed(img)
        sims = self.gallery @ q
        order = np.argsort(-sims)
        top1 = float(sims[order[0]])
        accepted = (top1 >= self.threshold) and not blank

        # top-k distinct products, best image for each
        results, seen = [], set()
        for i in order:
            pid = int(self.catalog.product_id[i])
            if pid in seen:
                continue
            seen.add(pid)
            results.append({
                "rank": len(results) + 1,
                "similarity": float(sims[i]),
                "product_id": pid,
                "category": str(self.catalog.category[i]),
                "image": str(self.catalog.path[i]),
                "gallery_index": int(i),
            })
            if len(results) == k:
                break
        reason = "blank_image" if blank else ("below_threshold" if not accepted else None)
        return {"match": bool(accepted), "confidence": top1, "threshold": self.threshold,
                "refusal_reason": reason, "results": results}

    def thumb_path(self, gallery_index: int) -> Path | None:
        p = self.dir / thumb_rel(gallery_index)
        if p.exists():
            return p
        flat = self.dir / "thumbs" / f"{gallery_index}.jpg"   # pre-sharding bundles
        return flat if flat.exists() else None
