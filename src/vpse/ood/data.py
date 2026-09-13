"""Out-of-catalog query images: any folder of images, labels are all -1."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: Path, limit: int | None = None, seed: int = 0) -> list[Path]:
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS)
    if limit is not None and len(files) > limit:
        import random
        rng = random.Random(seed)
        files = sorted(rng.sample(files, limit))
    return files


class ImageFolderFlat(Dataset):
    def __init__(self, files: list[Path], transform):
        self.files = list(files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), -1
