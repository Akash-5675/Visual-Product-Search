"""Stanford Online Products dataset.

The official split files (Ebay_train.txt / Ebay_test.txt) list:
    image_id class_id super_class_id path
Train uses class ids 1..11318, test uses 11319..22634 — products in the test
set are never seen in training, which is what makes this retrieval rather
than classification.
"""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(data_root: Path, split: str) -> pd.DataFrame:
    """split: 'train' or 'test'. Returns df with class_id remapped to 0..C-1."""
    f = Path(data_root) / f"Ebay_{split}.txt"
    df = pd.read_csv(f, sep=" ")
    df.columns = [c.strip() for c in df.columns]
    # contiguous labels for loss heads (e.g. ArcFace needs 0..C-1)
    df["label"] = df["class_id"].astype("category").cat.codes
    return df


def train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SOPDataset(Dataset):
    def __init__(self, data_root: Path, split: str, transform):
        self.data_root = Path(data_root)
        self.df = load_split(data_root, split)
        self.transform = transform
        self.labels = self.df["label"].to_numpy()
        # category (12 of them) -- used by Phase 3 to separate look-alike
        # errors (right category, wrong product) from off-target ones
        self.super_labels = self.df["super_class_id"].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.data_root / row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])
