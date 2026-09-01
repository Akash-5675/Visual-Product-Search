from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # paths
    data_root: Path = Path("data/Stanford_Online_Products")
    cache_dir: Path = Path("cache")        # cached embeddings
    results_dir: Path = Path("results")

    # model
    embedding_dim: int = 512
    freeze_backbone: bool = False          # True for the Phase 1 baseline

    # data
    image_size: int = 224
    num_workers: int = 2

    # training (Phase 2)
    loss: str = "triplet_hard"             # triplet_random | triplet_hard | arcface
    epochs: int = 30
    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    triplet_margin: float = 0.2
    arcface_scale: float = 30.0
    arcface_margin: float = 0.3
    # PK batches: P classes x K images. Bigger batch => better in-batch mining.
    batch_p: int = 16
    batch_k: int = 4

    # retrieval / eval
    recall_ks: tuple = (1, 5, 10)
    # a full eval re-embeds the whole test split; don't do it every epoch
    eval_every: int = 3

    device: str = "cuda"

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.cache_dir = Path(self.cache_dir)
        self.results_dir = Path(self.results_dir)
