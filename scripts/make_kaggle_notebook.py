"""Generate a self-contained Kaggle notebook.

Embeds the whole `vpse` package into one setup cell so the notebook runs on
Kaggle with no GitHub clone and no uploaded dataset of source code. Re-run this
after changing anything under src/vpse/ to regenerate.

    python scripts/make_kaggle_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "kaggle_selfcontained.ipynb"


def collect_sources() -> dict[str, str]:
    files = {}
    for p in sorted(SRC.rglob("*.py")):
        files[p.relative_to(SRC).as_posix()] = p.read_text(encoding="utf-8")
    return files


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def build() -> dict:
    payload = json.dumps(collect_sources(), indent=0)
    assert "'''" not in payload, "source contains ''' - would break the here-doc"

    setup = f"""import json, pathlib, sys

# The vpse package, embedded so this notebook is self-contained.
_FILES = json.loads(r'''
{payload}
''')
_SRC = pathlib.Path('/kaggle/working/src')
for _rel, _text in _FILES.items():
    _p = _SRC / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_text(_text, encoding='utf-8')
sys.path.insert(0, str(_SRC))
print(f'wrote {{len(_FILES)}} modules to {{_SRC}}')"""

    cells = [
        md("""# Visual Product Search Engine — Kaggle runner

Self-contained: the `vpse` package is embedded below, so nothing needs to be
cloned or uploaded. **Before running:** *Add Data* → add the Stanford Online
Products dataset, and set *Settings → Accelerator* to **GPU**.

Phase 1 (frozen ResNet50 baseline) takes ~10 min on a T4. Phase 2 training
cells are at the bottom."""),
        code("!pip install -q faiss-cpu"),
        code(setup),
        code("""import glob
from pathlib import Path

hits = glob.glob('/kaggle/input/**/Ebay_train.txt', recursive=True)
assert hits, 'SOP dataset not attached — use Add Data'
DATA_ROOT = Path(hits[0]).parent
print('data:', DATA_ROOT)

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE, '| gpu:',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"""),
        md("""## Phase 0 — verify the split

Train and test must share no products. That disjointness is what makes this
retrieval on unseen products rather than classification."""),
        code("""from vpse.data.sop import load_split

tr, te = load_split(DATA_ROOT, 'train'), load_split(DATA_ROOT, 'test')
print(f'{len(tr):,} train / {len(te):,} test images')
print(f'{tr.class_id.nunique():,} train / {te.class_id.nunique():,} test products')
print('classes disjoint:', set(tr.class_id).isdisjoint(set(te.class_id)))"""),
        md("""## Phase 1 — untrained baseline

Raw ImageNet pool5 features, no training. Every trained model must beat these
numbers."""),
        code("""import json
from vpse.data.sop import SOPDataset, eval_transform
from vpse.models.embedder import Embedder
from vpse.retrieval.eval import evaluate, leave_one_out_neighbors
from vpse.retrieval.index import embed_dataset, save_embeddings

test_ds = SOPDataset(DATA_ROOT, 'test', eval_transform())
model = Embedder(freeze_backbone=True, use_head=False)   # raw 2048-d features

embs, labels = embed_dataset(model, test_ds, DEVICE, batch_size=256, num_workers=4)
save_embeddings(Path('/kaggle/working/cache/baseline_test.npz'), embs, labels)

baseline = evaluate(embs, labels)
print(json.dumps(baseline, indent=2))
Path('/kaggle/working/baseline.json').write_text(json.dumps(baseline, indent=2))"""),
        code("""%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

neighbors = leave_one_out_neighbors(embs, k_max=5)
rng = np.random.default_rng(0)
queries = rng.choice(len(test_ds), 6, replace=False)

fig, axes = plt.subplots(6, 6, figsize=(13, 14))
for r, q in enumerate(queries):
    for c, idx in enumerate([q] + neighbors[q, :5].tolist()):
        ax = axes[r, c]
        ax.imshow(Image.open(test_ds.data_root / test_ds.df.iloc[idx]['path']).convert('RGB'))
        ax.axis('off')
        if c == 0:
            ax.set_title('query', fontsize=9)
        else:
            hit = labels[idx] == labels[q]
            ax.set_title('match' if hit else 'wrong', fontsize=9,
                         color='green' if hit else 'red')
fig.tight_layout()
fig.savefig('/kaggle/working/baseline_grid.png', dpi=120)
plt.show()"""),
        md("""## Phase 2 — metric learning

Run each loss in turn and record Recall@k. Batch size drives mining quality:
use the largest `batch_p * batch_k` that fits. Best checkpoint and metrics land
in `/kaggle/working/results/`.

Change `loss` to `triplet_random` → `triplet_hard` → `arcface` and rerun to
build the comparison table."""),
        code("""from vpse.config import Config
from vpse.train import main as train_main

cfg = Config(
    data_root=DATA_ROOT,
    results_dir=Path('/kaggle/working/results'),
    loss='triplet_hard',     # triplet_random | triplet_hard | arcface
    epochs=15,
    batch_p=32, batch_k=4,   # 128 images per batch
    num_workers=4,
    device=DEVICE,
)
train_main(cfg)"""),
        md("""## Save your numbers

Download `baseline.json`, `results/*.json`, and `baseline_grid.png` from the
Kaggle output panel — those fill in the README results table."""),
    ]

    return {"cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
                "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
