"""Generate the self-contained Phase 3 Kaggle notebook.

Unlike the Phase 0-2 notebook, this one *loads* the trained checkpoint rather
than retraining, so a full run is ~10 minutes.

    python scripts/make_phase3_notebook.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kaggle_notebook import code, collect_sources, md  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_phase3.ipynb"

SETUP = """import json, pathlib, sys

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

LOCATE = """import glob
from pathlib import Path
import torch

hits = glob.glob('/kaggle/input/**/Ebay_train.txt', recursive=True)
assert hits, 'attach the Stanford Online Products dataset'
DATA_ROOT = Path(hits[0]).parent

ckpts = sorted(glob.glob('/kaggle/input/**/best_*.pt', recursive=True))
if not ckpts:
    print('No checkpoint found under /kaggle/input. Attached inputs:')
    for r in sorted(Path('/kaggle/input').glob('*')):
        print(' ', r.name)
    raise SystemExit('Attach the Phase 2 run output (Add Input -> Your Work) so '
                     'best_triplet_hard.pt is visible.')
CKPT = next((c for c in ckpts if 'triplet_hard' in c), ckpts[0])  # Phase 2 winner

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('data  :', DATA_ROOT)
print('ckpt  :', CKPT)
print('device:', DEVICE)"""

EMBED = """import json
from vpse.data.sop import SOPDataset, eval_transform
from vpse.models.embedder import Embedder
from vpse.retrieval.index import embed_dataset
from vpse.retrieval.eval import leave_one_out_neighbors, metrics_from_neighbors

test_ds = SOPDataset(DATA_ROOT, 'test', eval_transform())

model = Embedder(embedding_dim=512)          # must match the training config
state = torch.load(CKPT, map_location='cpu')
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing and not unexpected, (missing, unexpected)
print('checkpoint loaded cleanly')

embs, labels = embed_dataset(model, test_ds, DEVICE, batch_size=256, num_workers=4)
base_nbrs = leave_one_out_neighbors(embs, k_max=100, device=DEVICE)
base = metrics_from_neighbors(base_nbrs, labels)
print('trained model:', json.dumps(base, indent=2))
print('sanity check: R@1 should land near the Phase 2 value of 0.7277')"""

ERRORS = """from vpse.analysis.errors import error_breakdown, hard_queries, recall_on_subset
from vpse.analysis.grids import retrieval_grid
from IPython.display import Image as IPyImage

breakdown = error_breakdown(base_nbrs, labels, test_ds.super_labels)
print(json.dumps(breakdown, indent=2))

hard = hard_queries(base_nbrs, labels, test_ds.super_labels)
print(f'look-alike failures: {len(hard):,} queries')
print('their recall:', recall_on_subset(base_nbrs, labels, hard))

grid = retrieval_grid(test_ds, base_nbrs, labels, hard[:6],
                      Path('/kaggle/working/lookalike_failures.png'),
                      title='Look-alike failures: top-1 is the wrong product')
IPyImage(str(grid))"""

TTA = """from vpse.retrieval.tta import embed_dataset_tta

tta_results = {}
for views in (('identity', 'hflip'), ('identity', 'hflip', 'zoom')):
    e, _ = embed_dataset_tta(model, test_ds, DEVICE, views=views,
                             batch_size=256, num_workers=4)
    nb = leave_one_out_neighbors(e, k_max=100, device=DEVICE)
    m = metrics_from_neighbors(nb, labels)
    m['hard_R@1'] = recall_on_subset(nb, labels, hard)['R@1']
    tta_results['TTA ' + '+'.join(views)] = (m, nb)
    print('+'.join(views), {k: round(v, 4) for k, v in m.items()})"""

QE = """from vpse.retrieval.eval import neighbors as nn_search
from vpse.retrieval.rerank import expand_queries

header = f"{'k':>3} {'alpha':>6} {'R@1':>8} {'R@5':>8} {'mAP':>8} {'hardR@1':>9}"
print(header)
qe_results = {}
for k in (1, 2, 3, 5):
    for a in (1.0, 3.0):
        exp = expand_queries(embs, embs, base_nbrs, k=k, alpha=a)
        nb = nn_search(exp, embs, 100, exclude_self=True, device=DEVICE)
        m = metrics_from_neighbors(nb, labels)
        m['hard_R@1'] = recall_on_subset(nb, labels, hard)['R@1']
        qe_results[f'alpha-QE k={k} a={a}'] = (m, nb)
        print(f"{k:>3} {a:>6.1f} {m['R@1']:>8.4f} {m['R@5']:>8.4f} "
              f"{m['mAP@100']:>8.4f} {m['hard_R@1']:>9.4f}")"""

SUMMARY = """rows = {'trained (plain)': ({**base, 'hard_R@1': 0.0}, base_nbrs)}
rows.update(tta_results)
rows.update(qe_results)

best = max(rows, key=lambda n: rows[n][0]['R@1'])
print(f"{'variant':<28} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'mAP':>8} {'hardR@1':>9}")
summary = {}
for name, (m, nb) in rows.items():
    summary[name] = {k: float(v) for k, v in m.items()}
    star = ' *' if name == best else ''
    print(f"{name:<28} {m['R@1']:>8.4f} {m['R@5']:>8.4f} {m['R@10']:>8.4f} "
          f"{m['mAP@100']:>8.4f} {m['hard_R@1']:>9.4f}{star}")

print(f'best by R@1: {best}')
Path('/kaggle/working/phase3.json').write_text(json.dumps(
    {'breakdown': breakdown, 'n_hard': int(len(hard)),
     'variants': summary, 'best': best}, indent=2))
print('wrote phase3.json')"""


def build() -> dict:
    payload = json.dumps(collect_sources(), indent=0)
    assert "'''" not in payload, "source contains ''' - would break the here-doc"

    cells = [
        md("""# Phase 3 — the look-alike problem

Loads the **trained** Phase 2 checkpoint (no retraining, ~10 min) and:

1. splits top-1 errors into *look-alike* (right category, wrong product) and
   *off-target*, with examples;
2. tries test-time augmentation and alpha-weighted query expansion;
3. reports before/after Recall@1 **on the look-alike cases specifically**.

**Setup:** attach the SOP dataset as usual, *and* attach the Phase 2 run's
output so the checkpoint is reachable — *Add Input → Your Work →* the notebook
version that trained `triplet_hard`. Accelerator: GPU."""),
        code("!pip install -q faiss-cpu"),
        code(SETUP.format(payload=payload)),
        code(LOCATE),
        md("""## The trained model

Re-embed the test split with the Phase 2 winner. R@1 should reproduce the
number recorded in the README; if it does not, the wrong checkpoint is
attached."""),
        code(EMBED),
        md("""## Where do the errors come from?

SOP labels each image with a product *and* a category. When top-1 is the wrong
product but the **same category**, the model found the right kind of thing and
picked the wrong instance — the look-alike failure. That share is what Phase 3
attacks."""),
        code(ERRORS),
        md("""## Fix attempt 1 — test-time augmentation

Average each embedding over several views. On a small untrained probe this was
roughly neutral overall while recovering some look-alike failures, so the
question is whether it holds up on the trained model."""),
        code(TTA),
        md("""## Fix attempt 2 — re-ranking by query expansion

Blend each query with its top-k neighbours and search again. `k` matters: SOP
products have only ~5 images, so a large neighbourhood pulls in other products.
On the untrained probe, k=10 cost 6.5 points of R@1 — hence the sweep."""),
        code(QE),
        md("""## Before / after

Keep whatever actually won. If nothing beats the plain trained model, that is
the finding — report it rather than burying it."""),
        code(SUMMARY),
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
