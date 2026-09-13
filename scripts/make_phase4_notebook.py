"""Generate the self-contained Phase 4 (OOD refusal) Kaggle notebook.

Loads the trained checkpoint; no retraining. ~15 min on a T4.

    python scripts/make_phase4_notebook.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kaggle_notebook import code, collect_sources, md  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_phase4.ipynb"

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
assert ckpts, 'attach the Phase 2 run output (Add Input -> Your Work) for best_triplet_hard.pt'
CKPT = next((c for c in ckpts if 'triplet_hard' in c), ckpts[0])

# Foreign-image OOD set: prefer ImageNet-mini's val split; otherwise any image
# folder under /kaggle/input that is not SOP and not a notebook output.
cand = glob.glob('/kaggle/input/**/imagenet-mini/val', recursive=True)
OOD_DIR = Path(cand[0]) if cand else None
if OOD_DIR is None:
    for r in sorted(Path('/kaggle/input').glob('*')):
        if DATA_ROOT.is_relative_to(r) or list(r.rglob('best_*.pt')):
            continue
        if any(p.suffix.lower() in ('.jpg', '.jpeg', '.png') for p in r.rglob('*')):
            OOD_DIR = r
            break

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('data  :', DATA_ROOT)
print('ckpt  :', CKPT)
print('ood   :', OOD_DIR or '(none attached -- foreign-image experiment will be skipped)')
print('device:', DEVICE)"""

EMBED = """from vpse.data.sop import SOPDataset, eval_transform
from vpse.models.embedder import Embedder
from vpse.retrieval.tta import embed_dataset_tta

test_ds = SOPDataset(DATA_ROOT, 'test', eval_transform())
model = Embedder(embedding_dim=512)
missing, unexpected = model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)
assert not missing and not unexpected, (missing, unexpected)

# hflip TTA -- the Phase 3 winner, and what the served model will use
VIEWS = ('identity', 'hflip')
embs, labels = embed_dataset_tta(model, test_ds, DEVICE, views=VIEWS,
                                 batch_size=256, num_workers=4)
supers = test_ds.super_labels
CATS = {int(c): n for c, n in zip(test_ds.df.super_class_id, test_ds.df.path.str.split('/').str[0])}
print(f'{len(embs):,} test embeddings; categories: {sorted(CATS.values())}')"""

HELDOUT = """import numpy as np
from vpse.ood.gate import auroc, operating_points, similarity_scores
from vpse.ood.protocol import heldout_category_split
from vpse.ood.plots import plot_refusal
from IPython.display import Image as IPyImage, display

RESULTS = Path('/kaggle/working/results/ood'); RESULTS.mkdir(parents=True, exist_ok=True)
cats = sorted(set(supers.tolist()))
folds = [tuple(cats[i::4]) for i in range(4)]          # 4 disjoint folds of 3 categories

heldout = {'folds': [], 'top1': {}, 'top3': {}}
all_in, all_ood = {1: [], 3: []}, {1: [], 3: []}
for f, hold in enumerate(folds):
    g_idx, o_idx = heldout_category_split(supers, hold)
    G, O = embs[g_idx], embs[o_idx]
    row = {'fold': f, 'holdout': [CATS[c] for c in hold],
           'n_gallery': int(len(g_idx)), 'n_ood': int(len(o_idx))}
    for k in (1, 3):
        s_in = similarity_scores(G, G, k=k, exclude_self=True, device=DEVICE)
        s_ood = similarity_scores(O, G, k=k, device=DEVICE)
        all_in[k].append(s_in); all_ood[k].append(s_ood)
        row[f'auroc_top{k}'] = auroc(s_in, s_ood)
    heldout['folds'].append(row)
    print(f"fold {f}  hold out {row['holdout']}  "
          f"AUROC top1 {row['auroc_top1']:.3f}  top3 {row['auroc_top3']:.3f}")

for k in (1, 3):
    s_in, s_ood = np.concatenate(all_in[k]), np.concatenate(all_ood[k])
    heldout[f'top{k}'] = {'auroc': auroc(s_in, s_ood),
                          'operating_points': operating_points(s_in, s_ood)}
print(f"pooled AUROC: top1 {heldout['top1']['auroc']:.3f}  top3 {heldout['top3']['auroc']:.3f}")
print('operating points (top1):')
for r in heldout['top1']['operating_points']:
    print(f"  budget {r['valid_refused_budget']:.0%}: refuse {r['ood_refused']:.1%} of OOD "
          f"at threshold {r['threshold']:.3f}")

s_in, s_ood = np.concatenate(all_in[1]), np.concatenate(all_ood[1])
p = plot_refusal(s_in, s_ood, RESULTS / 'heldout_categories.png',
                 'Out-of-catalog = held-out SOP categories (products the catalog does not carry)')
(RESULTS / 'heldout_categories.json').write_text(json.dumps(heldout, indent=2))
display(IPyImage(str(p)))"""

FOREIGN = """foreign = None
if OOD_DIR is not None:
    from vpse.ood.data import ImageFolderFlat, list_images
    files = list_images(OOD_DIR, limit=5000)
    print(f'{len(files):,} foreign images from {OOD_DIR}')
    ood_ds = ImageFolderFlat(files, eval_transform())
    ood_embs, _ = embed_dataset_tta(model, ood_ds, DEVICE, views=VIEWS,
                                    batch_size=256, num_workers=4)

    foreign = {'source': str(OOD_DIR), 'n_ood': len(files), 'n_gallery': int(len(embs))}
    for k in (1, 3):
        s_in = similarity_scores(embs, embs, k=k, exclude_self=True, device=DEVICE)
        s_ood = similarity_scores(ood_embs, embs, k=k, device=DEVICE)
        foreign[f'top{k}'] = {'auroc': auroc(s_in, s_ood),
                              'operating_points': operating_points(s_in, s_ood)}
        if k == 1:
            f_in, f_ood = s_in, s_ood
    print(f"AUROC: top1 {foreign['top1']['auroc']:.3f}  top3 {foreign['top3']['auroc']:.3f}")
    print('operating points (top1):')
    for r in foreign['top1']['operating_points']:
        print(f"  budget {r['valid_refused_budget']:.0%}: refuse {r['ood_refused']:.1%} of OOD "
              f"at threshold {r['threshold']:.3f}")
    p = plot_refusal(f_in, f_ood, RESULTS / 'foreign_images.png',
                     'Out-of-catalog = foreign images (ImageNet-mini val)')
    (RESULTS / 'foreign_images.json').write_text(json.dumps(foreign, indent=2))
    display(IPyImage(str(p)))
else:
    print('skipped: no foreign-image dataset attached')"""

SUMMARY = """chosen_budget = 0.05
pick = heldout['top1']['operating_points']
row = next(r for r in pick if r['valid_refused_budget'] == chosen_budget)
gate = {'score': 'top1_cosine', 'views': list(VIEWS), 'threshold': row['threshold'],
        'calibrated_on': 'heldout_categories', 'valid_refused_budget': chosen_budget,
        'expected_valid_refused': row['valid_refused'], 'expected_ood_refused': row['ood_refused']}
if foreign:
    gate['foreign_refused_at_this_threshold'] = float((f_ood < row['threshold']).mean())
(RESULTS / 'gate.json').write_text(json.dumps(gate, indent=2))
print(json.dumps(gate, indent=2))
print('\\nThe threshold is calibrated on the HARD case (held-out product categories) at a 5%')
print('false-refusal budget. Foreign images are easier, so the same threshold refuses more of them.')"""


def build() -> dict:
    payload = json.dumps(collect_sources(), indent=0)
    assert "'''" not in payload

    cells = [
        md("""# Phase 4 — the "no match" refusal layer

A plain retrieval system answers every query, including a photo of a dog. This
phase adds a gate: score each query by its similarity to the nearest catalog
item, and refuse when the score is too low. The threshold is reported as a
tradeoff — **% of out-of-catalog queries correctly refused** against **% of
valid queries wrongly refused** — not as a single accuracy number.

Two kinds of out-of-catalog query, from hard to easy:

1. **Held-out SOP categories.** Drop 3 of the 12 categories from the gallery;
   their images are products the catalog doesn't carry, photographed the same
   way. Needs no extra data. Done as 4 folds so every category is held out once.
2. **Foreign images** (ImageNet-mini). Mostly not products at all. Optional —
   attach [ifigotin/imagenetmini-1000](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
   to run it.

**Setup:** SOP dataset + the Phase 2 `triplet_hard` output (*Add Input → Your
Work*) + optionally ImageNet-mini. Accelerator: GPU. ~15 min."""),
        code("!pip install -q faiss-cpu"),
        code(SETUP.format(payload=payload)),
        code(LOCATE),
        md("""## Embed the catalog

Uses the Phase 3 winner (hflip TTA) so the scores match what the served model
will produce."""),
        code(EMBED),
        md("""## Experiment 1 — held-out categories (hard)

For each fold: gallery = 9 categories, out-of-catalog queries = the other 3.
In-catalog scores are leave-one-out within the gallery. AUROC summarises the
whole curve; the operating points say what a fixed false-refusal budget buys."""),
        code(HELDOUT),
        md("""## Experiment 2 — foreign images (easy)

Gallery = the full test split; queries = ImageNet-mini validation images. Note
ImageNet has toasters, mugs and bicycles too, so this is a mixture of clearly
unrelated and category-overlapping queries."""),
        code(FOREIGN),
        md("""## Pick the gate

Calibrate on the hard case at a 5% false-refusal budget and record the
threshold. That single number is what the API in Phase 5 will apply."""),
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
