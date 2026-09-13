"""Generate the Phase 5 export notebook: checkpoint -> ONNX + gallery + bundles.

Loads the trained checkpoint, embeds the catalog (SOP test split) with hflip
TTA, exports ONNX, checks parity, and writes two zips:

    bundle_full.zip   all 60,502 gallery images, no thumbnails  -> for the API
    bundle_demo.zip   ~2,000 products with thumbnails            -> for HF Spaces

    python scripts/make_phase5_notebook.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kaggle_notebook import code, collect_sources, md  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_phase5_export.ipynb"

GATE = json.loads((ROOT / "results" / "ood" / "gate.json").read_text())

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
assert ckpts, 'attach the Phase 2 triplet_hard output (Add Input -> Your Work)'
CKPT = next((c for c in ckpts if 'triplet_hard' in c), ckpts[0])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('data  :', DATA_ROOT); print('ckpt  :', CKPT); print('device:', DEVICE)"""

EXPORT = """from vpse.models.embedder import Embedder
from vpse.serve.export import export_onnx, check_parity

model = Embedder(embedding_dim=512)
missing, unexpected = model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)
assert not missing and not unexpected, (missing, unexpected)

ONNX = Path('/kaggle/working/model.onnx')
export_onnx(model, ONNX)
diff = check_parity(model, ONNX)
print(f'ONNX exported: {ONNX.stat().st_size/2**20:.1f} MB, torch/onnx max abs diff {diff:.2e}')"""

EMBED = """from vpse.data.sop import SOPDataset, eval_transform
from vpse.retrieval.tta import embed_dataset_tta

test_ds = SOPDataset(DATA_ROOT, 'test', eval_transform())
VIEWS = ('identity', 'hflip')                       # Phase 3 winner
embs, labels = embed_dataset_tta(model, test_ds, DEVICE, views=VIEWS,
                                 batch_size=256, num_workers=4)
print(embs.shape)"""

BUNDLES = f"""import shutil
from vpse.serve.bundle import build_bundle, degenerate_images, subset_by_products
from vpse.serve.engine import SearchEngine

GATE = {json.dumps(GATE)}   # Phase 4 result: calibrated on held-out categories @ 5% budget

# catalog hygiene: SOP contains at least one all-black listing photo, which a
# blank query matches at similarity 1.0. Drop near-blank images from the gallery.
drop = degenerate_images(test_ds.df, DATA_ROOT)
print(f'dropping {{len(drop)}} degenerate catalog images:', [test_ds.df.path[i] for i in drop][:5])

full = build_bundle('/kaggle/working/bundle_full', ONNX, embs, test_ds.df, DATA_ROOT,
                    GATE, thumbs=False, drop_idx=drop)
keep = subset_by_products(test_ds.df, max_products=2000)
demo = build_bundle('/kaggle/working/bundle_demo', ONNX, embs, test_ds.df, DATA_ROOT,
                    GATE, keep_idx=keep, thumbs=True, drop_idx=drop)
print(f'full bundle: {{len(embs):,}} images;  demo bundle: {{len(keep):,}} images / 2,000 products')

# end-to-end check through the engine (onnxruntime, numpy -- no torch)
from PIL import Image
eng = SearchEngine(demo)
q = Image.open(DATA_ROOT / test_ds.df.path[keep[0]])
out = eng.search(q, k=5)
assert out['results'][0]['gallery_index'] == 0, 'query image should retrieve itself first'
print('self-retrieval OK; match =', out['match'], 'confidence =', round(out['confidence'], 3))
blank = eng.search(Image.new('RGB', (256, 256), (0, 0, 0)))
assert not blank['match'] and blank['refusal_reason'] == 'blank_image'
print('solid-black query refused:', blank['refusal_reason'], '| nearest sim now', round(blank['confidence'], 3))

for name in ('bundle_full', 'bundle_demo'):
    z = shutil.make_archive(f'/kaggle/working/{{name}}', 'zip', f'/kaggle/working/{{name}}')
    shutil.rmtree(f'/kaggle/working/{{name}}')
    print(f'{{name}}.zip  {{Path(z).stat().st_size/2**20:.0f}} MB')"""


def build() -> dict:
    payload = json.dumps(collect_sources(), indent=0)
    assert "'''" not in payload
    cells = [
        md("""# Phase 5 — export for serving

Turns the trained checkpoint into deployable artifacts: an ONNX model, the
catalog embeddings, thumbnails for the demo, and the calibrated refusal gate.

**Setup:** SOP dataset + the Phase 2 `triplet_hard` output. GPU. ~10 min.

**Outputs** (download from the Output panel):
- `bundle_full.zip` — every catalog image, no thumbnails → local FastAPI
- `bundle_demo.zip` — 2,000 products with thumbnails → HuggingFace Space"""),
        code("!pip install -q faiss-cpu onnx onnxruntime"),
        code(SETUP.format(payload=payload)),
        code(LOCATE),
        md("## Export ONNX and verify it matches PyTorch"),
        code(EXPORT),
        md("## Embed the catalog (hflip TTA)"),
        code(EMBED),
        md("## Build and zip the bundles"),
        code(BUNDLES),
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
