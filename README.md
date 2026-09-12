# Visual Product Search Engine with OOD-aware Retrieval

Image → top-k matching products from a catalog, or an explicit **"no match"** refusal when the
query is out-of-catalog. Built on Stanford Online Products (SOP): trained on one set of product
classes, evaluated on *unseen* products — retrieval, not classification.

## Architecture

```
query image → ResNet50 backbone → embedding head (L2-normed, d=512)
            → FAISS index over gallery embeddings → top-k
            → OOD gate (distance threshold) → results OR "no match"
```

## Results

| Model | Loss | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---|---|---|---|---|
| ResNet50 (frozen, no training) | — | 53.09 | 64.76 | 68.99 | 27.78 |
| ResNet50 + head | Triplet (random negatives) | 59.54 | 73.30 | 78.21 | 36.60 |
| ResNet50 + head | Triplet (batch-hard mining) | **72.77** | 84.14 | 87.56 | 50.88 |
| ResNet50 + head | ArcFace (untuned) | 70.37 | 81.51 | 85.04 | 46.83 |

Recall/mAP as percentages, SOP test split (60,502 images, 11,316 unseen products),
leave-one-out protocol. Baseline measured on a Kaggle T4: 3m22s to embed, 1m36s to
evaluate (that search now runs on GPU when one is present). Each training run:
15 epochs, PK batches of 32x4, ~90 min on a T4.

**Batch-hard mining wins.** It gains more at R@1 (+13.2 over random negatives) than at
R@10 (+9.5), which is the signature of fixing confidently-wrong look-alikes rather than
merely reshuffling the tail. ArcFace is normally competitive or better, but was run at
the same 15-epoch budget with default scale/margin and no tuning — treat its row as a
lower bound, not a verdict on the method.

### OOD refusal

Refusal operating curve (% garbage queries correctly refused vs. % valid queries wrongly
refused) lives in `results/ood/` once Phase 4 runs.

## Project layout

```
src/vpse/
  config.py        # all hyperparameters in one dataclass
  data/sop.py      # Stanford Online Products dataset + train/test class split
  data/samplers.py # PK sampler (P classes × K images) for mining-friendly batches
  models/embedder.py   # ResNet50 backbone + embedding head
  losses/triplet.py    # triplet loss: random + batch-hard negative mining
  losses/arcface.py    # ArcFace head + loss
  retrieval/index.py   # FAISS index build / query
  retrieval/eval.py    # Recall@k, mAP
  ood/gate.py          # distance-based refusal threshold + operating curve
  retrieval/tta.py     # test-time augmentation (multi-view embedding average)
  retrieval/rerank.py  # alpha-weighted query expansion
  analysis/errors.py   # look-alike vs off-target error breakdown
  analysis/grids.py    # query -> top-k result figures
  train.py             # training loop (works locally or on Kaggle)
scripts/
  run_baseline.py           # Phase 1: frozen backbone → index → metrics → grid
  make_kaggle_notebook.py   # regenerate the phases 0-2 Kaggle notebook
  make_phase3_notebook.py   # regenerate the Phase 3 Kaggle notebook
notebooks/         # Kaggle-facing notebooks (thin wrappers around src/)
```

## Setup

```bash
pip install -r requirements.txt
```

### Running on Kaggle GPU

`notebooks/kaggle_selfcontained.ipynb` embeds the whole `vpse` package, so it needs no
clone and no uploaded source. On Kaggle: *File → Upload Notebook*, *Add Data* → Stanford
Online Products, *Settings → Accelerator* → GPU, then Run All. Baseline takes ~10 min.

Regenerate it after editing anything under `src/vpse/`:

```bash
python scripts/make_kaggle_notebook.py
```

Locally the same code runs on CPU at roughly 7 images/sec (~2.5 h for the test split).

Dataset: [Stanford Online Products](https://www.kaggle.com/datasets/kwentar/stanford-online-products)
(~120k images, 22,634 products, 12 categories). Place/extract under `data/Stanford_Online_Products/`
so that `Ebay_train.txt` and `Ebay_test.txt` sit in that folder. The official split is built in:
train = class ids 1–11318, test = 11319–22634 (disjoint products).

## Plan / status

- [x] **Phase 0** — setup, data loading, verify split
      (59,551 train / 60,502 test images; 11,318 vs 11,316 products; splits disjoint)
- [x] **Phase 1** — frozen ResNet50 baseline + FAISS retrieval loop + metrics
      (R@1 53.09 / R@5 64.76 / R@10 68.99 / mAP@100 27.78)
- [x] **Phase 2** — metric learning: triplet → batch-hard → ArcFace (results table)
      (best: batch-hard triplet, R@1 72.77)
- [ ] **Phase 3** — look-alike error analysis, harder mining, TTA, re-ranking
- [ ] **Phase 4** — OOD refusal layer + precision/recall tradeoff curve
- [ ] **Phase 5** — ONNX export, FastAPI endpoint, demo, README polish
