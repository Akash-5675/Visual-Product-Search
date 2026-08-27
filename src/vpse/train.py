"""Phase 2 training loop. Runs locally (small batches) or on Kaggle GPU.

Usage:
    python -m vpse.train --loss triplet_hard --epochs 30
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vpse.config import Config
from vpse.data.samplers import PKSampler
from vpse.data.sop import SOPDataset, eval_transform, train_transform
from vpse.losses.arcface import ArcFaceHead
from vpse.losses.triplet import triplet_batch_hard, triplet_random
from vpse.models.embedder import Embedder
from vpse.retrieval.eval import evaluate
from vpse.retrieval.index import embed_dataset


def main(cfg: Config):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    train_ds = SOPDataset(cfg.data_root, "train", train_transform(cfg.image_size))
    test_ds = SOPDataset(cfg.data_root, "test", eval_transform(cfg.image_size))

    sampler = PKSampler(train_ds.labels, cfg.batch_p, cfg.batch_k)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.num_workers,
                        pin_memory=True)

    model = Embedder(cfg.embedding_dim, freeze_backbone=cfg.freeze_backbone).to(device)
    params = [{"params": model.head.parameters(), "lr": cfg.lr_head},
              {"params": model.backbone.parameters(), "lr": cfg.lr_backbone}]

    arcface = None
    if cfg.loss == "arcface":
        n_classes = int(train_ds.labels.max()) + 1
        arcface = ArcFaceHead(cfg.embedding_dim, n_classes,
                              cfg.arcface_scale, cfg.arcface_margin).to(device)
        params.append({"params": arcface.parameters(), "lr": cfg.lr_head})

    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_r1 = 0.0
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                emb = model(x)
                if cfg.loss == "triplet_random":
                    loss = triplet_random(emb, y, cfg.triplet_margin)
                elif cfg.loss == "triplet_hard":
                    loss = triplet_batch_hard(emb, y, cfg.triplet_margin)
                elif cfg.loss == "arcface":
                    loss = arcface(emb, y)
                else:
                    raise ValueError(cfg.loss)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} "
                      f"loss {running / (step + 1):.4f}")
        sched.step()

        embs, labels = embed_dataset(model, test_ds, device,
                                     num_workers=cfg.num_workers)
        metrics = evaluate(embs, labels, cfg.recall_ks)
        print(f"epoch {epoch}: {metrics}")
        if metrics["R@1"] > best_r1:
            best_r1 = metrics["R@1"]
            torch.save(model.state_dict(), cfg.results_dir / f"best_{cfg.loss}.pt")
            (cfg.results_dir / f"best_{cfg.loss}.json").write_text(
                json.dumps({"epoch": epoch, **metrics}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="triplet_hard",
                    choices=["triplet_random", "triplet_hard", "arcface"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data-root", default="data/Stanford_Online_Products")
    ap.add_argument("--batch-p", type=int, default=16)
    ap.add_argument("--batch-k", type=int, default=4)
    args = ap.parse_args()
    main(Config(loss=args.loss, epochs=args.epochs, data_root=Path(args.data_root),
                batch_p=args.batch_p, batch_k=args.batch_k))
