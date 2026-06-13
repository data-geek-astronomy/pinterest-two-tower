"""
Training script for the Two-Tower Pinterest retrieval model.

Run:
    python scripts/train.py [--config config.yaml]

Key design choices logged here (interview talking points):
  - In-batch negatives: scales O(B^2) without extra data
  - Learnable temperature: adapts sharpness during training
  - Temporal val split: prevents future leakage
  - Early stopping on Recall@10
"""

import sys
import os
import yaml
import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_data import main as generate_data
from pipeline.dataset import build_dataloaders
from models.two_tower import build_model, InfoNCELoss, HardNegativeMiner
from evaluation.metrics import compute_all_metrics
from inference.faiss_index import build_faiss_index, search_index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--regenerate-data", action="store_true",
                   help="Force re-generate synthetic dataset")
    return p.parse_args()


def encode_all_items(model, item_features: np.ndarray, batch_size: int, device) -> np.ndarray:
    """Encode the entire item corpus into embeddings."""
    model.eval()
    all_embs = []
    t = torch.from_numpy(item_features).to(device)
    with torch.no_grad():
        for start in range(0, len(t), batch_size):
            batch = t[start: start + batch_size]
            emb = model.encode_items(batch)
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def evaluate(model, loader, item_features, cfg, device) -> dict:
    """
    Evaluate retrieval quality using brute-force exact search (val/test).
    At scale this would use the FAISS index.
    """
    model.eval()
    all_user_embs, all_pos_pin_ids = [], []

    with torch.no_grad():
        for user_feats, item_feats, weights in loader:
            user_feats = user_feats.to(device)
            u_emb = model.encode_users(user_feats)
            all_user_embs.append(u_emb.cpu().numpy())

    user_embs = np.vstack(all_user_embs)   # (N_val, D)

    # Gather ground-truth pin_ids from val/test loader dataset
    dataset = loader.dataset
    pos_pin_ids = dataset.pin_ids          # (N_val,)

    # Encode full item corpus
    item_embs = encode_all_items(
        model, item_features, cfg["training"]["batch_size"], device
    )   # (num_items, D)

    # Exact inner product search (embeddings already L2-normalized → cosine)
    scores = user_embs @ item_embs.T      # (N_val, num_items)
    k_max = max(cfg["evaluation"]["k_values"])
    top_k_idx = np.argsort(-scores, axis=1)[:, :k_max]  # (N_val, k_max)

    return compute_all_metrics(top_k_idx, pos_pin_ids, cfg["evaluation"]["k_values"])


def train_epoch(model, loader, optimizer, criterion, miner, device, epoch):
    model.train()
    total_loss = 0.0
    total_hard_negs = 0

    for user_feats, item_feats, weights in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        user_feats = user_feats.to(device)
        item_feats = item_feats.to(device)
        weights = weights.to(device)

        u_emb, i_emb = model(user_feats, item_feats)

        # Log hard negative statistics
        _, n_hard = miner.mine(u_emb.detach(), i_emb.detach())
        total_hard_negs += n_hard

        loss = criterion(u_emb, i_emb, model.temperature, weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader), total_hard_negs


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Setup ────────────────────────────────────────────────────────────────
    Path(cfg["paths"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir = Path(cfg["paths"]["data_dir"])
    if args.regenerate_data or not (data_dir / "raw/users.parquet").exists():
        logger.info("Generating synthetic dataset...")
        generate_data(args.config)

    train_loader, val_loader, test_loader, meta = build_dataloaders(cfg)
    logger.info(
        f"Data loaded | train: {len(train_loader.dataset)} "
        f"| val: {len(val_loader.dataset)} "
        f"| test: {len(test_loader.dataset)}"
    )

    # ── Model + Optimizer ────────────────────────────────────────────────────
    model = build_model(cfg, meta["user_feat_dim"], meta["item_feat_dim"]).to(device)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-5
    )
    criterion = InfoNCELoss()
    miner = HardNegativeMiner(cfg["training"]["hard_negative_ratio"])

    # ── Training Loop ────────────────────────────────────────────────────────
    best_recall = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_recall@10": [], "val_ndcg@10": []}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, n_hard = train_epoch(
            model, train_loader, optimizer, criterion, miner, device, epoch
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, meta["item_features"], cfg, device)

        history["train_loss"].append(train_loss)
        history["val_recall@10"].append(val_metrics.get("recall@10", 0))
        history["val_ndcg@10"].append(val_metrics.get("ndcg@10", 0))

        logger.info(
            f"Epoch {epoch:3d} | loss: {train_loss:.4f} | "
            f"Recall@10: {val_metrics.get('recall@10', 0):.4f} | "
            f"NDCG@10: {val_metrics.get('ndcg@10', 0):.4f} | "
            f"temp: {model.temperature.item():.4f} | "
            f"hard_negs: {n_hard}"
        )

        # ── Early Stopping ────────────────────────────────────────────────
        recall10 = val_metrics.get("recall@10", 0)
        if recall10 > best_recall:
            best_recall = recall10
            patience_counter = 0
            ckpt = Path(cfg["paths"]["model_dir"]) / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "cfg": cfg,
                "meta": {
                    "user_feat_dim": meta["user_feat_dim"],
                    "item_feat_dim": meta["item_feat_dim"],
                },
            }, ckpt)
            logger.info(f"  ✓ New best Recall@10: {best_recall:.4f} → saved")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ── Final Evaluation ─────────────────────────────────────────────────────
    logger.info("Loading best model for test evaluation...")
    ckpt = torch.load(Path(cfg["paths"]["model_dir"]) / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, meta["item_features"], cfg, device)
    logger.info("=" * 60)
    logger.info("TEST RESULTS:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 60)

    # ── Build FAISS Index ─────────────────────────────────────────────────────
    logger.info("Building FAISS index...")
    item_embs = encode_all_items(
        model, meta["item_features"], cfg["training"]["batch_size"], device
    )

    index = build_faiss_index(item_embs, cfg)
    import faiss
    faiss.write_index(index, cfg["paths"]["index_path"])
    np.save(cfg["paths"]["embeddings_path"], item_embs)
    logger.info(f"FAISS index saved → {cfg['paths']['index_path']}")

    # Save training history
    import json
    hist_path = Path(cfg["paths"]["model_dir"]) / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("✅ Training complete.")


if __name__ == "__main__":
    main()
