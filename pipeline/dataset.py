"""
Data pipeline: feature engineering + PyTorch Dataset/DataLoader.

In a production Pinterest setting this would be a Spark job;
here we replicate the logic with pandas for local execution.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from loguru import logger


# ─── Feature Engineering (Spark-equivalent logic) ────────────────────────────

def build_user_features(users: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich user table with behavioral aggregates from interaction history.
    In Spark: df.groupBy("user_id").agg(...)
    """
    agg = (
        interactions.groupby("user_id")
        .agg(
            total_interactions=("pin_id", "count"),
            total_saves=("interaction_type", lambda x: (x == "save").sum()),
            total_clicks=("interaction_type", lambda x: (x == "click").sum()),
            total_closeups=("interaction_type", lambda x: (x == "close_up").sum()),
            avg_weight=("weight", "mean"),
            unique_pins=("pin_id", "nunique"),
        )
        .reset_index()
    )

    users = users.merge(agg, on="user_id", how="left").fillna(0)
    logger.debug(f"User feature matrix: {users.shape}")
    return users


def build_item_features(pins: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich pin table with engagement statistics.
    """
    agg = (
        interactions.groupby("pin_id")
        .agg(
            interaction_count=("user_id", "count"),
            unique_users=("user_id", "nunique"),
            avg_weight=("weight", "mean"),
        )
        .reset_index()
    )

    pins = pins.merge(agg, on="pin_id", how="left").fillna(0)
    logger.debug(f"Pin feature matrix: {pins.shape}")
    return pins


def extract_user_feature_vector(users: pd.DataFrame, num_categories: int) -> np.ndarray:
    """
    Build final user feature vector:
    [interest_cols (30) | behavioral_cols (6) | metadata_cols (4)]
    """
    from data.generate_data import CATEGORIES
    cat_names = CATEGORIES[:num_categories]

    interest_cols = [f"interest_{c}" for c in cat_names]
    behavior_cols = [
        "total_interactions", "total_saves", "total_clicks",
        "total_closeups", "avg_weight", "unique_pins",
    ]
    meta_cols = ["account_age_days", "num_boards", "num_pins_saved", "is_mobile"]

    all_cols = interest_cols + behavior_cols + meta_cols
    feats = users[all_cols].values.astype(np.float32)

    # Normalize behavioral & meta columns (min-max)
    norm_cols_idx = list(range(num_categories, len(all_cols)))
    col_slice = feats[:, norm_cols_idx]
    col_min = col_slice.min(axis=0)
    col_max = col_slice.max(axis=0) + 1e-8
    feats[:, norm_cols_idx] = (col_slice - col_min) / (col_max - col_min)

    return feats


def extract_item_feature_vector(pins: pd.DataFrame, num_categories: int) -> np.ndarray:
    """
    Build final item feature vector:
    [cat_features (30) | visual_embeddings (128) | engagement_cols (3) | meta_cols (3)]
    """
    from data.generate_data import CATEGORIES
    cat_names = CATEGORIES[:num_categories]

    cat_cols = [f"cat_{c}" for c in cat_names]
    visual_cols = [f"visual_{i}" for i in range(128)]
    engage_cols = ["interaction_count", "unique_users", "avg_weight"]
    meta_cols = ["num_saves", "num_clicks", "is_promoted"]

    all_cols = cat_cols + visual_cols + engage_cols + meta_cols
    feats = pins[all_cols].values.astype(np.float32)

    # Normalize engagement & meta
    norm_idx = list(range(num_categories + 128, len(all_cols)))
    col_slice = feats[:, norm_idx]
    col_min = col_slice.min(axis=0)
    col_max = col_slice.max(axis=0) + 1e-8
    feats[:, norm_idx] = (col_slice - col_min) / (col_max - col_min)

    return feats


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class PinterestDataset(Dataset):
    """
    Returns (user_features, pos_item_features) pairs.
    Negative sampling is done in-batch inside the training loop.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_features: np.ndarray,   # shape (num_users, user_feat_dim)
        item_features: np.ndarray,   # shape (num_items, item_feat_dim)
    ):
        self.user_ids = interactions["user_id"].values
        self.pin_ids = interactions["pin_id"].values
        self.weights = interactions["weight"].values.astype(np.float32)
        self.user_features = torch.from_numpy(user_features)
        self.item_features = torch.from_numpy(item_features)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        pid = self.pin_ids[idx]
        return (
            self.user_features[uid],
            self.item_features[pid],
            torch.tensor(self.weights[idx]),
        )


# ─── Pipeline Entry Point ────────────────────────────────────────────────────

def build_dataloaders(cfg: dict):
    """
    Full pipeline: load → feature engineering → dataset → dataloader.
    """
    data_dir = Path(cfg["paths"]["data_dir"])
    num_cat = cfg["data"]["num_categories"]
    batch_size = cfg["training"]["batch_size"]

    logger.info("Loading raw data...")
    users_raw = pd.read_parquet(data_dir / "raw/users.parquet")
    pins_raw = pd.read_parquet(data_dir / "raw/pins.parquet")
    train_df = pd.read_parquet(data_dir / "processed/train.parquet")
    val_df = pd.read_parquet(data_dir / "processed/val.parquet")
    test_df = pd.read_parquet(data_dir / "processed/test.parquet")

    logger.info("Engineering features...")
    all_interactions = pd.concat([train_df, val_df, test_df])
    users = build_user_features(users_raw, all_interactions)
    pins = build_item_features(pins_raw, all_interactions)

    user_feats = extract_user_feature_vector(users, num_cat)
    item_feats = extract_item_feature_vector(pins, num_cat)

    logger.info(
        f"Feature dims → user: {user_feats.shape[1]} | item: {item_feats.shape[1]}"
    )

    train_ds = PinterestDataset(train_df, user_feats, item_feats)
    val_ds = PinterestDataset(val_df, user_feats, item_feats)
    test_ds = PinterestDataset(test_df, user_feats, item_feats)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    meta = {
        "user_feat_dim": user_feats.shape[1],
        "item_feat_dim": item_feats.shape[1],
        "num_users": len(users),
        "num_items": len(pins),
        "user_features": user_feats,
        "item_features": item_feats,
        "user_ids": users["user_id"].values,
        "pin_ids": pins["pin_id"].values,
        "pin_categories": pins["primary_category"].values,
    }

    return train_loader, val_loader, test_loader, meta
