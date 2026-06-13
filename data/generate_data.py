"""
Synthetic Pinterest-style dataset generator.

Generates:
  - users.parquet   : user profiles with interest vectors
  - pins.parquet    : pin metadata with category embeddings
  - interactions.parquet : save/click/close-up events with timestamps
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

# ─── Pinterest-like Categories ───────────────────────────────────────────────
CATEGORIES = [
    "home_decor", "fashion", "food_recipes", "travel", "fitness",
    "beauty", "art_crafts", "photography", "wedding", "parenting",
    "gardening", "technology", "architecture", "quotes", "hair",
    "tattoos", "cars", "animals", "music", "books",
    "outdoors", "sports", "education", "business", "minimalism",
    "vintage", "diy", "jewelry", "movies", "skincare",
]

INTERACTION_TYPES = ["save", "click", "close_up", "hide"]
INTERACTION_WEIGHTS = [0.35, 0.45, 0.15, 0.05]   # realistic skew


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_users(num_users: int, num_categories: int, seed: int) -> pd.DataFrame:
    """Generate user profiles with latent interest vectors per category."""
    rng = np.random.default_rng(seed)
    cat_names = CATEGORIES[:num_categories]

    # Each user has a sparse interest profile (Dirichlet → sparse via top-k)
    raw_interests = rng.dirichlet(np.ones(num_categories) * 0.5, size=num_users)

    # Sparsify: keep only top-5 categories per user
    sparse = np.zeros_like(raw_interests)
    top_k_idx = np.argsort(raw_interests, axis=1)[:, -5:]
    for i, idx in enumerate(top_k_idx):
        sparse[i, idx] = raw_interests[i, idx]
        sparse[i] /= sparse[i].sum()   # re-normalize

    interest_df = pd.DataFrame(sparse, columns=[f"interest_{c}" for c in cat_names])

    users = pd.DataFrame({
        "user_id": np.arange(num_users),
        "account_age_days": rng.integers(1, 2000, size=num_users),
        "num_boards": rng.integers(1, 50, size=num_users),
        "num_pins_saved": rng.integers(10, 5000, size=num_users),
        "is_mobile": rng.choice([0, 1], size=num_users, p=[0.3, 0.7]),
    })

    return pd.concat([users, interest_df], axis=1)


def generate_pins(num_pins: int, num_categories: int, seed: int) -> pd.DataFrame:
    """Generate pin metadata with category distributions and feature vectors."""
    rng = np.random.default_rng(seed + 1)
    cat_names = CATEGORIES[:num_categories]

    # Each pin belongs primarily to one category but has soft multi-label
    primary_cat = rng.integers(0, num_categories, size=num_pins)
    cat_features = np.zeros((num_pins, num_categories))
    for i, pc in enumerate(primary_cat):
        cat_features[i, pc] = rng.uniform(0.6, 1.0)
        # add noise from 1-2 secondary categories
        sec = rng.choice([c for c in range(num_categories) if c != pc],
                         size=rng.integers(1, 3), replace=False)
        cat_features[i, sec] = rng.uniform(0.05, 0.3, size=len(sec))

    cat_feat_df = pd.DataFrame(cat_features, columns=[f"cat_{c}" for c in cat_names])

    # Visual embedding (128-dim, simulating image encoder output)
    visual_embs = rng.standard_normal((num_pins, 128)).astype(np.float32)
    # Inject category signal: pins in same category cluster together
    for cat_id in range(num_categories):
        mask = primary_cat == cat_id
        centroid = rng.standard_normal(128) * 2
        visual_embs[mask] += centroid

    visual_emb_df = pd.DataFrame(
        visual_embs, columns=[f"visual_{i}" for i in range(128)]
    )

    pins = pd.DataFrame({
        "pin_id": np.arange(num_pins),
        "primary_category": [cat_names[c] for c in primary_cat],
        "primary_category_id": primary_cat,
        "num_saves": rng.integers(0, 100000, size=num_pins),
        "num_clicks": rng.integers(0, 500000, size=num_pins),
        "is_promoted": rng.choice([0, 1], size=num_pins, p=[0.9, 0.1]),
        "has_price": rng.choice([0, 1], size=num_pins, p=[0.7, 0.3]),
        "content_length": rng.integers(10, 500, size=num_pins),
    })

    return pd.concat([pins, cat_feat_df, visual_emb_df], axis=1)


def generate_interactions(
    users: pd.DataFrame,
    pins: pd.DataFrame,
    num_interactions: int,
    num_categories: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate user-pin interactions with interest-driven sampling.
    Users are more likely to interact with pins matching their interests.
    """
    rng = np.random.default_rng(seed + 2)
    cat_names = CATEGORIES[:num_categories]

    num_users = len(users)
    num_pins = len(pins)

    user_interests = users[[f"interest_{c}" for c in cat_names]].values  # (U, C)
    pin_cats = pins[[f"cat_{c}" for c in cat_names]].values              # (P, C)

    # Build affinity matrix in chunks to avoid OOM
    rows, cols, types, ts = [], [], [], []
    chunk = 5000

    logger.info(f"Sampling {num_interactions} interactions...")
    sampled = 0
    while sampled < num_interactions:
        n = min(chunk, num_interactions - sampled)
        uid = rng.integers(0, num_users, size=n)

        # Compute affinity scores for sampled users vs all pins
        scores = user_interests[uid] @ pin_cats.T   # (n, P)
        scores = np.clip(scores, 1e-8, None)
        probs = scores / scores.sum(axis=1, keepdims=True)

        # Sample one pin per user proportional to affinity
        pid = np.array([rng.choice(num_pins, p=p) for p in probs])
        itype = rng.choice(
            INTERACTION_TYPES, size=n, p=INTERACTION_WEIGHTS
        )
        timestamp = rng.integers(
            pd.Timestamp("2023-01-01").value // 10**9,
            pd.Timestamp("2024-12-31").value // 10**9,
            size=n,
        )

        rows.extend(uid.tolist())
        cols.extend(pid.tolist())
        types.extend(itype.tolist())
        ts.extend(timestamp.tolist())
        sampled += n

    interactions = pd.DataFrame({
        "user_id": rows,
        "pin_id": cols,
        "interaction_type": types,
        "timestamp": ts,
    })

    # Assign weights: save=3, click=1, close_up=2, hide=-1
    weight_map = {"save": 3, "click": 1, "close_up": 2, "hide": -1}
    interactions["weight"] = interactions["interaction_type"].map(weight_map)

    # Drop duplicate (user, pin) keeping max weight
    interactions = (
        interactions.sort_values("weight", ascending=False)
        .drop_duplicates(subset=["user_id", "pin_id"], keep="first")
        .reset_index(drop=True)
    )

    # Keep only positive interactions for training
    interactions = interactions[interactions["weight"] > 0].reset_index(drop=True)

    logger.info(f"Generated {len(interactions)} positive interactions")
    return interactions


def train_val_test_split(
    interactions: pd.DataFrame, train: float, val: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: last interaction per user → test, second-last → val, rest → train.
    Mirrors real evaluation protocol for recommendation systems.
    """
    interactions = interactions.sort_values(["user_id", "timestamp"])

    test_idx = interactions.groupby("user_id").tail(1).index
    remaining = interactions.drop(test_idx)
    val_idx = remaining.groupby("user_id").tail(1).index
    train_df = remaining.drop(val_idx).reset_index(drop=True)
    val_df = interactions.loc[val_idx].reset_index(drop=True)
    test_df = interactions.loc[test_idx].reset_index(drop=True)

    logger.info(
        f"Split → train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    dc = cfg["data"]
    out = Path(cfg["paths"]["data_dir"])
    raw = out / "raw"
    proc = out / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    logger.info("Generating users...")
    users = generate_users(dc["num_users"], dc["num_categories"], dc["seed"])
    users.to_parquet(raw / "users.parquet", index=False)
    logger.info(f"  → {len(users)} users saved")

    logger.info("Generating pins...")
    pins = generate_pins(dc["num_pins"], dc["num_categories"], dc["seed"])
    pins.to_parquet(raw / "pins.parquet", index=False)
    logger.info(f"  → {len(pins)} pins saved")

    logger.info("Generating interactions...")
    interactions = generate_interactions(
        users, pins, dc["num_interactions"], dc["num_categories"], dc["seed"]
    )
    interactions.to_parquet(raw / "interactions.parquet", index=False)

    logger.info("Splitting dataset...")
    train_df, val_df, test_df = train_val_test_split(
        interactions, dc["train_split"], dc["val_split"], dc["seed"]
    )
    train_df.to_parquet(proc / "train.parquet", index=False)
    val_df.to_parquet(proc / "val.parquet", index=False)
    test_df.to_parquet(proc / "test.parquet", index=False)

    logger.info("✅ Dataset generation complete.")
    logger.info(f"   Users: {len(users)} | Pins: {len(pins)} | Interactions: {len(interactions)}")


if __name__ == "__main__":
    main()
