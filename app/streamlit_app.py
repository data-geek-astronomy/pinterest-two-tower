"""
Streamlit Demo: Pinterest Two-Tower Retrieval

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import torch
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.two_tower import build_model
from inference.faiss_index import load_faiss_index, search_index

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pinterest Two-Tower Retrieval",
    page_icon="📌",
    layout="wide",
)

CATEGORIES = [
    "home_decor", "fashion", "food_recipes", "travel", "fitness",
    "beauty", "art_crafts", "photography", "wedding", "parenting",
    "gardening", "technology", "architecture", "quotes", "hair",
    "tattoos", "cars", "animals", "music", "books",
    "outdoors", "sports", "education", "business", "minimalism",
    "vintage", "diy", "jewelry", "movies", "skincare",
]

CATEGORY_EMOJIS = {
    "home_decor": "🏠", "fashion": "👗", "food_recipes": "🍳", "travel": "✈️",
    "fitness": "💪", "beauty": "💄", "art_crafts": "🎨", "photography": "📷",
    "wedding": "💍", "parenting": "👶", "gardening": "🌱", "technology": "💻",
    "architecture": "🏛️", "quotes": "💬", "hair": "💇", "tattoos": "🖊️",
    "cars": "🚗", "animals": "🐾", "music": "🎵", "books": "📚",
    "outdoors": "🏔️", "sports": "⚽", "education": "🎓", "business": "💼",
    "minimalism": "⬜", "vintage": "🕰️", "diy": "🔧", "jewelry": "💎",
    "movies": "🎬", "skincare": "🧴",
}


# ─── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    model_path = Path(cfg["paths"]["model_dir"]) / "best_model.pt"
    if not model_path.exists():
        return None, None, None, None, None, cfg

    ckpt = torch.load(model_path, map_location="cpu")
    meta_dims = ckpt["meta"]
    model = build_model(cfg, meta_dims["user_feat_dim"], meta_dims["item_feat_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    index = load_faiss_index(cfg["paths"]["index_path"], nprobe=cfg["faiss"]["nprobe"])
    item_embs = np.load(cfg["paths"]["embeddings_path"])

    # Load user and pin data
    user_df = pd.read_parquet("data/raw/users.parquet")
    pin_df = pd.read_parquet("data/raw/pins.parquet")

    # Load history for training history chart
    hist_path = Path(cfg["paths"]["model_dir"]) / "training_history.json"
    history = json.load(open(hist_path)) if hist_path.exists() else None

    return model, index, item_embs, user_df, pin_df, cfg, history


def build_user_feature(user_row, interactions_agg, cfg):
    """Reconstruct a user feature vector for the demo."""
    num_cat = cfg["data"]["num_categories"]
    cat_names = CATEGORIES[:num_cat]

    interest_vals = [user_row.get(f"interest_{c}", 0.0) for c in cat_names]
    behavior_vals = [
        interactions_agg.get("total_interactions", 0),
        interactions_agg.get("total_saves", 0),
        interactions_agg.get("total_clicks", 0),
        interactions_agg.get("total_closeups", 0),
        interactions_agg.get("avg_weight", 0),
        interactions_agg.get("unique_pins", 0),
    ]
    meta_vals = [
        float(user_row.get("account_age_days", 0)) / 2000,
        float(user_row.get("num_boards", 0)) / 50,
        float(user_row.get("num_pins_saved", 0)) / 5000,
        float(user_row.get("is_mobile", 0)),
    ]

    return np.array(interest_vals + behavior_vals + meta_vals, dtype=np.float32)


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("📌 Pinterest Two-Tower Retrieval Demo")
st.caption("ML Engineer Interview Project · AK · 2024")

result = load_artifacts()
if len(result) == 7:
    model, index, item_embs, user_df, pin_df, cfg, history = result
else:
    model, index, item_embs, user_df, pin_df, cfg = result
    history = None

if model is None:
    st.error(
        "⚠️ No trained model found. Run `python scripts/train.py --regenerate-data` first."
    )
    st.stop()

# ─── Sidebar: controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    top_k = st.slider("Top-K results", 5, 50, 20)
    nprobe = st.slider(
        "FAISS nprobe (speed ↔ recall)",
        1, 50, cfg["faiss"]["nprobe"],
        help="Higher = more accurate but slower. Tradeoff central to ANN retrieval."
    )
    index.nprobe = nprobe

    st.divider()
    st.header("👤 Select User")
    user_id = st.number_input(
        "User ID", min_value=0,
        max_value=cfg["data"]["num_users"] - 1,
        value=42
    )

    st.divider()
    st.header("🎨 Or Custom Interest Profile")
    use_custom = st.checkbox("Build custom user profile")
    custom_interests = {}
    if use_custom:
        st.caption("Adjust interest weights (will sum-normalize):")
        for cat in CATEGORIES[:cfg["data"]["num_categories"]]:
            custom_interests[cat] = st.slider(
                f"{CATEGORY_EMOJIS.get(cat, '')} {cat}", 0.0, 1.0, 0.0, step=0.05
            )

# ─── Main Panel ──────────────────────────────────────────────────────────────
tabs = st.tabs(["🔍 Retrieval", "📊 Training Curves", "🏗️ Architecture"])

# ── Tab 1: Retrieval ─────────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("User Profile")

        if use_custom:
            total = sum(custom_interests.values()) or 1.0
            interests = {k: v / total for k, v in custom_interests.items()}
            user_feat = np.array(
                [interests[c] for c in CATEGORIES[:cfg["data"]["num_categories"]]]
                + [0] * 10,  # zero behavioral features for custom user
                dtype=np.float32,
            )
        else:
            user_row = user_df[user_df["user_id"] == user_id].iloc[0].to_dict()
            user_feat = build_user_feature(user_row, {}, cfg)
            interests = {
                c: user_row.get(f"interest_{c}", 0.0)
                for c in CATEGORIES[:cfg["data"]["num_categories"]]
            }

        # Show top interests
        top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:6]
        st.markdown("**Top interests:**")
        for cat, val in top_interests:
            if val > 0.01:
                emoji = CATEGORY_EMOJIS.get(cat, "")
                st.progress(float(val), text=f"{emoji} {cat}: {val:.2%}")

        # Interest distribution chart
        fig, ax = plt.subplots(figsize=(4, 3))
        cats = [c for c, _ in top_interests]
        vals = [v for _, v in top_interests]
        bars = ax.barh(cats, vals, color="#e60023")
        ax.set_xlabel("Interest weight")
        ax.set_xlim(0, max(vals) * 1.2)
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader(f"Top-{top_k} Retrieved Pins")

        # Encode user + search
        with torch.no_grad():
            u_tensor = torch.from_numpy(user_feat[np.newaxis, :]).to("cpu")
            u_emb = model.encode_users(u_tensor).numpy()

        scores, item_ids = search_index(index, u_emb, top_k)
        scores = scores[0]
        item_ids = item_ids[0]

        # Build results table
        results = []
        for rank, (iid, score) in enumerate(zip(item_ids, scores)):
            if iid < len(pin_df):
                pin_row = pin_df.iloc[int(iid)]
                results.append({
                    "Rank": rank + 1,
                    "Pin ID": int(iid),
                    "Category": f"{CATEGORY_EMOJIS.get(pin_row['primary_category'], '')} {pin_row['primary_category']}",
                    "Similarity": f"{score:.4f}",
                    "Saves": f"{int(pin_row['num_saves']):,}",
                    "Promoted": "✓" if pin_row["is_promoted"] else "",
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Category distribution of retrieved pins
        cats_retrieved = [r["Category"].split(" ", 1)[-1] for r in results]
        cat_counts = pd.Series(cats_retrieved).value_counts()

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        cat_counts.plot(kind="bar", ax=ax2, color="#e60023", alpha=0.85)
        ax2.set_title("Category distribution of retrieved pins")
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)

        # Embedding similarity heatmap (user vs top-5 items)
        st.subheader("Embedding Space Visualization")
        top5_embs = item_embs[item_ids[:5]]
        sim_matrix = np.vstack([u_emb, top5_embs]) @ np.vstack([u_emb, top5_embs]).T
        labels = ["User"] + [f"Pin#{i+1}" for i in range(5)]

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            sim_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
            xticklabels=labels, yticklabels=labels, ax=ax3,
            vmin=-1, vmax=1
        )
        ax3.set_title("Cosine similarity matrix (user + top-5 pins)")
        fig3.tight_layout()
        st.pyplot(fig3)

# ── Tab 2: Training Curves ───────────────────────────────────────────────────
with tabs[1]:
    if history:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        epochs = list(range(1, len(history["train_loss"]) + 1))
        ax1.plot(epochs, history["train_loss"], color="#e60023", linewidth=2)
        ax1.set_title("Training Loss (InfoNCE)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.3)

        ax2.plot(epochs, history["val_recall@10"], label="Recall@10",
                 color="#e60023", linewidth=2)
        ax2.plot(epochs, history["val_ndcg@10"], label="NDCG@10",
                 color="#0076d3", linewidth=2, linestyle="--")
        ax2.set_title("Validation Metrics")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.legend()
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)

        col1, col2, col3 = st.columns(3)
        col1.metric("Best Recall@10", f"{max(history['val_recall@10']):.4f}")
        col2.metric("Best NDCG@10", f"{max(history['val_ndcg@10']):.4f}")
        col3.metric("Min Train Loss", f"{min(history['train_loss']):.4f}")
    else:
        st.info("Train the model first to see learning curves.")

# ── Tab 3: Architecture ──────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Two-Tower Architecture")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────┐
    │              TWO-TOWER MODEL                      │
    │                                                   │
    │  User Features (64-dim)    Item Features (164-dim)│
    │         │                         │               │
    │    ┌────▼────┐               ┌────▼────┐         │
    │    │  MLP    │               │  MLP    │         │
    │    │256→128  │               │256→128  │         │
    │    │LayerNorm│               │LayerNorm│         │
    │    │  GELU   │               │  GELU   │         │
    │    │Dropout  │               │Dropout  │         │
    │    └────┬────┘               └────┬────┘         │
    │         │                         │               │
    │    ┌────▼──────────embedding───────▼────┐        │
    │    │        Shared Space (64-dim)        │        │
    │    │     L2-Normalized embeddings        │        │
    │    └─────────────────────────────────────┘        │
    │                     │                             │
    │            InfoNCE Loss (τ learnable)             │
    │         In-batch negatives (B-1 per sample)      │
    └──────────────────────────────────────────────────┘

    Inference:
    ┌────────────┐     ┌─────────────────┐     ┌────────────────┐
    │ User query │────▶│  User Tower     │────▶│ FAISS IVFFlat  │
    │ (features) │     │ (encode online) │     │ ANN Search     │
    └────────────┘     └─────────────────┘     └───────┬────────┘
                                                        │
                       ┌─────────────────┐             │
                       │ Item embeddings │◀────────────┘
                       │ (pre-computed,  │
                       │  indexed in     │
                       │  FAISS)         │
                       └─────────────────┘
    ```
    """)

    st.subheader("Key Design Decisions")
    st.markdown("""
    | Decision | Choice | Why |
    |---|---|---|
    | Loss | InfoNCE (in-batch negatives) | Scales O(B²) without extra data; B-1 free negatives per step |
    | Temperature | Learnable parameter | Adapts embedding sharpness; starts at 0.07, tuned by SGD |
    | Normalization | L2 on output | Dot product = cosine similarity; FAISS InnerProduct correct |
    | Tower architecture | MLP + LayerNorm + GELU | Stable training; GELU smoother than ReLU for dense inputs |
    | Negative mining | Semi-hard (logged, not re-weighted) | Hard negs tracked for monitoring collapse |
    | ANN | FAISS IVFFlat | 10-100x speedup vs exact at 95%+ recall |
    | Evaluation | Temporal leave-one-out | Prevents future leakage; mirrors production A/B |
    """)
