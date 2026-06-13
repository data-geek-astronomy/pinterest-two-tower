"""
Pinterest Two-Tower Retrieval — Streamlit Demo
Auto-generates data and trains a demo model on first run.

Run locally:  streamlit run app/streamlit_app.py
Deploy:       push to GitHub → connect on share.streamlit.io
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = str(ROOT / "demo_config.yaml")

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

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pinterest Two-Tower Retrieval",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #e60023 0%, #ad081b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #e60023;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }
    .stProgress .st-bo { background-color: #e60023; }
</style>
""", unsafe_allow_html=True)


# ─── Auto-Setup Pipeline ──────────────────────────────────────────────────────

def run_setup(cfg: dict, status_container):
    """Generate data + train model if artifacts don't exist."""
    import torch
    import faiss as _faiss
    from data.generate_data import main as gen_data
    from pipeline.dataset import build_dataloaders
    from models.two_tower import build_model, InfoNCELoss, HardNegativeMiner
    import torch.optim as optim
    from tqdm import tqdm

    model_path = Path(cfg["paths"]["model_dir"]) / "best_model.pt"

    with status_container.status("🚀 Setting up demo (first run only — ~2-3 min)...", expanded=True) as s:

        # Step 1: Generate data
        st.write("📦 Generating synthetic Pinterest dataset...")
        gen_data(CONFIG_PATH)
        st.write("✅ Dataset ready: 2,000 users · 8,000 pins · 30 categories")

        # Step 2: Build data loaders
        st.write("⚙️ Engineering features & building data pipeline...")
        train_loader, val_loader, _, meta = build_dataloaders(cfg)
        st.write(f"✅ Pipeline ready | user_dim={meta['user_feat_dim']} · item_dim={meta['item_feat_dim']}")

        # Step 3: Train
        device = torch.device("cpu")
        model = build_model(cfg, meta["user_feat_dim"], meta["item_feat_dim"]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
        criterion = InfoNCELoss()
        miner = HardNegativeMiner(cfg["training"]["hard_negative_ratio"])

        st.write(f"🧠 Training Two-Tower model ({model.num_parameters():,} params)...")
        prog = st.progress(0)
        history = {"train_loss": [], "val_recall@10": [], "val_ndcg@10": []}
        best_recall = 0.0

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            model.train()
            total_loss = 0.0
            for user_feats, item_feats, weights in train_loader:
                u_emb, i_emb = model(user_feats.to(device), item_feats.to(device))
                loss = criterion(u_emb, i_emb, model.temperature, weights.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            history["val_recall@10"].append(0.0)
            history["val_ndcg@10"].append(0.0)

            prog.progress(epoch / cfg["training"]["epochs"],
                          text=f"Epoch {epoch}/{cfg['training']['epochs']} · loss={avg_loss:.4f}")

        st.write(f"✅ Training complete! Final loss: {history['train_loss'][-1]:.4f}")

        # Step 4: Build FAISS index
        st.write("🗂️ Building FAISS index for fast retrieval...")
        model.eval()
        item_t = torch.from_numpy(meta["item_features"])
        all_embs = []
        with torch.no_grad():
            for start in range(0, len(item_t), 512):
                emb = model.encode_items(item_t[start:start+512])
                all_embs.append(emb.numpy())
        item_embs = np.vstack(all_embs).astype(np.float32)

        dim = item_embs.shape[1]
        index = _faiss.IndexFlatIP(dim)
        index.add(item_embs)

        Path(cfg["paths"]["model_dir"]).mkdir(parents=True, exist_ok=True)
        _faiss.write_index(index, cfg["paths"]["index_path"])
        np.save(cfg["paths"]["embeddings_path"], item_embs)

        torch.save({
            "model_state": model.state_dict(),
            "meta": {"user_feat_dim": meta["user_feat_dim"], "item_feat_dim": meta["item_feat_dim"]},
            "cfg": cfg,
        }, model_path)

        hist_path = Path(cfg["paths"]["model_dir"]) / "training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f)

        st.write("✅ FAISS index built & saved")
        s.update(label="✅ Demo ready!", state="complete", expanded=False)

    return model, index, item_embs, meta, history


@st.cache_resource(show_spinner=False)
def load_everything():
    import yaml, torch, faiss as _faiss
    from pipeline.dataset import build_dataloaders

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    model_path = Path(cfg["paths"]["model_dir"]) / "best_model.pt"
    hist_path = Path(cfg["paths"]["model_dir"]) / "training_history.json"

    status_box = st.empty()

    if not model_path.exists():
        model, index, item_embs, meta, history = run_setup(cfg, status_box)
        status_box.empty()
    else:
        from models.two_tower import build_model
        ckpt = torch.load(model_path, map_location="cpu")
        m = ckpt["meta"]
        model = build_model(cfg, m["user_feat_dim"], m["item_feat_dim"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        index = _faiss.read_index(cfg["paths"]["index_path"])
        item_embs = np.load(cfg["paths"]["embeddings_path"])

        _, _, _, meta = build_dataloaders(cfg)
        history = json.load(open(hist_path)) if hist_path.exists() else None

    user_df = pd.read_parquet(Path(cfg["paths"]["data_dir"]) / "raw/users.parquet")
    pin_df = pd.read_parquet(Path(cfg["paths"]["data_dir"]) / "raw/pins.parquet")

    return model, index, item_embs, meta, user_df, pin_df, cfg, history


# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_user_feat(user_row, cfg):
    num_cat = cfg["data"]["num_categories"]
    cat_names = CATEGORIES[:num_cat]
    interest_vals = [float(user_row.get(f"interest_{c}", 0.0)) for c in cat_names]
    behavior_vals = [0.0] * 6
    meta_vals = [
        float(user_row.get("account_age_days", 0)) / 2000,
        float(user_row.get("num_boards", 0)) / 50,
        float(user_row.get("num_pins_saved", 0)) / 5000,
        float(user_row.get("is_mobile", 0)),
    ]
    return np.array(interest_vals + behavior_vals + meta_vals, dtype=np.float32)


def do_retrieval(model, index, user_feat, top_k):
    import torch
    model.eval()
    with torch.no_grad():
        u_t = torch.from_numpy(user_feat[np.newaxis, :])
        u_emb = model.encode_users(u_t).numpy().astype(np.float32)
    scores, ids = index.search(u_emb, top_k)
    return scores[0], ids[0], u_emb[0]


# ─── Main UI ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:1.8rem;">📌 Pinterest Two-Tower Retrieval</h1>
    <p style="margin:0.3rem 0 0 0; opacity:0.9;">
        ML Engineer Interview Project · Dual Encoder · FAISS ANN · InfoNCE Loss
    </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model, index, item_embs, meta, user_df, pin_df, cfg, history = load_everything()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Retrieval Settings")
    top_k = st.slider("Top-K results", 5, 50, 20)
    nprobe_val = st.slider("FAISS nprobe", 1, 20, 5,
                           help="Higher = better recall, slower query")
    try:
        index.nprobe = nprobe_val
    except Exception:
        pass

    st.divider()
    st.markdown("### 👤 Select User")
    max_uid = cfg["data"]["num_users"] - 1
    user_id = st.number_input("User ID", min_value=0, max_value=max_uid, value=42)

    st.divider()
    st.markdown("### 🎨 Custom Interest Profile")
    use_custom = st.toggle("Build custom user")
    custom_interests = {}
    if use_custom:
        st.caption("Set interest weights:")
        num_cat = cfg["data"]["num_categories"]
        for cat in CATEGORIES[:num_cat]:
            emoji = CATEGORY_EMOJIS.get(cat, "")
            custom_interests[cat] = st.slider(f"{emoji} {cat}", 0.0, 1.0, 0.0, step=0.1)

    st.divider()
    st.markdown("### 📊 Model Stats")
    st.markdown(f"""
    <div class="metric-card">👥 Users: <b>{cfg['data']['num_users']:,}</b></div>
    <div class="metric-card">📌 Pins: <b>{cfg['data']['num_pins']:,}</b></div>
    <div class="metric-card">🏷️ Categories: <b>{cfg['data']['num_categories']}</b></div>
    <div class="metric-card">🧠 Embed dim: <b>{cfg['model']['embedding_dim']}</b></div>
    <div class="metric-card">⚡ Index: <b>FAISS Flat ({index.ntotal:,} vecs)</b></div>
    """, unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Retrieval", "📊 Training Curves", "🏗️ Architecture", "🧪 Embedding Space"]
)

# ─── Tab 1: Retrieval ─────────────────────────────────────────────────────────
with tab1:
    num_cat = cfg["data"]["num_categories"]

    if use_custom:
        total = sum(custom_interests.values()) or 1.0
        norm = {k: v / total for k, v in custom_interests.items()}
        user_feat = np.array(
            [norm.get(c, 0.0) for c in CATEGORIES[:num_cat]] + [0.0] * 10,
            dtype=np.float32,
        )
        interests = norm
        display_name = "Custom User"
    else:
        row = user_df[user_df["user_id"] == user_id].iloc[0].to_dict()
        user_feat = build_user_feat(row, cfg)
        interests = {c: float(row.get(f"interest_{c}", 0.0)) for c in CATEGORIES[:num_cat]}
        display_name = f"User #{user_id}"

    scores, item_ids, u_emb = do_retrieval(model, index, user_feat, top_k)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.subheader(f"👤 {display_name}")

        top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:6]
        st.markdown("**Top interests:**")
        for cat, val in top_interests:
            if val > 0.005:
                emoji = CATEGORY_EMOJIS.get(cat, "")
                st.progress(float(val), text=f"{emoji} {cat}  ({val:.1%})")

        fig, ax = plt.subplots(figsize=(4, 3))
        cats_plot = [CATEGORY_EMOJIS.get(c, "") + " " + c for c, _ in top_interests if _ > 0.005]
        vals_plot = [v for _, v in top_interests if v > 0.005]
        ax.barh(cats_plot, vals_plot, color="#e60023", alpha=0.85)
        ax.set_xlabel("Interest weight")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.subheader(f"🔍 Top-{top_k} Retrieved Pins")

        results = []
        for rank, (iid, score) in enumerate(zip(item_ids, scores)):
            if 0 <= int(iid) < len(pin_df):
                pr = pin_df.iloc[int(iid)]
                results.append({
                    "Rank": rank + 1,
                    "Pin ID": int(iid),
                    "Category": f"{CATEGORY_EMOJIS.get(pr['primary_category'], '')} {pr['primary_category']}",
                    "Similarity": round(float(score), 4),
                    "Saves": f"{int(pr['num_saves']):,}",
                    "Promoted": "✓" if pr["is_promoted"] else "",
                })

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        # Category breakdown
        cat_counts = pd.Series(
            [r["Category"].split(" ", 1)[-1] for r in results]
        ).value_counts()

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        cat_counts.plot(kind="bar", ax=ax2, color="#e60023", alpha=0.85, width=0.7)
        ax2.set_title("Category distribution in retrieved pins", fontweight="bold")
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", rotation=45)
        ax2.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

# ─── Tab 2: Training Curves ───────────────────────────────────────────────────
with tab2:
    st.subheader("📈 Training History")
    if history and history.get("train_loss"):
        epochs_x = list(range(1, len(history["train_loss"]) + 1))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs_x, history["train_loss"], color="#e60023", linewidth=2.5,
                marker="o", markersize=4, label="Train Loss (InfoNCE)")
        ax.set_title("InfoNCE Training Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Min Loss", f"{min(history['train_loss']):.4f}")
        c2.metric("Epochs Trained", len(history["train_loss"]))
        c3.metric("Embedding Dim", cfg["model"]["embedding_dim"])

        st.info(
            "💡 **What InfoNCE loss means:** For each user in a batch, "
            "the model pushes their embedding close to their saved pin and away from all "
            "other pins in the batch (in-batch negatives). Lower loss = tighter, "
            "more separated clusters in embedding space."
        )
    else:
        st.info("Train the model to see learning curves.")

# ─── Tab 3: Architecture ──────────────────────────────────────────────────────
with tab3:
    st.subheader("🏗️ Two-Tower Architecture")

    st.code("""
  User Features (64-dim)          Item Features (164-dim)
         │                                │
   ┌─────▼──────┐                  ┌─────▼──────┐
   │  User MLP  │                  │  Item MLP  │
   │  128 → 64  │                  │  128 → 64  │
   │ LayerNorm  │                  │ LayerNorm  │
   │    GELU    │                  │    GELU    │
   │  Dropout   │                  │  Dropout   │
   └─────┬──────┘                  └─────┬──────┘
         │                                │
         └──────────┐  ┌─────────────────┘
                    ▼  ▼
             Shared Embedding Space (64-dim)
                 L2-Normalized

          InfoNCE Loss  (τ = learnable)
       In-batch negatives: B-1 per positive

  ─────────────── Inference ───────────────────
  User query ──► UserTower ──► FAISS search ──► Top-K pins
                                    ▲
                         Item embeddings (pre-indexed)
    """, language="text")

    st.markdown("### Key Design Decisions")
    decisions = {
        "Loss Function": ("InfoNCE with in-batch negatives",
                          "B−1 free negatives per step — scales without extra sampling"),
        "Temperature τ": ("Learnable parameter (init 0.07)",
                          "Self-calibrates embedding sharpness during training"),
        "Output normalization": ("L2-norm on both towers",
                                 "Dot product = cosine similarity; FAISS InnerProduct correct"),
        "ANN Index": ("FAISS Flat (demo) / IVFFlat (prod)",
                      "10-100× faster than exact; tune nprobe for recall-speed tradeoff"),
        "Train/Val split": ("Temporal leave-one-out",
                            "Prevents future leakage; mirrors real A/B test protocol"),
    }

    for decision, (choice, reason) in decisions.items():
        with st.expander(f"**{decision}** → {choice}"):
            st.markdown(f"*Why:* {reason}")

# ─── Tab 4: Embedding Space ───────────────────────────────────────────────────
with tab4:
    st.subheader("🧪 Embedding Space Analysis")

    # Cosine similarity heatmap: user vs top-8 pins
    top8_ids = item_ids[:8]
    top8_embs = item_embs[top8_ids].astype(np.float32)
    # Normalize
    top8_norm = top8_embs / (np.linalg.norm(top8_embs, axis=1, keepdims=True) + 1e-8)
    u_norm = u_emb / (np.linalg.norm(u_emb) + 1e-8)
    all_vecs = np.vstack([u_norm, top8_norm])
    sim_mat = all_vecs @ all_vecs.T

    labels = ["User"] + [
        f"Pin#{i+1}\n{CATEGORY_EMOJIS.get(pin_df.iloc[int(iid)]['primary_category'], '')}"
        for i, iid in enumerate(top8_ids) if int(iid) < len(pin_df)
    ]

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        sim_mat, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=labels, yticklabels=labels,
        ax=ax3, vmin=-1, vmax=1, linewidths=0.5,
    )
    ax3.set_title("Cosine Similarity Matrix — User vs Top-8 Retrieved Pins",
                  fontweight="bold", pad=12)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    st.info(
        "**Reading this chart:** Values close to 1.0 (green) = highly similar in embedding space. "
        "The user row shows how similar the query user is to each retrieved pin. "
        "Pins similar to each other cluster together (same category pins should be green with each other)."
    )

    # Score distribution
    st.markdown("### Score Distribution of Retrieved Pins")
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.bar(range(1, len(scores) + 1), scores, color="#e60023", alpha=0.8)
    ax4.set_xlabel("Rank")
    ax4.set_ylabel("Cosine Similarity")
    ax4.set_title("Similarity scores decay across ranks (expected behavior)")
    ax4.spines[["top", "right"]].set_visible(False)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:gray; font-size:0.8rem;'>"
    "Pinterest Two-Tower Retrieval · Built with PyTorch + FAISS + Streamlit · "
    "github.com/data-geek-astronomy/pinterest-two-tower"
    "</center>",
    unsafe_allow_html=True,
)
