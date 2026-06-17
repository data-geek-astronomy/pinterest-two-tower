---
title: Pinterest Two-Tower Retrieval
emoji: 📌
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Two-Tower retrieval system for Pinterest-style personalization
---

# Pinterest Two-Tower Retrieval System

A production-grade **Two-Tower (Dual Encoder)** retrieval system built to personalize Pinterest-style content feeds. Designed as an ML Engineer interview project demonstrating the full ML lifecycle from data generation to ANN-indexed inference.

---

## Architecture

```
User Features ──► UserTower (MLP) ──►  L2-norm  ──►  FAISS IVFFlat
                                       embedding       ANN Index
Item Features ──► ItemTower (MLP) ──►  L2-norm  ──►  (pre-computed)
                                       embedding

Training: InfoNCE loss with in-batch negatives + learnable temperature
```

### Key Design Decisions

| Component | Choice | Rationale |
|---|---|---|
| Loss | InfoNCE (in-batch negatives) | B−1 free negatives per step, scales with batch size |
| Temperature τ | Learnable | Adapts embedding sharpness during training |
| Output | L2-normalized | Dot product = cosine; compatible with FAISS METRIC_INNER_PRODUCT |
| ANN Index | FAISS IVFFlat | 10–100× faster than exact search at 95%+ recall |
| Negatives | Semi-hard monitoring | Hard neg count logged per epoch to detect representation collapse |
| Evaluation | Temporal leave-one-out split | Prevents future leakage; mirrors production A/B test protocol |

---

## Project Structure

```
pinterest-two-tower/
├── config.yaml              # All hyperparameters in one place
├── data/
│   └── generate_data.py     # Synthetic Pinterest dataset (10K users, 50K pins)
├── pipeline/
│   └── dataset.py           # Feature engineering + PyTorch Dataset/DataLoader
├── models/
│   └── two_tower.py         # TwoTowerModel, InfoNCELoss, HardNegativeMiner
├── evaluation/
│   └── metrics.py           # Recall@K, NDCG@K, MRR
├── inference/
│   └── faiss_index.py       # FAISS build/load/search + PinterestRetriever
├── scripts/
│   └── train.py             # Full training loop with early stopping
└── app/
    └── streamlit_app.py     # Interactive retrieval demo
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data + train (≈10–20 min on CPU)
python scripts/train.py --regenerate-data

# 3. Launch the Streamlit demo
streamlit run app/streamlit_app.py
```

---

## Dataset (Synthetic)

Mirrors a Pinterest-style interaction graph:

- **10,000 users** with sparse interest profiles across 30 categories (home_decor, fashion, travel, …)
- **50,000 pins** with 128-dim simulated visual embeddings + category features
- **~500K interactions** (saves, clicks, close-ups) sampled with interest-driven affinity
- **Temporal split**: leave-last-out for val/test — no future leakage

---

## Training

```bash
python scripts/train.py [--config config.yaml] [--regenerate-data]
```

**What happens:**
1. Loads/generates data → feature engineering (Spark-equivalent aggregations)
2. Trains TwoTowerModel with InfoNCE loss + in-batch negatives
3. Validates with Recall@K / NDCG@K on temporal hold-out
4. Early stops on Recall@10 (patience=5)
5. Builds FAISS IVFFlat index from item embeddings
6. Saves model checkpoint + training history

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Recall@K** | Fraction of users where the positive pin appears in top-K retrieved |
| **NDCG@K** | Ranking quality — rewards finding the positive at higher ranks |
| **MRR** | Mean Reciprocal Rank across all queries |

---

## Inference

```python
from inference.faiss_index import PinterestRetriever, load_faiss_index
import numpy as np, torch, yaml

cfg = yaml.safe_load(open("config.yaml"))
model = ...   # load from checkpoint
index = load_faiss_index(cfg["paths"]["index_path"])

retriever = PinterestRetriever(model, index, item_features, pin_metadata, cfg, "cpu")
results = retriever.retrieve(user_feature_vector)
# [{'rank': 1, 'pin_id': 12345, 'score': 0.94, 'category': 'travel'}, ...]
```

---

## FAISS Index Tuning

| Parameter | Effect |
|---|---|
| `nlist` | Number of Voronoi cells; more = finer partitioning, slower build |
| `nprobe` | Cells searched at query time; higher = better recall, slower query |
| Index type | `IVFFlat` → `IVFPQ` for 4–8× memory reduction at production scale |

---

## Interview Talking Points

1. **Why in-batch negatives?** They come for free at scale — a batch of 512 gives 511 negatives per positive. No extra sampling needed.
2. **Why learnable temperature?** A fixed τ is a hyperparameter to tune. Learning it lets the model self-calibrate embedding sharpness.
3. **Why temporal split?** Chronological hold-out mirrors the real production scenario where we predict *future* interactions, not past ones.
4. **ANN tradeoff?** `nprobe` is the knob: `nprobe=1` → fastest, lowest recall; `nprobe=nlist` → exact search. We tune for P99 latency SLA.
5. **What breaks at Pinterest scale?** (a) item corpus too large for one FAISS node → shard by category; (b) in-batch negatives too easy → hard negative mining from high-confidence false positives in the ANN results.

---

## Tech Stack

- **PyTorch** — model training
- **FAISS** — approximate nearest neighbor index
- **Pandas** — feature engineering (Spark-equivalent logic)
- **Streamlit** — interactive demo
- **Loguru** — structured logging
