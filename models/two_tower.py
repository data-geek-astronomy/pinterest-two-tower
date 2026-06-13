"""
Two-Tower (Dual Encoder) Model for Pinterest Personalized Retrieval.

Architecture:
  UserTower: user_features → MLP → L2-normalized embedding (d)
  ItemTower: item_features → MLP → L2-normalized embedding (d)

Training objective:
  InfoNCE / in-batch softmax contrastive loss.
  For a batch of size B, each (user_i, item_i) pair is positive;
  all other items in the batch serve as negatives.

At inference:
  - Pre-compute item embeddings → build FAISS index
  - For a query user → encode → ANN search in FAISS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ─── MLP Tower ───────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Shared MLP backbone used by both towers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Two-Tower Model ─────────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Dual encoder producing L2-normalized embeddings in a shared space.

    Args:
        user_feat_dim  : dimension of raw user feature vector
        item_feat_dim  : dimension of raw item feature vector
        embedding_dim  : shared embedding space dimension
        hidden_dims    : list of hidden layer sizes for each MLP
        dropout        : dropout probability
        temperature    : InfoNCE softmax temperature
    """

    def __init__(
        self,
        user_feat_dim: int,
        item_feat_dim: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.2,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = nn.Parameter(
            torch.tensor(temperature), requires_grad=True
        )

        self.user_tower = MLP(user_feat_dim, hidden_dims, embedding_dim, dropout)
        self.item_tower = MLP(item_feat_dim, hidden_dims, embedding_dim, dropout)

        self._init_weights()
        logger.info(
            f"TwoTowerModel | user_dim={user_feat_dim} → item_dim={item_feat_dim} "
            f"→ embed_dim={embedding_dim} | temp={temperature}"
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_users(self, user_feats: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized user embeddings. Shape: (B, D)"""
        return F.normalize(self.user_tower(user_feats), p=2, dim=-1)

    def encode_items(self, item_feats: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized item embeddings. Shape: (B, D)"""
        return F.normalize(self.item_tower(item_feats), p=2, dim=-1)

    def forward(
        self,
        user_feats: torch.Tensor,
        item_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (user_embeddings, item_embeddings) — both L2-normalized.
        Loss is computed externally for flexibility.
        """
        u_emb = self.encode_users(user_feats)
        i_emb = self.encode_items(item_feats)
        return u_emb, i_emb

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Loss Functions ───────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    In-batch InfoNCE (NT-Xent) contrastive loss.

    Given a batch of (user, pos_item) pairs, each pos_item is the positive
    for its user and all other items in the batch act as negatives.

    Loss = -mean(log softmax(sim(u_i, i_i) / τ))
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        user_emb: torch.Tensor,   # (B, D)
        item_emb: torch.Tensor,   # (B, D)
        temperature: torch.Tensor,
        weights: torch.Tensor | None = None,  # (B,) optional interaction weights
    ) -> torch.Tensor:
        # Clamp temperature to avoid numerical instability
        temp = torch.clamp(temperature, min=0.01, max=1.0)

        # Similarity matrix (B, B): rows=users, cols=items
        logits = torch.matmul(user_emb, item_emb.T) / temp  # (B, B)

        # Diagonal entries are positives
        labels = torch.arange(logits.size(0), device=logits.device)

        if weights is not None:
            # Weight the loss by interaction quality (saves > clicks)
            loss_per_sample = F.cross_entropy(logits, labels, reduction="none")
            w = weights / weights.sum()
            return (loss_per_sample * w).sum()
        else:
            return F.cross_entropy(logits, labels)


class HardNegativeMiner:
    """
    Semi-hard negative mining within a batch.

    Identifies negatives that are closer to the query than the positive
    but not so hard they dominate training.
    """

    def __init__(self, hard_ratio: float = 0.3):
        self.hard_ratio = hard_ratio

    @torch.no_grad()
    def mine(
        self,
        user_emb: torch.Tensor,  # (B, D)
        item_emb: torch.Tensor,  # (B, D)
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Returns a permuted item_emb where hard negatives are brought
        closer to the front of each row's negative list.
        """
        B = user_emb.size(0)
        sim = torch.matmul(user_emb, item_emb.T)  # (B, B)

        # For each user, rank non-positive items by similarity (desc)
        pos_sim = sim.diag().unsqueeze(1)   # (B, 1)
        mask = torch.eye(B, dtype=torch.bool, device=sim.device)
        sim_neg = sim.masked_fill(mask, -1e9)

        # Hard negatives: sim_neg > pos_sim (violators of margin)
        hard_mask = sim_neg > pos_sim
        n_hard = int(B * self.hard_ratio)

        # Sort negatives: hard first, then easy
        _, sorted_idx = sim_neg.sort(dim=1, descending=True)

        # Re-index item embeddings to surface hard negatives per query
        # (used for logging/analysis; actual loss uses full in-batch)
        hard_neg_idx = sorted_idx[:, :n_hard]  # (B, n_hard)

        return hard_neg_idx, hard_mask.sum().item()


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(cfg: dict, user_feat_dim: int, item_feat_dim: int) -> TwoTowerModel:
    mc = cfg["model"]
    return TwoTowerModel(
        user_feat_dim=user_feat_dim,
        item_feat_dim=item_feat_dim,
        embedding_dim=mc["embedding_dim"],
        hidden_dims=mc["hidden_dims"],
        dropout=mc["dropout"],
        temperature=mc["temperature"],
    )
