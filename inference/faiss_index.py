"""
FAISS ANN Index: build, save, load, and query.

Index type: IVFFlat (Inverted File with Flat quantizer)
  - Splits embedding space into `nlist` Voronoi cells
  - At query time, searches `nprobe` nearest cells
  - Balances speed vs recall (tune nprobe for the tradeoff)

For production Pinterest scale:
  - IVFPQ (Product Quantization) for memory efficiency
  - GPU index for sub-millisecond latency
  - Distributed sharding across item partitions
"""

import numpy as np
import faiss
from pathlib import Path
from loguru import logger


def build_faiss_index(
    item_embeddings: np.ndarray,
    cfg: dict,
) -> faiss.Index:
    """
    Build an IVFFlat FAISS index from item embeddings.

    Args:
        item_embeddings : (num_items, embedding_dim) float32
        cfg             : full config dict

    Returns:
        Trained and populated FAISS index
    """
    fc = cfg["faiss"]
    dim = item_embeddings.shape[1]
    nlist = fc["nlist"]
    nprobe = fc["nprobe"]

    assert item_embeddings.dtype == np.float32, "FAISS requires float32"
    assert item_embeddings.flags["C_CONTIGUOUS"], "Array must be C-contiguous"

    logger.info(f"Building IVFFlat index | dim={dim} | nlist={nlist} | nprobe={nprobe}")

    # Quantizer: flat L2 for cell centroids
    quantizer = faiss.IndexFlatIP(dim)   # Inner Product (cosine on L2-norm vecs)

    # IVF index
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe

    # Train: learns Voronoi cell centroids via k-means
    logger.info("Training FAISS index (k-means clustering)...")
    index.train(item_embeddings)

    # Add all item embeddings
    index.add(item_embeddings)

    logger.info(f"FAISS index built | total vectors: {index.ntotal}")
    return index


def load_faiss_index(index_path: str, nprobe: int = 10) -> faiss.Index:
    """Load a serialized FAISS index from disk."""
    index = faiss.read_index(index_path)
    index.nprobe = nprobe
    logger.info(f"Loaded FAISS index from {index_path} | ntotal={index.ntotal}")
    return index


def search_index(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    top_k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for top-K nearest items.

    Args:
        index            : FAISS index
        query_embeddings : (B, D) float32 user embeddings
        top_k            : number of results per query

    Returns:
        scores   : (B, K) similarity scores
        item_ids : (B, K) retrieved item indices
    """
    query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    scores, item_ids = index.search(query_embeddings, top_k)
    return scores, item_ids


class PinterestRetriever:
    """
    End-to-end retrieval: encode user → FAISS search → return ranked pins.

    This is the inference-time object used by the Streamlit demo and
    would be served behind a gRPC endpoint in production.
    """

    def __init__(
        self,
        model,
        index: faiss.Index,
        item_features: np.ndarray,
        pin_metadata: dict,
        cfg: dict,
        device,
    ):
        self.model = model
        self.index = index
        self.item_features = item_features
        self.pin_metadata = pin_metadata   # {pin_id: {category, ...}}
        self.cfg = cfg
        self.device = device
        self.top_k = cfg["faiss"]["top_k"]

    def retrieve(self, user_feature_vector: np.ndarray) -> list[dict]:
        """
        Given a user feature vector, return top-K ranked pins.

        Args:
            user_feature_vector : (user_feat_dim,) float32

        Returns:
            list of dicts with {pin_id, score, category}
        """
        import torch

        self.model.eval()
        with torch.no_grad():
            u_feat = torch.from_numpy(
                user_feature_vector[np.newaxis, :].astype(np.float32)
            ).to(self.device)
            u_emb = self.model.encode_users(u_feat).cpu().numpy()

        scores, item_ids = search_index(self.index, u_emb, self.top_k)
        scores = scores[0]
        item_ids = item_ids[0]

        results = []
        for rank, (iid, score) in enumerate(zip(item_ids, scores)):
            meta = self.pin_metadata.get(int(iid), {})
            results.append({
                "rank": rank + 1,
                "pin_id": int(iid),
                "score": float(score),
                "category": meta.get("category", "unknown"),
            })
        return results
