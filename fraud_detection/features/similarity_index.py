"""
similarity_index.py

Provides embedding + similarity helpers and a lightweight quick_similarity fallback
that pipeline_runner can call.

APIs:
 - embed_text(text) -> np.ndarray | None
 - load_embeddings(path) -> np.ndarray | None
 - build_index_from_embeddings(embs) -> faiss index | None
 - compute_similarity_score(index, query_vector) -> float
 - quick_similarity(text) -> float  # safe fallback used by pipeline_runner
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np

# logger
try:
    from fraud_detection.logging.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fraud_detection.features.similarity_index")


# lazy model / faiss imports
_embedding_model = None
_faiss = None


def _project_root() -> str:
    pr = os.environ.get("FD_PROJECT_ROOT")
    if pr:
        return pr
    # assume file is under fraud_detection/features/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_faiss():
    global _faiss
    if _faiss is not None:
        return _faiss
    try:
        import faiss as _faiss_local  # type: ignore
        _faiss = _faiss_local
        return _faiss
    except Exception as e:
        logger.debug("faiss not available: %s", e)
        _faiss = None
        return None


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading embedding model: %s", model_name)
        _embedding_model = SentenceTransformer(model_name)
        return _embedding_model
    except Exception as e:
        logger.debug("sentence-transformers not available: %s", e)
        _embedding_model = None
        return None


def embed_text(text: str) -> Optional[np.ndarray]:
    """
    Return normalized vector (float32 1D) or None when embedding model missing.
    """
    if not text or str(text).strip() == "":
        return None
    model = load_embedding_model()
    if model is None:
        return None
    try:
        vecs = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs[0], dtype=np.float32)
    except Exception as e:
        logger.exception("embed_text failed: %s", e)
        return None


def load_embeddings(path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Load embeddings numpy array from disk. Default: data/processed/embeddings.npy
    """
    try:
        if path is None:
            path = os.path.join(_project_root(), "data", "processed", "embeddings.npy")
        if not os.path.exists(path):
            logger.debug("Embeddings file not found: %s", path)
            return None
        arr = np.load(path)
        if arr is None or len(arr) == 0:
            return None
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        logger.exception("load_embeddings failed for %s", path)
        return None


def build_index_from_embeddings(embs: np.ndarray):
    """
    Build simple FAISS IndexFlatL2 if faiss available, else return None.
    """
    faiss_mod = _ensure_faiss()
    if faiss_mod is None:
        logger.debug("FAISS not available, cannot build index")
        return None
    try:
        dim = int(embs.shape[1])
        index = faiss_mod.IndexFlatL2(dim)
        index.add(embs.astype("float32"))
        logger.info("FAISS index built with %d vectors (dim=%d)", embs.shape[0], dim)
        return index
    except Exception:
        logger.exception("build_index_from_embeddings failed")
        return None


def compute_similarity_score(index, query_vector):
    """
    Given a FAISS index and query vector, return similarity in [0,1].
    If index is None, returns 0.0.
    """
    try:
        if index is None or query_vector is None:
            return 0.0
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        distances, _ = index.search(q, k=1)
        d = float(distances[0][0])
        sim = 1.0 / (1.0 + d)
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        logger.exception("compute_similarity_score failed")
        return 0.0


def quick_similarity(text: str) -> float:
    """
    Lightweight pipeline fallback:
      1) try embeddings + historical embeddings (data/processed/embeddings.npy)
      2) if not available, fall back to token overlap heuristic (non-zero PoC)
    Returns float in [0,1].
    """
    try:
        if not text or str(text).strip() == "":
            return 0.0

        # Prefer embedding-based similarity
        try:
            query_emb = embed_text(text)
            hist = load_embeddings()
            if query_emb is not None and hist is not None and hist.size:
                idx = build_index_from_embeddings(hist)
                if idx is not None:
                    return compute_similarity_score(idx, query_emb)
        except Exception:
            logger.debug("Embedding-based quick_similarity failed, falling back to heuristic", exc_info=True)

        # Fallback token-overlap heuristic:
        tokens = [t.strip().lower() for t in text.split() if t.strip()]
        if not tokens:
            return 0.0
        # Use simple unigram overlap with a tiny smoothing to avoid zero
        token_set = set(tokens)
        # if we have historical text samples file we could do more; for now compute self-similarity proxy:
        overlap_score = min(1.0, len(token_set) / 20.0)  # small heuristic: more unique tokens -> higher score up to 1
        return float(np.clip(overlap_score, 0.0, 1.0))
    except Exception:
        logger.exception("quick_similarity failed")
        return 0.0


# compatibility alias used in older pipeline code
def compute_similarity(index, query_vector):
    return compute_similarity_score(index, query_vector)
