"""
similarity_index.py
------------------
Phase 8 — Text Embedding Similarity

This module computes:
- Embeddings of text descriptions
- FAISS index for similarity search
- Similarity score to past fraudulent descriptions

Used in the scoring engine.
"""

import numpy as np
import faiss
from typing import List

from sentence_transformers import SentenceTransformer
from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# Global model cache
_embedding_model = None


# ------------------------------------------------------------
# Load embedding model (MiniLM from model.yaml)
# ------------------------------------------------------------
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    try:
        logger.info("Loading embedding model: %s", model_name)
        _embedding_model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error("Embedding model load failed: %s", e)
        raise

    return _embedding_model


# ------------------------------------------------------------
# Convert a list of descriptions → embedding vectors
# ------------------------------------------------------------
def get_embeddings(texts: List[str]):
    model = load_embedding_model()

    try:
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors
    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e)
        raise


# ------------------------------------------------------------
# Build FAISS index for historical (fraudulent) descriptions
# ------------------------------------------------------------
def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)

    index.add(vectors.astype("float32"))
    logger.info("FAISS index built with %d vectors (dim=%d)", vectors.shape[0], dim)

    return index


# ------------------------------------------------------------
# Query similarity score for a new description
# ------------------------------------------------------------
def compute_similarity_score(index, query_vector):
    query = np.array(query_vector).astype("float32").reshape(1, -1)

    distances, _ = index.search(query, k=1)

    # Convert L2 distance → similarity (0 to 1)
    similarity = 1 / (1 + distances[0][0])
    return float(similarity)
