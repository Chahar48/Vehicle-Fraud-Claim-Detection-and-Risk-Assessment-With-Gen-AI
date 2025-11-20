"""
embedder.py
-----------
GenAI Layer — Embedding Generator

Purpose:
- Centralized wrapper around sentence-transformers model
- Cached global model instance
- Provides embedding for claim descriptions
- Supports both single and batch embedding generation

This is used by `similarity_index.py` and by the scoring engine.
"""

import os
import numpy as np
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------
# PROJECT ROOT (env-aware)
# -------------------------------------------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Cached model instance
_EMBED_MODEL = None


# -------------------------------------------------------------
# Load embedding model once
# -------------------------------------------------------------
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads (or returns cached) embedding model.
    Used by ALL embedding operations across project.
    """

    global _EMBED_MODEL

    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    try:
        logger.info(f"Loading embedding model: {model_name}")
        _EMBED_MODEL = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        raise

    return _EMBED_MODEL


# -------------------------------------------------------------
# Embed a list of texts
# -------------------------------------------------------------
def embed_text_list(text_list: List[str], normalize: bool = True) -> np.ndarray:
    if not isinstance(text_list, list):
        raise ValueError("embed_text_list expects a list of strings.")

    model = load_embedder()

    try:
        embeddings = model.encode(
            text_list,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise


# -------------------------------------------------------------
# Embed a single text string
# -------------------------------------------------------------
def embed_text(text: str, normalize: bool = True) -> np.ndarray:
    if text is None or not str(text).strip():
        # Safe fallback: zero vector of MiniLM dimension
        return np.zeros(384, dtype="float32")

    model = load_embedder()

    try:
        vector = model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )[0]
        return vector
    except Exception as e:
        logger.error("Single text embedding failed: %s", e)
        raise


# -------------------------------------------------------------
# Optional save/load utilities
# -------------------------------------------------------------
def save_embeddings(vectors: np.ndarray, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, vectors)
        logger.info("Saved embeddings → %s", path)
    except Exception as e:
        logger.error("Failed saving embeddings: %s", e)


def load_embeddings(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found at: {path}")

    return np.load(path)

# ------------------------------------------------------------
# Manual test block
# ------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "Rear bumper damaged due to collision.",
        "Front windshield cracked by stone impact.",
        "Vehicle stolen from parking lot."
    ]

    print("\n=== Testing embedder.py ===")
    embs = embed_text_list(sample_texts)
    print("Embedding shape:", embs.shape)

    single_emb = embed_text("Car was hit from behind at signal.")
    print("Single embedding length:", len(single_emb))
