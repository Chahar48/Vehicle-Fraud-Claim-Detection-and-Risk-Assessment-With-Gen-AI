"""
anomaly_detector.py
-------------------
IsolationForest-based anomaly detection module.

Responsibilities:
- Train IsolationForest using numeric feature matrix
- Save model + scaler using FD_PROJECT_ROOT paths
- Predict anomaly score (0..1)
- Provide manual test block

This file depends on model_utils for:
- MODEL_DIR
- save_model / load_model
"""

from __future__ import annotations
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from fraud_detection.logging.logger import get_logger
from fraud_detection.models.model_utils import (
    save_model,
    load_model,
    MODEL_DIR
)

logger = get_logger(__name__)

# ------------------------------------------------------------
# Model filenames
# ------------------------------------------------------------
MODEL_NAME = "anomaly_detector.pkl"
SCALER_NAME = "anomaly_scaler.pkl"

# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_anomaly_detector(
    X_train,
    n_estimators=200,
    contamination=0.05,   # recommended 0.05 for claims
    random_state=42
):
    """
    Train IsolationForest model on numeric feature matrix.

    X_train must be numeric-only (Phase 8 feature_builder output).
    """
    logger.info("[ANOM] Starting IsolationForest training...")

    X_train = np.array(X_train, dtype=float)
    X_train = np.nan_to_num(X_train, nan=-1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_scaled)

    # Save artifacts
    model_path, scaler_path = save_model(
        model,
        scaler,
        model_name=MODEL_NAME
    )

    logger.info("[ANOM] Training completed.")
    logger.info("[ANOM] Model saved → %s", model_path)
    logger.info("[ANOM] Scaler saved → %s", scaler_path)

    return model, scaler


# ------------------------------------------------------------
# Internal: Convert raw score_samples → normalized anomaly score
# ------------------------------------------------------------
def _normalize_iforest_score(raw_score: float) -> float:
    """
    IsolationForest.score_samples():
        higher → more normal
        lower → more anomalous

    We convert to 0..1 anomaly score:
        0 = normal
        1 = highly anomalous
    """
    inverted = -raw_score
    score = 1 / (1 + np.exp(-inverted))       # sigmoid
    return float(np.clip(score, 0.0, 1.0))


# ------------------------------------------------------------
# Predict anomaly score (0..1)
# ------------------------------------------------------------
def predict_anomaly_score(X):
    """
    X = numeric feature vector (Phase 8 output)
    Returns: float anomaly_score in range 0–1
    """
    model, scaler = load_model(MODEL_NAME, load_scaler=True)

    X = np.array(X, dtype=float)
    X = np.nan_to_num(X, nan=-1)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    X_scaled = scaler.transform(X)

    raw_scores = model.score_samples(X_scaled)
    raw = float(raw_scores[0])

    anomaly = _normalize_iforest_score(raw)

    logger.info("[ANOM] raw_score=%s → anomaly_score=%.4f", raw, anomaly)

    return anomaly


# ------------------------------------------------------------
# Threshold logic
# ------------------------------------------------------------
def is_anomalous(score: float, threshold: float = 0.70) -> bool:
    """
    Returns True if anomaly score >= threshold.
    """
    return score >= threshold


# ------------------------------------------------------------
# Manual Test Block
# ------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    logging.basicConfig(level=logging.INFO)

    logger.info("\n=== MANUAL TEST: anomaly_detector.py ===")

    # Synthetic feature matrix (as Phase 8 numeric features)
    X_train = np.array([
        [10000, 0.2, 1, 200, 0],
        [12000, 0.25, 1, 150, 0],
        [8000, 0.30, 2, 400, 0],
        [500000, 5.0, 10, 1, 1],     # anomalous
        [5, 0.00001, 0, 2000, 1],    # anomalous
    ])

    logger.info("Training IsolationForest model...")
    train_anomaly_detector(X_train)

    test_sample = [20000, 0.90, 3, 500, 0]

    score = predict_anomaly_score(test_sample)
    logger.info("Test sample anomaly score: %.4f", score)

    flag = is_anomalous(score, threshold=0.70)
    logger.info("Is anomalous? → %s", flag)
