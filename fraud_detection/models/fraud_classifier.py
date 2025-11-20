"""
fraud_classifier.py
-------------------
Supervised fraud detection model (XGBoost).

Responsibilities:
- Train classifier on numeric features (Phase 8 output)
- Save model + scaler under FD_PROJECT_ROOT
- Predict fraud probability (0..1)
- Provide manual test block

This module uses model_utils for:
- model directory handling
- saving/loading
- numeric matrix consistency
"""

from __future__ import annotations
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from fraud_detection.logging.logger import get_logger
from fraud_detection.models.model_utils import (
    save_model,
    load_model,
)

logger = get_logger(__name__)

# ------------------------------------------------------------
# File names (relative to MODEL_DIR from model_utils)
# ------------------------------------------------------------
MODEL_NAME = "fraud_classifier.pkl"
SCALER_NAME = "fraud_classifier_scaler.pkl"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ensure_numpy(X):
    """Convert input to numpy array and ensure 2D."""
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
def train_fraud_classifier(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    persist: bool = True
):
    """
    Train supervised fraud classifier (XGBoost).
    X = numeric feature matrix (Phase 8)
    y = labels (0=legit, 1=fraud)

    Returns:
        model, scaler, metrics_dict
    """

    logger.info("[XGB] Starting supervised training...")

    X = _ensure_numpy(X)
    y = np.array(y, dtype=int)

    # ----------------------------------------------
    # Sanity checks
    # ----------------------------------------------
    unique = np.unique(y)
    if len(unique) < 2:
        raise ValueError("Training requires at least two classes (0 and 1).")

    stratify = y if len(unique) == 2 else None

    # ----------------------------------------------
    # Train/validation split
    # ----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # ----------------------------------------------
    # Scaling
    # ----------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ----------------------------------------------
    # Compute class weight for imbalance
    # ----------------------------------------------
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    weight = (neg / pos) if pos > 0 else 1.0

    # ----------------------------------------------
    # Model definition
    # ----------------------------------------------
    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=weight,
        random_state=random_state,
        n_jobs=-1
    )

    logger.info("[XGB] Fitting model...")
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

    # ----------------------------------------------
    # Evaluation
    # ----------------------------------------------
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score
    )

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(unique) == 2 else np.nan,
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    logger.info("[XGB] Training completed. Metrics: %s", metrics)

    # ----------------------------------------------
    # Save artifacts
    # ----------------------------------------------
    if persist:
        save_model(model, scaler, model_name=MODEL_NAME)

    return model, scaler, metrics


# ------------------------------------------------------------
# PREDICT FRAUD PROBABILITY
# ------------------------------------------------------------
def predict_fraud_proba(X):
    """
    Returns fraud probability (0..1)
    X = numeric feature vector or batch
    """

    model, scaler = load_model(MODEL_NAME, load_scaler=True)

    X_arr = _ensure_numpy(X)
    X_scaled = scaler.transform(X_arr)

    probs = model.predict_proba(X_scaled)[:, 1]
    return probs


# ------------------------------------------------------------
# Manual Test Block
# ------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    logging.basicConfig(level=logging.INFO)

    logger.info("\n=== MANUAL TEST: fraud_classifier.py ===")

    # Synthetic dataset
    X_train = np.array([
        [12000, 0.20, 1, 200, 0],
        [9000, 0.18, 0, 300, 1],
        [500000, 5.0, 10, 10, 1],   # clear fraud
        [10000, 0.22, 1, 150, 0],
        [8000,  0.15, 0, 400, 0],
        [200000, 3.0, 8, 5, 1],     # fraud pattern
    ])

    y_train = np.array([0, 0, 1, 0, 0, 1])

    model, scaler, metrics = train_fraud_classifier(X_train, y_train)
    logger.info("Metrics: %s", metrics)

    test_sample = [25000, 0.35, 3, 220, 0]
    prob = predict_fraud_proba(test_sample)[0]

    logger.info("Fraud Probability for test sample: %.4f", prob)
