# fraud_detection/models/model_utils.py
"""
Model utilities (I/O, evaluation, and numeric matrix helpers)

Paths are resolved using FD_PROJECT_ROOT environment variable (Option A).
If FD_PROJECT_ROOT is not set, falls back to project-relative path.

Provides:
- save_model / load_model
- model_exists
- evaluate_model (AUC, precision, recall, f1)
- split_features_and_labels
- get_numeric_matrix (select numeric columns and sort them for stable ordering)
"""

from __future__ import annotations
import os
from pathlib import Path
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------
# Resolve MODEL_DIR via FD_PROJECT_ROOT (Option A)
# ---------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    # fallback to repository layout (two levels above models/)
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "fraud_detection" / "models" / "artifacts"
MODEL_DIR_STR = str(MODEL_DIR)

# ---------------------------
# Helper: ensure model dir exists
# ---------------------------
def ensure_model_dir():
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception("Failed to create model directory %s: %s", MODEL_DIR_STR, e)
        raise

# ---------------------------
# Save model + optional scaler
# ---------------------------
def save_model(model, scaler: Optional[object] = None, model_name: str = "model.joblib") -> Tuple[str, Optional[str]]:
    """
    Save model (+ optional scaler) to artifact folder.
    Returns (model_path, scaler_path_or_None)
    """
    ensure_model_dir()
    model_path = MODEL_DIR / model_name
    scaler_path = None

    try:
        joblib.dump(model, model_path)
        logger.info("Saved model → %s", model_path)
        if scaler is not None:
            scaler_path = MODEL_DIR / f"{model_name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info("Saved scaler → %s", scaler_path)
    except Exception as e:
        logger.exception("Failed saving model/scaler: %s", e)
        raise

    return str(model_path), (str(scaler_path) if scaler_path is not None else None)

# ---------------------------
# Load model (+ optional scaler)
# ---------------------------
def load_model(model_name: str = "model.joblib", load_scaler: bool = True) -> Tuple[object, Optional[object]]:
    """
    Load model and (optionally) associated scaler from artifacts directory.
    Raises FileNotFoundError if model missing.
    """
    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        logger.error("Model not found at %s", model_path)
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        model = joblib.load(model_path)
        logger.info("Loaded model → %s", model_path)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise

    scaler = None
    if load_scaler:
        scaler_path = MODEL_DIR / f"{model_name}_scaler.joblib"
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler → %s", scaler_path)
            except Exception as e:
                logger.warning("Failed to load scaler at %s: %s", scaler_path, e)
        else:
            logger.debug("Scaler not present for model %s", model_name)

    return model, scaler

# ---------------------------
# Check model exists
# ---------------------------
def model_exists(model_name: str = "model.joblib") -> bool:
    return (MODEL_DIR / model_name).exists()

# ---------------------------
# Split dataframe into X (features) and y (labels)
# ---------------------------
def split_features_and_labels(df: pd.DataFrame, label_col: str = "label") -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a DataFrame with a label column, return (X_values, y_values).
    """
    if label_col not in df.columns:
        logger.error("Label column '%s' missing in DataFrame columns: %s", label_col, df.columns.tolist())
        raise ValueError(f"Label column '{label_col}' missing.")

    X_df = df.drop(columns=[label_col])
    y = df[label_col].values
    logger.debug("Split into X shape=%s, y shape=%s", X_df.shape, y.shape)
    return X_df.values, y

# ---------------------------
# Get numeric matrix (sorted columns) — CRITICAL for consistent ordering
# ---------------------------
def get_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Select numeric-only columns from DataFrame, sort column names, and return numpy array.
    This ensures the column ordering is deterministic across runs (important for saved models).
    Raises ValueError if no numeric columns found.
    """
    if df is None:
        logger.error("get_numeric_matrix received None DataFrame")
        raise ValueError("Input DataFrame is None")

    num_df = df.select_dtypes(include=[np.number]).copy()

    if num_df.empty:
        logger.error("No numeric columns found. Available columns: %s", df.columns.tolist())
        raise ValueError("No numeric columns available in DataFrame")

    # Sort columns alphabetically for stable ordering
    sorted_cols = sorted(num_df.columns.tolist())
    num_df = num_df.reindex(sorted_cols, axis=1)

    logger.debug("Numeric matrix columns (sorted): %s", sorted_cols)
    return num_df.values

# ---------------------------
# Evaluate model predictions
# ---------------------------
def evaluate_model(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """
    Compute metrics: ROC-AUC (when possible), precision, recall, f1.
    y_true: array-like ground-truth labels (0/1)
    y_pred_proba: array-like predicted probabilities (0..1)
    Returns dict with metrics.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    # Guard shapes
    if y_true.shape[0] != y_pred_proba.shape[0]:
        logger.error("y_true and y_pred_proba length mismatch: %s vs %s", y_true.shape, y_pred_proba.shape)
        raise ValueError("Length mismatch between y_true and y_pred_proba")

    # Binary predicted labels
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {}
    try:
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
        else:
            metrics["roc_auc"] = float("nan")
            logger.warning("ROC-AUC cannot be computed (single-class y_true).")
    except Exception as e:
        logger.exception("ROC-AUC computation failed: %s", e)
        metrics["roc_auc"] = float("nan")

    try:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception as e:
        logger.exception("Precision/Recall/F1 computation failed: %s", e)
        metrics["precision"] = metrics["recall"] = metrics["f1"] = float("nan")

    logger.info("Evaluation metrics: %s", metrics)
    try:
        logger.debug("Classification report:\n%s", classification_report(y_true, y_pred, zero_division=0))
    except Exception:
        pass

    return metrics

# ---------------------------
# Manual test block
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import pandas as pd

    logger.info("Running manual tests for model_utils.py")

    # small sample dataframe with numeric columns and label
    df = pd.DataFrame({
        "claim_amount": [10000, 500000, 12000, 8000],
        "amount_ratio": [0.2, 5.0, 0.25, 0.1],
        "claims_last_12m": [1, 10, 2, 0],
        "label": [0, 1, 0, 0]
    })

    X, y = split_features_and_labels(df, label_col="label")
    logger.info("Split shapes: X=%s, y=%s", X.shape, y.shape)

    X_num = get_numeric_matrix(df.drop(columns=["label"]))
    logger.info("Numeric matrix shape: %s", X_num.shape)

    # sample preds for evaluation test
    y_pred_proba = np.array([0.05, 0.9, 0.1, 0.2])
    metrics = evaluate_model(y, y_pred_proba, threshold=0.5)
    logger.info("Eval metrics: %s", metrics)


# ------------------------------------------------------------
# Manual Test Block
# ------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    df = pd.DataFrame({
        "claim_amount": [10000, 500000, 12000, 8000],
        "amount_ratio": [0.2, 5.0, 0.25, 0.1],
        "claims_last_12m": [1, 10, 2, 0],
        "label": [0, 1, 0, 0]
    })

    logger.info("Testing model_utils manually...")
    X, y = split_features_and_labels(df)
    X_num = get_numeric_matrix(df.drop(columns=["label"]))
    logger.info("Numeric feature matrix:\n%s", X_num)