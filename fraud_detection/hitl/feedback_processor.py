# fraud_detection/hitl/feedback_processor.py
"""
HITL feedback processor.

Responsibilities:
- Save reviewer labels to CSV under <FD_PROJECT_ROOT>/data/labels/labels.csv
- Mark the corresponding review_queue item reviewed when possible
- Provide helper to export merged results for retraining

Design:
- Uses FD_PROJECT_ROOT env var (Option A)
- Uses fraud_detection.hitl.review_queue functions to mark reviewed (local import inside functions to avoid circular imports)
- Uses pandas for CSV handling (append gracefully)
"""

from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

import pandas as pd

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------
# Project root (env-first)
# ---------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_PATH = LABELS_DIR / "labels.csv"

# ---------------------------
# Append label locally (pandas)
# ---------------------------
def _append_label_local(record: Dict[str, Any]) -> Path:
    """
    Append a single label record (dict) to LABELS_PATH CSV.
    Ensures columns are consistent. Returns path.
    """
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    rec_df = pd.DataFrame([record])

    if LABELS_PATH.exists():
        try:
            existing = pd.read_csv(LABELS_PATH, dtype=str)
            out_df = pd.concat([existing, rec_df], ignore_index=True, sort=False)
        except Exception:
            # if existing CSV is corrupted, overwrite with new
            out_df = rec_df
    else:
        out_df = rec_df

    # Ensure deterministic column order (claim_id first if present)
    cols = list(out_df.columns)
    if "claim_id" in cols:
        cols.remove("claim_id")
        cols = ["claim_id"] + cols
    out_df.to_csv(LABELS_PATH, index=False)
    logger.info("Appended label to %s", LABELS_PATH)
    return LABELS_PATH


# ---------------------------
# Public: save review result
# ---------------------------
def save_review_result(claim_id: str, label: int, reviewer_id: str, notes: str = "", queue_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Save the review label and (if possible) mark the queue item as reviewed.

    Returns the saved record dict.
    """
    if not claim_id:
        raise ValueError("claim_id is required")

    ts = datetime.utcnow().isoformat()
    record = {
        "claim_id": str(claim_id),
        "label": int(label),
        "reviewer_id": str(reviewer_id),
        "notes": str(notes),
        "queue_id": queue_id or "",
        "timestamp": ts
    }

    # 1) Save label locally
    try:
        _append_label_local(record)
    except Exception as e:
        logger.exception("Failed to append label locally: %s", e)
        raise

    # 2) Attempt to mark queue reviewed (import lazily to avoid circular import)
    try:
        from fraud_detection.hitl import review_queue
        if queue_id:
            try:
                review_queue.mark_review_completed(queue_id, reviewer_id, result_summary={"label": label, "notes": notes})
            except Exception as e:
                logger.warning("Failed to mark queue_id %s reviewed: %s", queue_id, e)
        else:
            # Try to find first matching queue entry for this claim_id and mark it reviewed
            try:
                df = review_queue._load_queue_df()
                matches = df[df["claim_id"].astype(str) == str(claim_id)]
                if not matches.empty:
                    qid = matches.iloc[0]["queue_id"]
                    try:
                        review_queue.mark_review_completed(qid, reviewer_id, result_summary={"label": label, "notes": notes})
                    except Exception as e:
                        logger.warning("Failed to mark found queue_id %s reviewed: %s", qid, e)
            except Exception as e:
                logger.debug("Could not auto-mark queue reviewed: %s", e)
    except Exception as e:
        logger.debug("review_queue import failed or marking skipped: %s", e)

    logger.info("Saved review result for claim_id=%s label=%s reviewer=%s", claim_id, label, reviewer_id)
    return record


def get_labeled_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Return DataFrame of labels (empty frame if none).
    """
    p = Path(path) if path else LABELS_PATH
    if not p.exists():
        return pd.DataFrame(columns=["claim_id", "label", "reviewer_id", "notes", "queue_id", "timestamp"])
    try:
        return pd.read_csv(p, dtype=str)
    except Exception:
        logger.exception("Failed to read labels CSV at %s", p)
        return pd.DataFrame(columns=["claim_id", "label", "reviewer_id", "notes", "queue_id", "timestamp"])


def export_for_retraining(results_summary_path: Optional[Path] = None, labels_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Merge labels with results summary (if results_summary exists).
    Returns merged DataFrame for retraining.
    """
    results_path = Path(results_summary_path) if results_summary_path else (PROJECT_ROOT / "data" / "results" / "results_summary.csv")
    labels_p = Path(labels_path) if labels_path else LABELS_PATH

    labels_df = get_labeled_data(labels_p)
    if not results_path.exists():
        logger.info("Results summary not found at %s; returning only labels", results_path)
        return labels_df

    try:
        results_df = pd.read_csv(results_path, dtype=str)
    except Exception:
        try:
            results_df = pd.read_csv(results_path)
        except Exception as e:
            logger.exception("Failed to read results_summary at %s: %s", results_path, e)
            return labels_df

    merged = pd.merge(labels_df, results_df, how="left", on="claim_id")
    return merged


# ---------------------------
# Manual test
# ---------------------------
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    logger.info("HITL feedback_processor manual test starting")

    # Save label (no queue_id)
    rec = save_review_result("TEST_CLAIM_123", 1, "rev_100", notes="looks fraudulent", queue_id=None)
    print("Saved label record:", rec)

    df = get_labeled_data()
    print("Labels DF preview:\n", df.tail(5))

    # Try export (will return labels if results_summary not present)
    merged = export_for_retraining()
    print("Merged (preview):\n", merged.head())

    logger.info("Manual test finished")
