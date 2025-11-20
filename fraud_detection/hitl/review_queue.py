# fraud_detection/hitl/review_queue.py
"""
HITL review queue manager.

Stores queue as CSV under:
  <FD_PROJECT_ROOT>/data/hitl/review_queue.csv

Design goals:
- Use FD_PROJECT_ROOT env var (fallback to package-relative)
- Robust CSV read/write with pandas
- JSON payload column that stores the full claim/result object
- Deterministic queue_id generation (timestamp-based)
- Avoid duplicates by default when enqueueing
- Manual-testable via __main__
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading

import pandas as pd

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------
# Project root: env-first (Option A)
# ---------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]  # fallback: repo root

HITL_DIR = PROJECT_ROOT / "data" / "hitl"
HITL_DIR.mkdir(parents=True, exist_ok=True)

QUEUE_PATH = HITL_DIR / "review_queue.csv"

# Queue columns (canonical)
QUEUE_COLUMNS = [
    "queue_id",
    "claim_id",
    "enqueued_at",
    "status",          # pending / in_progress / reviewed / error
    "assigned_to",
    "payload",         # JSON string of the claim/result dict
    "notes",
    "updated_at"
]

# simple lock for file operations to avoid races in single-process environments
_lock = threading.Lock()


# ---------------------------
# Internal helpers
# ---------------------------
def _ensure_queue_exists() -> None:
    if not QUEUE_PATH.exists():
        df = pd.DataFrame(columns=QUEUE_COLUMNS)
        df.to_csv(QUEUE_PATH, index=False)
        logger.info("Created new HITL queue at %s", QUEUE_PATH)


def _load_queue_df() -> pd.DataFrame:
    """Load queue CSV into DataFrame (string dtype to avoid NaN issues)."""
    _ensure_queue_exists()
    try:
        df = pd.read_csv(QUEUE_PATH, dtype=str).fillna("")
        # Ensure all expected columns exist
        for c in QUEUE_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        # Keep canonical order
        df = df[QUEUE_COLUMNS]
        return df
    except Exception as e:
        logger.exception("Failed to load HITL queue CSV: %s", e)
        # If corrupted, return empty frame with columns
        return pd.DataFrame(columns=QUEUE_COLUMNS)


def _save_queue_df(df: pd.DataFrame) -> None:
    """Save queue DataFrame to CSV (atomic write using temp file)."""
    # Ensure canonical columns
    for c in QUEUE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[QUEUE_COLUMNS]

    tmp_path = QUEUE_PATH.with_suffix(".tmp")
    try:
        with _lock:
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(QUEUE_PATH)
    except Exception as e:
        logger.exception("Failed to save HITL queue CSV: %s", e)
        raise


# ---------------------------
# Public API
# ---------------------------
def enqueue_for_review(claim_result: Dict[str, Any], allow_duplicates: bool = False) -> str:
    """
    Add claim_result to HITL queue and return queue_id.
    - claim_result must have 'claim_id' key (non-empty).
    - If allow_duplicates is False, will not add a second pending item for same claim_id.
    """
    if not isinstance(claim_result, dict):
        raise ValueError("claim_result must be a dict")

    claim_id = str(claim_result.get("claim_id", "") or "").strip()
    if not claim_id:
        raise ValueError("claim_result must include non-empty 'claim_id' to enqueue")

    _ensure_queue_exists()
    df = _load_queue_df()

    # check existing pending duplicate
    if not allow_duplicates:
        mask = (df["status"] == "pending") & (df["claim_id"].astype(str) == claim_id)
        if mask.any():
            existing = df.loc[mask].iloc[0]
            logger.info("Duplicate pending entry found for claim_id=%s queue_id=%s", claim_id, existing["queue_id"])
            return str(existing["queue_id"])

    queue_id = f"q_{int(datetime.utcnow().timestamp() * 1000)}"
    now = datetime.utcnow().isoformat()

    try:
        payload_str = json.dumps(claim_result, default=str)
    except Exception:
        # fallback to safe repr string
        payload_str = json.dumps({"claim_id": claim_id, "note": "payload not JSON serializable"}, default=str)

    new_row = {
        "queue_id": queue_id,
        "claim_id": claim_id,
        "enqueued_at": now,
        "status": "pending",
        "assigned_to": "",
        "payload": payload_str,
        "notes": "",
        "updated_at": now
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_queue_df(df)

    logger.info("Enqueued claim_id=%s queue_id=%s", claim_id, queue_id)
    return queue_id


def list_pending_reviews(limit: int = 200) -> List[Dict[str, Any]]:
    """
    Return list of pending queue items (decoded payloads).
    """
    df = _load_queue_df()
    pending = df[df["status"] == "pending"].sort_values("enqueued_at").head(limit)

    out: List[Dict[str, Any]] = []
    for _, r in pending.iterrows():
        try:
            payload = json.loads(r["payload"]) if r.get("payload") else {}
        except Exception:
            payload = {}
        out.append({
            "queue_id": r["queue_id"],
            "claim_id": r["claim_id"],
            "enqueued_at": r["enqueued_at"],
            "notes": r.get("notes", ""),
            "payload": payload
        })
    return out


def get_next_review(assign_to: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Claim (pop) the oldest pending item and mark it in_progress.
    Returns the queue entry dict or None if no pending.
    """
    df = _load_queue_df()
    pending = df[df["status"] == "pending"]
    if pending.empty:
        logger.debug("No pending HITL items")
        return None

    pending = pending.sort_values("enqueued_at")
    row = pending.iloc[0].to_dict()
    queue_id = row["queue_id"]

    # update df
    idxs = df.index[df["queue_id"] == queue_id].tolist()
    if not idxs:
        logger.error("Queue id %s disappeared unexpectedly", queue_id)
        return None
    idx = idxs[0]

    df.at[idx, "status"] = "in_progress"
    df.at[idx, "assigned_to"] = assign_to or ""
    df.at[idx, "updated_at"] = datetime.utcnow().isoformat()
    _save_queue_df(df)

    try:
        payload = json.loads(row.get("payload") or "{}")
    except Exception:
        payload = {}

    logger.info("Claim claimed for review queue_id=%s assigned_to=%s", queue_id, assign_to or "")
    return {
        "queue_id": queue_id,
        "claim_id": row.get("claim_id"),
        "assigned_to": assign_to,
        "enqueued_at": row.get("enqueued_at"),
        "payload": payload,
        "notes": row.get("notes", "")
    }


def mark_review_completed(queue_id: str, reviewer_id: str, result_summary: Optional[Dict[str, Any]] = None) -> None:
    """
    Mark an item reviewed (status=reviewed), append result_summary to notes.
    """
    df = _load_queue_df()
    mask = df["queue_id"] == queue_id
    if not mask.any():
        logger.error("mark_review_completed: queue_id not found: %s", queue_id)
        raise ValueError(f"queue_id={queue_id} not found")

    idx = df.index[mask][0]
    df.at[idx, "status"] = "reviewed"
    df.at[idx, "assigned_to"] = reviewer_id or ""
    existing_notes = str(df.at[idx, "notes"] or "")
    if result_summary:
        try:
            add = json.dumps(result_summary, default=str)
        except Exception:
            add = str(result_summary)
        df.at[idx, "notes"] = (existing_notes + " | " + add).strip(" |")
    df.at[idx, "updated_at"] = datetime.utcnow().isoformat()
    _save_queue_df(df)
    logger.info("Marked reviewed queue_id=%s by reviewer=%s", queue_id, reviewer_id)


def mark_review_error(queue_id: str, notes: str = "") -> None:
    df = _load_queue_df()
    mask = df["queue_id"] == queue_id
    if not mask.any():
        logger.error("mark_review_error: queue_id not found: %s", queue_id)
        raise ValueError(f"queue_id={queue_id} not found")
    idx = df.index[mask][0]
    df.at[idx, "status"] = "error"
    df.at[idx, "notes"] = notes
    df.at[idx, "updated_at"] = datetime.utcnow().isoformat()
    _save_queue_df(df)
    logger.info("Marked error queue_id=%s", queue_id)


# ---------------------------
# Manual test
# ---------------------------
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    logger.info("HITL review_queue manual test starting")
    sample = {"claim_id": "TEST_CLAIM_01", "final_score": 0.72, "action": "manual_review", "input": {}}

    qid = enqueue_for_review(sample, allow_duplicates=False)
    logger.info("Enqueued qid=%s", qid)

    pend = list_pending_reviews()
    logger.info("Pending count=%d", len(pend))

    nxt = get_next_review(assign_to="rev_1")
    logger.info("Got next review: %s", nxt)

    if nxt:
        mark_review_completed(nxt["queue_id"], reviewer_id="rev_1", result_summary={"label": 1})
        logger.info("Completed review for %s", nxt["queue_id"])

    logger.info("Manual test finished")
