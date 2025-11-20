"""
csv_ingest.py

Responsibilities:
- Load CSV (reads as strings to avoid pandas dtype surprises)
- Validate required columns (configurable)
- Persist the original CSV to RAW via file_saver.save_file()
- Validate & normalize each row using ClaimSchema and sanitize_dict
- Return list of claim dicts ready for downstream processing
"""

from pathlib import Path
from typing import List, Dict, Any
import os
import yaml

import pandas as pd

from fraud_detection.logging.logger import get_logger
from fraud_detection.schemas.claim_schema import ClaimSchema
from fraud_detection.utils.sanitizers import sanitize_dict
from fraud_detection.ingestion.file_saver import save_file

logger = get_logger(__name__)

# ---------------------------
# Project root (env or fallback)
# ---------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------
# Default required columns (can be customized later)
# ---------------------------
DEFAULT_REQUIRED_COLUMNS = ["claim_id", "policy_id", "claim_amount", "incident_date", "description"]

def _load_required_columns() -> List[str]:
    cfg = PROJECT_ROOT / "fraud_detection" / "configs" / "app.yaml"
    try:
        if cfg.exists():
            with cfg.open("r", encoding="utf-8") as fh:
                conf = yaml.safe_load(fh)
            # For now, app.yaml doesn't list columns; keep default.
    except Exception:
        pass
    return DEFAULT_REQUIRED_COLUMNS

REQUIRED_COLUMNS = _load_required_columns()


# ---------------------------
# CSV functions
# ---------------------------
def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def load_claims_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # Read as str to avoid pandas auto dtype that can produce NaN/inf/np types
    df = pd.read_csv(p, dtype=str)
    validate_schema(df)
    logger.info("Loaded CSV (%d rows): %s", len(df), path)
    return df


def _validate_row_to_claim(row: pd.Series) -> Dict[str, Any]:
    raw = row.to_dict()
    try:
        claim_obj = ClaimSchema(**raw)
        claim_dict = claim_obj.dict()
        claim_dict = sanitize_dict(claim_dict)
        return claim_dict
    except Exception as e:
        logger.exception("Row schema validation failed")
        partial = sanitize_dict(raw)
        partial["_schema_error"] = str(e)
        return partial


def ingest_file(path: str) -> List[Dict[str, Any]]:
    """
    Loads CSV, saves raw CSV to RAW folder, returns list of claim dicts.
    """
    df = load_claims_csv(path)
    saved = save_file(path)  # persists original csv into RAW folder
    logger.info("Persisted raw CSV to: %s", saved)

    claims = []
    for _, row in df.iterrows():
        claims.append(_validate_row_to_claim(row))

    logger.info("CSV ingestion prepared %d claim(s)", len(claims))
    return claims
