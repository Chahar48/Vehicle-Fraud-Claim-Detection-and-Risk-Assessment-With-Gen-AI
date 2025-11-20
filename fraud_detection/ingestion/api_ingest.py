"""
api_ingest.py

Responsibilities:
- Parse and minimally validate incoming API payload (JSON or dict)
- Use ClaimSchema for full validation & normalization
- Sanitize output via sanitize_dict
- Save uploaded files (via file_saver.save_files)
- Return (claim_dict, list_of_saved_paths)
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fraud_detection.logging.logger import get_logger
from fraud_detection.schemas.claim_schema import ClaimSchema
from fraud_detection.utils.sanitizers import sanitize_dict
from fraud_detection.ingestion.file_saver import save_files

logger = get_logger(__name__)

# ---------------------------
# Project root resolution
# ---------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------
# Minimal required fields (presence check)
# ---------------------------
MINIMAL_REQUIRED = ["claim_id", "claim_amount", "incident_date"]


def _check_minimal_required(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing = [f for f in MINIMAL_REQUIRED if payload.get(f) in (None, "", [], {})]
    return (len(missing) == 0, missing)


def handle_api_payload(payload_json_or_dict: Any, files: List[Any] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Accepts payload (JSON string or dict) and optional files (UploadFile objects).
    Returns (validated_sanitized_claim_dict, list_of_saved_paths).
    Raises ValueError on bad payload.
    """
    # Parse payload
    if isinstance(payload_json_or_dict, str):
        try:
            payload = json.loads(payload_json_or_dict)
        except Exception as e:
            logger.exception("Invalid JSON payload")
            raise ValueError(f"Invalid JSON payload: {e}")
    elif isinstance(payload_json_or_dict, dict):
        payload = payload_json_or_dict
    else:
        raise ValueError("payload must be JSON string or dict")

    ok, missing = _check_minimal_required(payload)
    if not ok:
        raise ValueError(f"Missing required fields: {missing}")

    # Normalize keys (strip)
    standardized = {str(k).strip(): v for k, v in payload.items()}

    # Full validation & normalization via pydantic ClaimSchema
    try:
        claim_obj = ClaimSchema(**standardized)
        claim_dict = claim_obj.dict()
    except Exception as e:
        logger.exception("Claim schema validation failed")
        # include sanitized partial payload for debugging
        partial = sanitize_dict(standardized)
        raise ValueError(f"Claim schema validation failed: {e}. Partial: {partial}")

    # Sanitize to make JSON-safe
    claim_dict = sanitize_dict(claim_dict)

    # Save any attached files
    saved_paths = []
    if files:
        saved = save_files(files)
        saved_paths = [str(p.resolve()) for p in saved]

    logger.info("Handled API payload for claim_id=%s saved_files=%d", claim_dict.get("claim_id"), len(saved_paths))
    return claim_dict, saved_paths
