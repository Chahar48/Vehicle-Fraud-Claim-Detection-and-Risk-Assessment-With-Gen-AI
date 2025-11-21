# fraud_detection/ingestion/api_ingest.py
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import difflib
import logging

from fraud_detection.logging.logger import get_logger
from fraud_detection.schemas.claim_schema import ClaimSchema
from fraud_detection.schema.sanitizers import sanitize_dict
from fraud_detection.ingestion.file_saver import save_files

logger = get_logger(__name__)

FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

CANONICAL = [
    "claim_id","customer_id","policy_id","policy_id_record","claim_amount",
    "policy_sum_insured","incident_date","description","phone","garage_id"
]
# reuse alias map from csv_ingest for consistency (small copy)
_ALIAS = {
    "claimid": "claim_id", "claim id": "claim_id", "claim": "claim_id",
    "customer id":"customer_id", "customerid":"customer_id",
    "policy id": "policy_id", "policyid": "policy_id",
    "official policy id": "policy_id_record", "policy id record": "policy_id_record",
    "claim amount":"claim_amount", "amount":"claim_amount",
    "policy sum insured":"policy_sum_insured", "policy sum":"policy_sum_insured",
    "incident date":"incident_date","date":"incident_date",
    "desc":"description","description":"description",
    "phone":"phone","mobile":"phone","telephone":"phone",
    "garage id":"garage_id","garageid":"garage_id"
}

FUZZY_CUTOFF = 0.7

def _clean_key(k: Any) -> str:
    if k is None:
        return ""
    s = str(k).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

def _map_key(k: str) -> str:
    k2 = _clean_key(k)
    if not k2:
        return k
    if k2 in _ALIAS:
        return _ALIAS[k2]
    if k2 in CANONICAL:
        return k2
    # fuzzy
    cand = difflib.get_close_matches(k2, list(_ALIAS.keys()), n=1, cutoff=FUZZY_CUTOFF)
    if cand:
        return _ALIAS[cand[0]]
    cand2 = difflib.get_close_matches(k2, CANONICAL, n=1, cutoff=FUZZY_CUTOFF)
    if cand2:
        return cand2[0]
    # fallback: numeric-like keys or short alnum -> keep cleaned as is
    return re.sub(r"\s+", "_", k2)

def _normalize_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in payload.items():
        mapped = _map_key(k)
        out[mapped] = v
    return out

MINIMAL_REQUIRED = ["claim_id", "claim_amount", "incident_date"]

def _check_minimal_required(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing = [f for f in MINIMAL_REQUIRED if payload.get(f) in (None, "", [], {})]
    return (len(missing) == 0, missing)

def handle_api_payload(payload_json_or_dict: Any, files: List[Any] = None) -> Tuple[Dict[str, Any], List[str]]:
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

    # Normalize keys using fuzzy mapping
    standardized = _normalize_keys(payload)

    ok, missing = _check_minimal_required(standardized)
    if not ok:
        raise ValueError(f"Missing required fields: {missing}")

    try:
        claim_obj = ClaimSchema(**standardized)
        # model_dump or dict
        claim_dict = claim_obj.model_dump() if hasattr(claim_obj, "model_dump") else claim_obj.dict()
    except Exception as e:
        logger.exception("Claim schema validation failed")
        partial = sanitize_dict(standardized)
        raise ValueError(f"Claim schema validation failed: {e}. Partial: {partial}")

    claim_dict = sanitize_dict(claim_dict)
    saved_paths = []
    if files:
        saved = save_files(files)
        saved_paths = [str(p.resolve()) for p in saved]

    logger.info("Handled API payload for claim_id=%s saved_files=%d", claim_dict.get("claim_id"), len(saved_paths))
    return claim_dict, saved_paths
