# fraud_detection/preprocessing/schema_validator.py

import os
from typing import Any, Dict, Tuple, Optional
from pathlib import Path

from fraud_detection.logging.logger import get_logger
from fraud_detection.schemas.claim_schema import ClaimSchema
from fraud_detection.utils.sanitizers import sanitize_dict
from fraud_detection.preprocessing.field_normalizer import normalize_claim_dict

logger = get_logger(__name__)

# Project root env-first
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_plain_dict(pydantic_obj) -> Dict[str, Any]:
    """
    Return a plain python dict from either pydantic v1 or v2 model instance.
    - v1: .dict()
    - v2: .model_dump()
    """
    if pydantic_obj is None:
        return {}
    if hasattr(pydantic_obj, "model_dump"):
        # pydantic v2
        return pydantic_obj.model_dump()
    if hasattr(pydantic_obj, "dict"):
        # pydantic v1
        return pydantic_obj.dict()
    # Fallback: try to coerce
    try:
        return dict(pydantic_obj)
    except Exception:
        return {}


def validate_claim_dict(claim: Dict[str, Any], raise_on_error: bool = False) -> Tuple[bool, Optional[Any], Optional[Dict[str, Any]]]:
    """
    Validate a claim dict using ClaimSchema.
    Returns (is_valid, claim_obj, errors)
      - claim_obj is Pydantic model instance on success
      - errors is a dict or exception message on failure

    If raise_on_error=True, will raise the validation exception.
    """
    try:
        # First normalize fields (conservative)
        normalized = normalize_claim_dict(claim)

        # Validate via schema
        claim_obj = ClaimSchema(**normalized)

        logger.debug("validate_claim_dict: success for claim_id=%s", getattr(claim_obj, "claim_id", None))
        return True, claim_obj, None
    except Exception as e:
        logger.warning("validate_claim_dict failed: %s", e)
        if raise_on_error:
            raise
        # return sanitized partial for debugging
        try:
            partial = sanitize_dict(normalized if 'normalized' in locals() else claim)
        except Exception:
            partial = {}
        return False, None, {"error": str(e), "partial": partial}


def validate_and_sanitize(claim: Dict[str, Any], raise_on_error: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience: validate and return sanitized python dict (JSON-safe).
    On failure returns (False, partial_sanitized_dict)
    """
    ok, claim_obj, errors = validate_claim_dict(claim, raise_on_error=raise_on_error)
    if ok and claim_obj is not None:
        plain = to_plain_dict(claim_obj)
        sanitized = sanitize_dict(plain)
        return True, sanitized
    else:
        # return partial sanitized input
        return False, errors.get("partial", {}) if errors else {}
