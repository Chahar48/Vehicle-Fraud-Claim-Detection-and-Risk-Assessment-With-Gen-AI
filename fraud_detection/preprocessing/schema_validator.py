"""
Improved Schema Validator for fraud-detection-genai
---------------------------------------------------

Fixes:
 - Accepts extra fields (e.g., _field_confidence from field_normalizer)
 - Never drops normalized values when Pydantic validation fails
 - Returns full normalized claim, not partial dict
 - Treats schema validation as soft check (warning only)
"""

from __future__ import annotations
import os
from typing import Any, Dict, Tuple, Optional
from pathlib import Path

try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("schema_validator")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

from fraud_detection.schemas.claim_schema import ClaimSchema
from fraud_detection.utils.sanitizers import sanitize_dict

# robust normalize wrapper
try:
    from fraud_detection.preprocessing.field_normalizer import normalize_claim_dict
except Exception:
    def normalize_claim_dict(x): return x

FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_plain_dict(obj) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    try:
        return dict(obj)
    except:
        return {}


def validate_claim_dict(
    claim: Dict[str, Any],
    raise_on_error: bool = False
) -> Tuple[bool, Optional[Any], Optional[Dict[str, Any]]]:
    """
    Validate dict using ClaimSchema.
    Now SOFT validation:
      - Never blocks pipeline
      - Never drops fields
      - Only logs warnings
    """
    normalized = None
    try:
        normalized = normalize_claim_dict(claim)

        # Pydantic schema: allow extra fields
        # -------------------------------------------------
        # The ClaimSchema must have: model_config = {"extra": "allow"}
        # If not present, this still soft-falls back.
        # -------------------------------------------------

        claim_obj = ClaimSchema(**normalized)
        return True, claim_obj, None

    except Exception as e:
        logger.warning("Soft schema validation warning: %s", e)

        if raise_on_error:
            raise

        # DO NOT drop normalized fields — return full normalized dict
        sanitized = sanitize_dict(normalized or claim)
        return False, None, {"error": str(e), "partial": sanitized}


def validate_and_sanitize(
    claim: Dict[str, Any],
    raise_on_error: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (ok, validated_dict)
    If validation fails, still return FULL normalized sanitized dict.
    """
    ok, claim_obj, errors = validate_claim_dict(claim, raise_on_error=raise_on_error)

    if ok and claim_obj is not None:
        plain = to_plain_dict(claim_obj)
        return True, sanitize_dict(plain)

    # Soft fallback: return full normalized (not partial)
    return False, errors.get("partial", {}) if errors else {}
