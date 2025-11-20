# fraud_detection/decision_engine/rules.py
"""
Business rules for insurance claim evaluation (final, defensive).
Aligned with the current project schema (allowed fields):
  - claim_id
  - customer_id
  - policy_id
  - policy_id_record
  - claim_amount
  - policy_sum_insured  (used as median reference)
  - incident_date
  - description
  - phone
  - garage_id

Outputs deterministic flags and human-readable reasons for scoring/explainability.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
import pandas as pd

# safe logger fallback (project logger optional)
try:
    from fraud_detection.logging.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logger = logging.getLogger("fraud_detection.rules")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


# Tunable thresholds (PoC)
SOFT_SUSPICION_FACTOR = 3     # >3× median -> soft suspicion
HARD_THRESHOLD_FACTOR = 10    # >10× median -> hard flag


# -------------------------
# Helpers
# -------------------------
def _safe_parse_date(value: Optional[Any]):
    """
    Try to parse date-like value into pd.Timestamp; return None on failure.
    """
    if value is None:
        return None
    try:
        # pandas coerces many formats; returns NaT for invalid
        dt = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(dt) else dt
    except Exception:
        logger.debug("Date parse failed for %r", value)
        return None


# -------------------------
# Policy mismatch
# -------------------------
def policy_mismatch(policy_claim: Optional[str], policy_record: Optional[str]) -> bool:
    """
    Return True if both IDs provided and they differ.
    If either is missing, return False (missing flag handled separately).
    """
    try:
        if not policy_claim or not policy_record:
            return False
        return str(policy_claim).strip().upper() != str(policy_record).strip().upper()
    except Exception as e:
        logger.exception("policy_mismatch error: %s", e)
        return False


# -------------------------
# Claim amount checks
# -------------------------
def claim_amount_exceeds_threshold(
    claim_amount: Optional[float],
    median_amount: Optional[float],
    factor: int = HARD_THRESHOLD_FACTOR,
) -> bool:
    """
    True if claim_amount > factor * median_amount. Defensive: return False on invalid input.
    """
    try:
        if claim_amount is None or median_amount is None:
            logger.debug("claim_amount_exceeds_threshold: missing input")
            return False
        claim = float(claim_amount)
        median = float(median_amount)
        if median <= 0:
            logger.debug("claim_amount_exceeds_threshold: median <= 0")
            return False
        return claim > (factor * median)
    except Exception as e:
        logger.exception("claim_amount_exceeds_threshold error: %s", e)
        return False


def soft_suspicion(
    claim_amount: Optional[float],
    median_amount: Optional[float],
    factor: int = SOFT_SUSPICION_FACTOR,
) -> bool:
    """
    True if claim_amount > factor * median_amount (soft signal).
    """
    try:
        if claim_amount is None or median_amount is None:
            return False
        claim = float(claim_amount)
        median = float(median_amount)
        if median <= 0:
            return False
        return claim > (factor * median)
    except Exception as e:
        logger.exception("soft_suspicion error: %s", e)
        return False


# -------------------------
# Main rule extractor
# -------------------------
def extract_rule_flags(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    context keys (optional):
      - incident_date
      - policy_id (claimed)
      - policy_id_record (official)
      - claim_amount
      - median_amount_for_policy  (optional, fallback to policy_sum_insured)

    Returns flags dict expected by the scoring module.
    """

    reasons: List[str] = []

    # Defensive reads
    incident_date = context.get("incident_date")
    policy_id = context.get("policy_id")
    policy_record = context.get("policy_id_record")
    claim_amount = context.get("claim_amount")

    # median reference: accept either explicit median or policy_sum_insured
    median_amount = context.get("median_amount_for_policy") or context.get("policy_sum_insured") or None

    # Missing indicators (0/1)
    missing_incident_date = int(incident_date is None or str(incident_date).strip() == "")
    missing_policy_record = int(policy_record is None or str(policy_record).strip() == "")
    missing_median_amount = int(median_amount is None)

    # Policy mismatch
    mismatch = False
    try:
        mismatch = policy_mismatch(policy_id, policy_record)
        if mismatch:
            reasons.append("Policy ID mismatch between claimed and official record.")
        elif missing_policy_record:
            reasons.append("Official policy record missing or not provided.")
    except Exception:
        mismatch = False
        logger.exception("Error evaluating policy mismatch")

    # Claim amount thresholds
    too_high = False
    try:
        too_high = claim_amount_exceeds_threshold(claim_amount, median_amount)
        if too_high:
            reasons.append(f"Claim amount exceeds {HARD_THRESHOLD_FACTOR}× the typical policy median (hard threshold).")
    except Exception:
        too_high = False
        logger.exception("Error evaluating hard claim threshold")

    # Soft suspicion (only if not hard)
    soft_flag = False
    try:
        if not too_high:
            soft_flag = soft_suspicion(claim_amount, median_amount)
            if soft_flag:
                reasons.append(f"Claim amount unusually high (> {SOFT_SUSPICION_FACTOR}× median).")
    except Exception:
        soft_flag = False
        logger.exception("Error evaluating soft suspicion")

    # Missing field notes
    if missing_incident_date:
        reasons.append("Incident date is missing or invalid.")
    if missing_median_amount:
        reasons.append("Median amount for this policy unavailable; amount checks limited.")

    # Final flags. Keep 'policy_expired' present but False for compatibility.
    flags = {
        "policy_expired": False,
        "policy_mismatch": bool(mismatch),
        "claim_too_high": bool(too_high),
        "soft_suspicion": bool(soft_flag),
        "missing_policy_record": int(missing_policy_record),
        "missing_incident_date": int(missing_incident_date),
        "missing_median_amount": int(missing_median_amount),
        "reasons": reasons,
    }

    logger.debug("extract_rule_flags -> %s", flags)
    return flags


# -------------------------
# Manual test
# -------------------------
if __name__ == "__main__":
    import logging as _l

    _l.basicConfig(level=_l.INFO)

    sample = {
        "claim_id": "T1001",
        "customer_id": "C001",
        "policy_id": "POL-111",
        "policy_id_record": "POL-111",
        "claim_amount": 4500,
        "policy_sum_insured": 50000,
        "incident_date": "2024-06-10",
        "description": "Minor scratch on left door while parking.",
        "phone": "9876543210",
        "garage_id": "GAR-001",
    }

    print("RULE FLAGS:", extract_rule_flags(sample))
