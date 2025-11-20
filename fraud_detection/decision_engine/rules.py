# fraud_detection/decision_engine/rules.py
"""
Business rules for insurance claim evaluation.

Returns deterministic flags and human-readable reasons that
the decision engine can consume (scoring/explainability).
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import pandas as pd

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# Thresholds (PoC / tunable)
SOFT_SUSPICION_FACTOR = 3     # >3x median => soft suspicion
HARD_THRESHOLD_FACTOR = 10    # >10x median => hard override


# -------------------------
# Safe date parsing helper
# -------------------------
def _safe_parse_date(value: Optional[Any]):
    if value is None or (isinstance(value, str) and str(value).strip() == ""):
        return None
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            return None
        return dt
    except Exception as e:
        logger.debug("Date parse error for %s: %s", value, e)
        return None


# -------------------------
# Rule: policy expired
# -------------------------
def check_policy_expiry(policy_end_date: Optional[Any], incident_date: Optional[Any]) -> bool:
    end_dt = _safe_parse_date(policy_end_date)
    inc_dt = _safe_parse_date(incident_date)

    if end_dt is None or inc_dt is None:
        return False

    try:
        return bool(inc_dt > end_dt)
    except Exception:
        logger.exception("Error comparing dates in check_policy_expiry")
        return False


# -------------------------
# Rule: policy mismatch
# -------------------------
def policy_mismatch(policy_id_claim: Optional[str], policy_id_record: Optional[str]) -> bool:
    if not policy_id_claim or not policy_id_record:
        return False

    try:
        c = str(policy_id_claim).strip().upper()
        r = str(policy_id_record).strip().upper()
        return c != r
    except Exception:
        logger.exception("Error comparing policy IDs")
        return False


# -------------------------
# Rule: claim amount threshold (hard)
# -------------------------
def claim_amount_exceeds_threshold(claim_amount: Optional[float], median_amount: Optional[float], factor: int = HARD_THRESHOLD_FACTOR) -> bool:
    try:
        if claim_amount is None or median_amount is None:
            return False
        claim_amount = float(claim_amount)
        median_amount = float(median_amount)
        if median_amount <= 0:
            return False
        return claim_amount > (factor * median_amount)
    except Exception:
        logger.exception("Error in claim_amount_exceeds_threshold")
        return False


# -------------------------
# Soft suspicion (> SOFT_SUSPICION_FACTOR)
# -------------------------
def soft_suspicion(claim_amount: Optional[float], median_amount: Optional[float], factor: int = SOFT_SUSPICION_FACTOR) -> bool:
    try:
        if claim_amount is None or median_amount is None:
            return False
        claim_amount = float(claim_amount)
        median_amount = float(median_amount)
        if median_amount <= 0:
            return False
        return claim_amount > (factor * median_amount)
    except Exception:
        logger.exception("Error in soft_suspicion")
        return False


# -------------------------
# Main: extract rule flags
# -------------------------
def extract_rule_flags(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    context keys (optional):
      - policy_end_date
      - incident_date
      - policy_id (claimed)
      - policy_id_record (official)
      - claim_amount
      - median_amount_for_policy

    Returns:
      dict with boolean flags, integer missing indicators and 'reasons' list
    """
    reasons: List[str] = []

    policy_end = context.get("policy_end_date")
    incident_date = context.get("incident_date")
    policy_id = context.get("policy_id") or context.get("policy_id_claim")
    policy_record = context.get("policy_id_record")
    claim_amount = context.get("claim_amount")
    median_amount = context.get("median_amount_for_policy")

    missing_incident_date = int(incident_date is None or str(incident_date).strip() == "")
    missing_policy_end_date = int(policy_end is None or str(policy_end).strip() == "")
    missing_policy_record = int(policy_record is None or str(policy_record).strip() == "")
    missing_median_amount = int(median_amount is None or str(median_amount).strip() == "")

    # Evaluate rules
    try:
        policy_expired = check_policy_expiry(policy_end, incident_date)
        if policy_expired:
            reasons.append("Policy expired before the incident date.")
    except Exception:
        policy_expired = False
        logger.exception("Error evaluating policy expiry")

    try:
        mismatch = policy_mismatch(policy_id, policy_record)
        if mismatch:
            reasons.append("Claimed policy ID does not match the official policy record.")
        elif missing_policy_record:
            reasons.append("Official policy record missing.")
    except Exception:
        mismatch = False
        logger.exception("Error evaluating policy mismatch")

    try:
        too_high = claim_amount_exceeds_threshold(claim_amount, median_amount, factor=HARD_THRESHOLD_FACTOR)
        if too_high:
            reasons.append(f"Claim amount is greater than {HARD_THRESHOLD_FACTOR}× the median amount for this policy (hard threshold).")
    except Exception:
        too_high = False
        logger.exception("Error evaluating hard claim threshold")

    try:
        soft = soft_suspicion(claim_amount, median_amount, factor=SOFT_SUSPICION_FACTOR)
        if soft and not too_high:
            reasons.append(f"Claim amount is unusually high (> {SOFT_SUSPICION_FACTOR}× median).")
    except Exception:
        soft = False
        logger.exception("Error evaluating soft suspicion")

    # Add missing field reasons
    if missing_incident_date:
        reasons.append("Incident date is missing or invalid.")
    if missing_policy_end_date:
        reasons.append("Policy end date is missing.")
    if missing_median_amount:
        reasons.append("Median amount for policy unavailable for amount checks.")

    flags = {
        "policy_expired": bool(policy_expired),
        "policy_mismatch": bool(mismatch),
        "claim_too_high": bool(too_high),
        "soft_suspicion": bool(soft),
        "missing_policy_record": int(missing_policy_record),
        "missing_incident_date": int(missing_incident_date),
        "missing_policy_end_date": int(missing_policy_end_date),
        "missing_median_amount": int(missing_median_amount),
        "reasons": reasons
    }

    logger.debug("extract_rule_flags -> %s", flags)
    return flags


# Manual test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ctx = {
        "policy_end_date": "2023-12-31",
        "incident_date": "2024-02-12",
        "policy_id": "POL-123",
        "policy_id_record": "POL-123",
        "claim_amount": 500000,
        "median_amount_for_policy": 10000
    }
    print("RULE FLAGS:", extract_rule_flags(test_ctx))
