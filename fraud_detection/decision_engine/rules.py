# fraud_detection/decision_engine/rules.py
"""
Final Business Rules for Insurance Claim Evaluation
---------------------------------------------------

Implements deterministic, explainable rule flags used by the scoring engine.

Rules covered:
 - Missing critical fields
 - Policy mismatch
 - Claim amount vs median/policy_sum_insured
 - Soft suspicion (>3× median)
 - Hard threshold (>10× median)
 - Date validation (invalid or future incident dates)

Outputs:
  {
     "policy_mismatch": 0/1,
     "claim_too_high": 0/1,
     "soft_suspicion": 0/1,
     "missing_policy_record": 0/1,
     "missing_incident_date": 0/1,
     "missing_median_amount": 0/1,
     "invalid_incident_date": 0/1,
     "incident_date_in_future": 0/1,
     "reasons": [ ... ]
  }
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List
import pandas as pd

try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("rules")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


# Rule thresholds
SOFT_SUSPICION_FACTOR = 3    # >3× median = soft suspicion
HARD_THRESHOLD_FACTOR = 10   # >10× median = hard threshold


# -------------------------
# Helpers
# -------------------------
def _parse_date(v: Any) -> Optional[pd.Timestamp]:
    if v is None:
        return None
    try:
        dt = pd.to_datetime(v, errors="coerce")
        return None if pd.isna(dt) else dt
    except Exception:
        return None


def _is_future_date(dt: Optional[pd.Timestamp]) -> bool:
    if dt is None:
        return False
    try:
        today = pd.Timestamp.utcnow().normalize()
        return dt > today
    except Exception:
        return False


# -------------------------
# Policy mismatch
# -------------------------
def policy_mismatch(policy_claim: Optional[str], policy_record: Optional[str]) -> bool:
    try:
        if not policy_claim or not policy_record:
            return False
        return str(policy_claim).strip().upper() != str(policy_record).strip().upper()
    except Exception:
        return False


# -------------------------
# Amount-based rules
# -------------------------
def claim_too_high(claim_amount: Optional[float], median_amount: Optional[float]) -> bool:
    try:
        if claim_amount is None or median_amount is None or median_amount <= 0:
            return False
        return float(claim_amount) > (HARD_THRESHOLD_FACTOR * float(median_amount))
    except Exception:
        return False


def soft_suspicion(claim_amount: Optional[float], median_amount: Optional[float]) -> bool:
    try:
        if claim_amount is None or median_amount is None or median_amount <= 0:
            return False
        return float(claim_amount) > (SOFT_SUSPICION_FACTOR * float(median_amount))
    except Exception:
        return False


# -------------------------
# MAIN RULE EXTRACTOR
# -------------------------
def extract_rule_flags(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context incoming from pipeline_runner:
      - claim_amount
      - policy_sum_insured
      - median_amount_for_policy (optional)
      - incident_date
      - policy_id
      - policy_id_record
    """

    reasons: List[str] = []

    # Read values
    incident_raw = context.get("incident_date")
    policy_id = context.get("policy_id")
    policy_record = context.get("policy_id_record")
    claim_amount = context.get("claim_amount")

    # Determine median reference
    median_amount = (
        context.get("median_amount_for_policy")
        or context.get("policy_sum_insured")
        or None
    )

    # ------------ Missing checks ------------
    missing_policy_record = int(policy_record is None or str(policy_record).strip() == "")
    missing_median_amount = int(median_amount is None)
    missing_incident_date = int(incident_raw is None or str(incident_raw).strip() == "")

    # ------------ Date validation ------------
    incident_dt = _parse_date(incident_raw)
    invalid_incident_date = int(incident_dt is None and incident_raw is not None)
    future_incident = int(_is_future_date(incident_dt))

    if missing_incident_date:
        reasons.append("Incident date missing or empty.")

    if invalid_incident_date:
        reasons.append(f"Incident date '{incident_raw}' could not be parsed.")

    if future_incident:
        reasons.append("Incident date is in the future, which is suspicious.")

    # ------------ Policy mismatch ------------
    mismatch = policy_mismatch(policy_id, policy_record)
    if mismatch:
        reasons.append("Policy ID mismatch between claimed and official record.")
    elif missing_policy_record:
        reasons.append("Official policy record missing or not provided.")

    # ------------ Amount-based rules ------------
    too_high_flag = claim_too_high(claim_amount, median_amount)
    soft_flag = False

    if too_high_flag:
        reasons.append(
            f"Claim amount exceeds {HARD_THRESHOLD_FACTOR}× the expected median."
        )
    else:
        soft_flag = soft_suspicion(claim_amount, median_amount)
        if soft_flag:
            reasons.append(
                f"Claim amount appears unusually high (> {SOFT_SUSPICION_FACTOR}× median)."
            )

    if missing_median_amount:
        reasons.append("Median amount for this policy unavailable; amount checks limited.")

    # ------------ Final Flags ------------
    flags = {
        "policy_expired": False,             # kept for compatibility
        "policy_mismatch": int(mismatch),
        "claim_too_high": int(too_high_flag),
        "soft_suspicion": int(soft_flag),
        "missing_policy_record": missing_policy_record,
        "missing_incident_date": missing_incident_date,
        "missing_median_amount": missing_median_amount,
        "invalid_incident_date": invalid_incident_date,
        "incident_date_in_future": future_incident,
        "reasons": reasons,
    }

    logger.debug("Rules -> %s", flags)
    return flags
