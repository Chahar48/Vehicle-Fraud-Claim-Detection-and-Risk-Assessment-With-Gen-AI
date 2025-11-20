# fraud_detection/decision_engine/scoring.py
"""
Scoring engine: aggregate ML signals + rules → final score & action.

Inputs (via context dict, optional keys):
 - anomaly_score (float)        # raw anomaly or model-provided (will be normalized)
 - fraud_prob (float 0..1)
 - similarity_score (float 0..1)
 - blacklist_flag (int/bool)
 - missing_info_flag (int/bool)
 - normalized (dict)            # normalized claim fields for rule checks
 - input (dict)                 # original claim dict (optional)

Output:
 {
   "final_score": float,
   "action": "auto_approve"|"manual_review"|"reject",
   "breakdown": { ... }
 }
"""

from __future__ import annotations
from typing import Dict, Any
import logging
import numpy as np

from fraud_detection.logging.logger import get_logger
from fraud_detection.decision_engine import rules

logger = get_logger(__name__)


# ------------------------------------------------------------
# Normalize anomaly-like value into 0..1 (higher => more anomalous)
# Accepts values that may already be 0..1 or score_samples style.
# ------------------------------------------------------------
def normalize_anomaly_score(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        v = float(value)
        # If already in [0,1], treat as normalized
        if 0.0 <= v <= 1.0:
            return float(v)
        # otherwise clamp to [-10,10] then map via sigmoid-ish transform
        v = max(min(v, 10.0), -10.0)
        norm = 1.0 / (1.0 + np.exp(-v))  # sigmoid mapping
        return float(np.clip(norm, 0.0, 1.0))
    except Exception:
        logger.exception("normalize_anomaly_score failed for value=%s", value)
        return 0.0


def compute_final_score(context: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("compute_final_score started")
    anomaly_in = context.get("anomaly_score", 0.0)
    fraud_prob = float(context.get("fraud_prob", 0.0) or 0.0)
    similarity = float(context.get("similarity_score", 0.0) or 0.0)
    blacklist_flag = int(bool(context.get("blacklist_flag", 0)))
    missing_info_flag = int(bool(context.get("missing_info_flag", 0)))

    normalized_data = context.get("normalized", {}) or {}
    input_data = context.get("input", {}) or {}

    anomaly_score = normalize_anomaly_score(anomaly_in)

    # Prepare rule context
    rule_ctx = {
        "policy_end_date": normalized_data.get("policy_end_date") or input_data.get("policy_end_date"),
        "incident_date": normalized_data.get("incident_date") or input_data.get("incident_date"),
        "policy_id": normalized_data.get("policy_id") or input_data.get("policy_id"),
        "policy_id_record": normalized_data.get("policy_id_record") or input_data.get("policy_id_record"),
        "claim_amount": normalized_data.get("claim_amount") or input_data.get("claim_amount"),
        "median_amount_for_policy": normalized_data.get("policy_sum_insured") or normalized_data.get("median_amount_for_policy") or input_data.get("policy_sum_insured")
    }

    # Extract rule flags
    try:
        rule_flags = rules.extract_rule_flags(rule_ctx)
    except Exception:
        logger.exception("Failed to extract rule flags")
        rule_flags = {}

    logger.debug("Inputs -> fraud_prob=%s anomaly_score=%s similarity=%s blacklist=%s missing_info=%s",
                 fraud_prob, anomaly_score, similarity, blacklist_flag, missing_info_flag)

    # Weights (configurable later)
    w_fraud = 0.45
    w_anom = 0.30
    w_sim = 0.15
    w_black = 0.08
    w_missing = 0.02

    final_score = (
        w_fraud * fraud_prob +
        w_anom * anomaly_score +
        w_sim * similarity +
        w_black * float(blacklist_flag) +
        w_missing * float(missing_info_flag)
    )

    logger.info("Score before overrides: %.4f", final_score)

    force_manual = False

    # Hard overrides driven by rules
    if rule_flags.get("policy_expired"):
        force_manual = True
        final_score = max(final_score, 0.45)

    if rule_flags.get("policy_mismatch"):
        force_manual = True
        final_score = max(final_score, 0.45)

    if rule_flags.get("missing_policy_record"):
        force_manual = True
        final_score = max(final_score, 0.40)

    if rule_flags.get("missing_incident_date") or rule_flags.get("missing_policy_end_date"):
        force_manual = True
        final_score = max(final_score, 0.40)

    if rule_flags.get("claim_too_high"):
        force_manual = True
        final_score = max(final_score, 0.60)

    if blacklist_flag:
        force_manual = True
        final_score = max(final_score, 0.75)

    # Anomaly escalation
    if anomaly_score >= 0.6:
        final_score = max(final_score, 0.45 + 0.2 * (anomaly_score - 0.6))
        if anomaly_score >= 0.8:
            force_manual = True

    # Missing info bump
    if missing_info_flag and final_score < 0.35:
        final_score = max(final_score, 0.30)

    # Determine action
    if force_manual:
        action = "manual_review"
    else:
        if final_score >= 0.80:
            action = "reject"
        elif final_score >= 0.45:
            action = "manual_review"
        else:
            action = "auto_approve"

    breakdown = {
        "fraud_prob": float(fraud_prob),
        "anomaly_score": float(anomaly_score),
        "similarity_score": float(similarity),
        "blacklist_flag": int(blacklist_flag),
        "missing_info_flag": int(missing_info_flag),
        "rule_flags": rule_flags,
        "weights": {
            "fraud_prob": w_fraud,
            "anomaly_score": w_anom,
            "similarity_score": w_sim,
            "blacklist_flag": w_black,
            "missing_info_flag": w_missing
        }
    }

    result = {
        "final_score": round(float(final_score), 4),
        "action": action,
        "breakdown": breakdown
    }

    logger.info("compute_final_score finished -> action=%s score=%.4f", action, result["final_score"])
    return result


# Manual test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # sample context
    ctx = {
        "anomaly_score": 0.2,
        "fraud_prob": 0.12,
        "similarity_score": 0.05,
        "blacklist_flag": 0,
        "missing_info_flag": 0,
        "normalized": {
            "policy_end_date": "2025-01-01",
            "incident_date": "2024-02-12",
            "policy_id": "POL-123",
            "policy_id_record": "POL-123",
            "claim_amount": 5000,
            "policy_sum_insured": 100000
        }
    }
    print("SCORE:", compute_final_score(ctx))
