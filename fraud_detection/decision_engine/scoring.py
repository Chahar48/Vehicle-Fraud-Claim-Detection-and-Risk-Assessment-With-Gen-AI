# fraud_detection/decision_engine/scoring.py
"""
Final Scoring Engine for Fraud Detection & Risk Assessment

This module aggregates:
 - ML model signals (fraud_prob, anomaly_score, similarity_score)
 - Business rules (from rules.extract_rule_flags)
 - Enrichment signals (blacklist_flag)
 - Data completeness (missing_info_flag)

Outputs:
 {
   "final_score": float (0–1),
   "action": "auto_approve" | "manual_review" | "reject",
   "breakdown": {...}
 }
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import logging

try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("scoring")


# Safe import of rule engine
try:
    from fraud_detection.decision_engine import rules
except Exception:
    rules = None
    logger.warning("decision_engine.rules could not be imported; rule flags disabled.")


# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _normalize_0_1(value: Any) -> float:
    """
    Normalize any anomaly-like value into 0–1.
    Accepts:
       - raw scores outside 0–1
       - negative anomaly scores
       - None
    """
    try:
        if value is None:
            return 0.0
        v = float(value)
        if 0.0 <= v <= 1.0:
            return v
        # Avoid extreme ranges
        v = max(min(v, 10), -10)
        scaled = np.tanh(v)          # (-1,1)
        return float((scaled + 1) / 2)
    except Exception:
        return 0.0


# -------------------------------------------------------------------
# Final Scoring Logic
# -------------------------------------------------------------------
def compute_final_score(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dict:
     {
       "final_score": float,
       "action": string,
       "breakdown": dict
     }
    """

    try:
        logger.info("Starting scoring engine")

        # -------------------------------------------------------------------
        # Extract raw ML scores
        # -------------------------------------------------------------------
        fraud_prob = _safe_float(context.get("fraud_prob", 0))
        anomaly_score = _normalize_0_1(context.get("anomaly_score", 0))
        similarity_score = _safe_float(context.get("similarity_score", 0))
        blacklist_flag = int(bool(context.get("blacklist_flag", 0)))
        missing_info_flag = int(bool(context.get("missing_info_flag", 0)))

        fraud_prob = float(np.clip(fraud_prob, 0.0, 1.0))
        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))
        similarity_score = float(np.clip(similarity_score, 0.0, 1.0))

        normalized = context.get("normalized", {}) or {}
        source = context.get("input", {}) or {}

        # -------------------------------------------------------------------
        # Build rule context
        # -------------------------------------------------------------------
        rule_ctx = {
            "incident_date": normalized.get("incident_date") or source.get("incident_date"),
            "policy_id": normalized.get("policy_id") or source.get("policy_id"),
            "policy_id_record": normalized.get("policy_id_record") or source.get("policy_id_record"),
            "claim_amount": normalized.get("claim_amount") or source.get("claim_amount"),
            "policy_sum_insured": normalized.get("policy_sum_insured") or source.get("policy_sum_insured"),
            "median_amount_for_policy": (
                normalized.get("policy_sum_insured")
                or normalized.get("median_amount_for_policy")
                or source.get("policy_sum_insured")
            ),
        }

        if rules:
            try:
                rule_flags = rules.extract_rule_flags(rule_ctx)
            except Exception:
                logger.exception("Rule extraction failed")
                rule_flags = {}
        else:
            rule_flags = {}

        # -------------------------------------------------------------------
        # Weight configuration (tunable)
        # -------------------------------------------------------------------
        w = {
            "fraud_prob": 0.45,
            "anomaly_score": 0.30,
            "similarity_score": 0.15,
            "blacklist": 0.08,
            "missing_info": 0.02
        }

        # -------------------------------------------------------------------
        # Base score (no overrides)
        # -------------------------------------------------------------------
        base_score = (
            w["fraud_prob"] * fraud_prob +
            w["anomaly_score"] * anomaly_score +
            w["similarity_score"] * similarity_score +
            w["blacklist"] * blacklist_flag +
            w["missing_info"] * missing_info_flag
        )

        final_score = float(np.clip(base_score, 0.0, 1.0))
        force_manual = False
        force_reject = False

        logger.info(f"Base score = {final_score:.4f}")

        # -------------------------------------------------------------------
        # Rule-Based Overrides (new)
        # -------------------------------------------------------------------

        # 🔹 Hard triggers → always manual
        if rule_flags.get("policy_mismatch"):
            force_manual = True
            final_score = max(final_score, 0.45)

        if rule_flags.get("policy_expired"):
            force_manual = True
            final_score = max(final_score, 0.50)

        if rule_flags.get("claim_too_high"):
            force_manual = True
            final_score = max(final_score, 0.60)

        if blacklist_flag:
            force_manual = True
            final_score = max(final_score, 0.75)

        # 🔹 Date issues
        if rule_flags.get("invalid_incident_date"):
            force_manual = True
            final_score = max(final_score, 0.45)

        if rule_flags.get("incident_date_in_future"):
            force_manual = True
            final_score = max(final_score, 0.60)

        # 🔹 Missing policy record
        if rule_flags.get("missing_policy_record"):
            force_manual = True
            final_score = max(final_score, 0.40)

        if rule_flags.get("missing_incident_date"):
            force_manual = True
            final_score = max(final_score, 0.35)

        # 🔹 Missing median
        if rule_flags.get("missing_median_amount"):
            final_score = max(final_score, 0.30)

        # 🔹 Soft suspicion (3× median)
        if rule_flags.get("soft_suspicion"):
            # Does NOT force manual, only bumps
            final_score = max(final_score, 0.35)

        # 🔹 High anomaly escalation
        if anomaly_score >= 0.6:
            final_score = max(final_score, 0.45 + 0.2 * (anomaly_score - 0.6))
            if anomaly_score >= 0.8:
                force_manual = True

        # -------------------------------------------------------------------
        # Final Decision
        # -------------------------------------------------------------------
        if force_reject:
            action = "reject"
        elif force_manual:
            action = "manual_review"
        else:
            if final_score >= 0.80:
                action = "reject"
            elif final_score >= 0.45:
                action = "manual_review"
            else:
                action = "auto_approve"

        # -------------------------------------------------------------------
        # Build breakdown
        # -------------------------------------------------------------------
        breakdown = {
            "fraud_prob": fraud_prob,
            "anomaly_score": anomaly_score,
            "similarity_score": similarity_score,
            "blacklist_flag": blacklist_flag,
            "missing_info_flag": missing_info_flag,
            "rule_flags": rule_flags,
            "weights": w,
        }

        return {
            "final_score": round(final_score, 4),
            "action": action,
            "breakdown": breakdown
        }

    except Exception as e:
        logger.exception("Fatal scoring error: %s", e)
        return {
            "final_score": None,
            "action": "manual_review",
            "breakdown": {"error": str(e)}
        }

