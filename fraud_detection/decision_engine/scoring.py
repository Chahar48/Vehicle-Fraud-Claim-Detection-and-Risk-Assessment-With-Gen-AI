# fraud_detection/decision_engine/scoring.py
"""
Scoring engine: aggregate ML signals + rules -> final score & action.

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

# safe logger fallback
try:
    from fraud_detection.logging.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fraud_detection.scoring")


# import rules defensively (should exist in project)
try:
    from fraud_detection.decision_engine import rules
except Exception:
    rules = None
    logger.warning("decision_engine.rules not importable; rule flags will be empty.")


# ------------------------------------------------------------
# Normalize anomaly-like value into 0..1 (higher => more anomalous)
# Accepts values that may already be 0..1 or raw anomaly scores.
# ------------------------------------------------------------
def normalize_anomaly_score(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        v = float(value)
        # If already in [0,1], treat as normalized
        if 0.0 <= v <= 1.0:
            return float(v)
        # otherwise clamp to [-10,10] then map via tanh -> 0..1
        v = max(min(v, 10.0), -10.0)
        scaled = np.tanh(v)  # (-1,1)
        norm = (scaled + 1.0) / 2.0  # (0,1)
        return float(np.clip(norm, 0.0, 1.0))
    except Exception:
        logger.exception("normalize_anomaly_score failed for value=%s", value)
        return 0.0


def _safe_get_number(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def compute_final_score(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute final fraud/risk score and action from provided context.
    Always returns a dict with keys: final_score, action, breakdown.
    """
    try:
        logger.info("compute_final_score started")

        # Defensive extraction
        anomaly_in = context.get("anomaly_score", 0.0)
        fraud_prob = _safe_get_number(context.get("fraud_prob", 0.0))
        similarity = _safe_get_number(context.get("similarity_score", 0.0))
        blacklist_flag = int(bool(context.get("blacklist_flag", 0)))
        missing_info_flag = int(bool(context.get("missing_info_flag", 0)))

        normalized_data = context.get("normalized", {}) or {}
        input_data = context.get("input", {}) or {}

        # Normalize anomaly into 0..1 (higher => more anomalous)
        anomaly_score = normalize_anomaly_score(anomaly_in)

        # Prepare rule context using the allowed fields only
        rule_ctx = {
            "incident_date": normalized_data.get("incident_date") or input_data.get("incident_date"),
            "policy_id": normalized_data.get("policy_id") or input_data.get("policy_id"),
            "policy_id_record": normalized_data.get("policy_id_record") or input_data.get("policy_id_record"),
            "claim_amount": normalized_data.get("claim_amount") or input_data.get("claim_amount"),
            "median_amount_for_policy": (
                normalized_data.get("policy_sum_insured")
                or normalized_data.get("median_amount_for_policy")
                or input_data.get("policy_sum_insured")
            ),
        }

        # Extract rule flags defensively
        try:
            rule_flags = rules.extract_rule_flags(rule_ctx) if rules is not None else {}
        except Exception:
            logger.exception("Failed to extract rule flags")
            rule_flags = {}

        logger.debug(
            "Inputs -> fraud_prob=%s anomaly_score=%s similarity=%s blacklist=%s missing_info=%s",
            fraud_prob, anomaly_score, similarity, blacklist_flag, missing_info_flag,
        )

        # Weights (configurable later)
        w_fraud = 0.45
        w_anom = 0.30
        w_sim = 0.15
        w_black = 0.08
        w_missing = 0.02

        # Clamp inputs
        fraud_prob = float(np.clip(fraud_prob, 0.0, 1.0))
        similarity = float(np.clip(similarity, 0.0, 1.0))
        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        # Aggregate
        final_score = (
            w_fraud * fraud_prob
            + w_anom * anomaly_score
            + w_sim * similarity
            + w_black * float(bool(blacklist_flag))
            + w_missing * float(bool(missing_info_flag))
        )

        logger.info("Score before overrides: %.4f", final_score)

        # Conservative overrides
        force_manual = False

        # Rule-driven overrides (use keys that rules provides; absent keys default False/0)
        if rule_flags.get("policy_expired"):
            force_manual = True
            final_score = max(final_score, 0.45)
            logger.info("policy_expired -> forcing manual review / bumping score")

        if rule_flags.get("policy_mismatch"):
            force_manual = True
            final_score = max(final_score, 0.45)
            logger.info("policy_mismatch -> forcing manual review / bumping score")

        if rule_flags.get("missing_policy_record"):
            force_manual = True
            final_score = max(final_score, 0.40)
            logger.info("missing_policy_record -> conservative manual review")

        if rule_flags.get("missing_incident_date"):
            force_manual = True
            final_score = max(final_score, 0.40)
            logger.info("missing_incident_date -> conservative manual review")

        if rule_flags.get("claim_too_high"):
            force_manual = True
            final_score = max(final_score, 0.60)
            logger.info("claim_too_high -> forcing manual review (hard threshold)")

        if bool(blacklist_flag):
            force_manual = True
            final_score = max(final_score, 0.75)
            logger.info("blacklist_flag -> forcing manual review (high risk)")

        # Anomaly escalation
        if anomaly_score >= 0.6:
            bump = 0.45 + 0.2 * (anomaly_score - 0.6)
            final_score = max(final_score, bump)
            logger.info("High anomaly (%.3f) -> bumped final_score to %.4f", anomaly_score, final_score)
            if anomaly_score >= 0.8:
                force_manual = True
                logger.info("Very high anomaly -> force manual review")

        # Small bump for missing info if needed
        if missing_info_flag and final_score < 0.35:
            final_score = max(final_score, 0.30)
            logger.debug("Missing info bump -> final_score now %.4f", final_score)

        # Final action
        if force_manual:
            action = "manual_review"
        else:
            if final_score >= 0.80:
                action = "reject"
            elif final_score >= 0.45:
                action = "manual_review"
            else:
                action = "auto_approve"

        # Breakdown for explainability
        breakdown = {
            "fraud_prob": float(fraud_prob),
            "anomaly_score": float(anomaly_score),
            "similarity_score": float(similarity),
            "blacklist_flag": int(bool(blacklist_flag)),
            "missing_info_flag": int(bool(missing_info_flag)),
            "rule_flags": rule_flags,
            "weights": {
                "fraud_prob": w_fraud,
                "anomaly_score": w_anom,
                "similarity_score": w_sim,
                "blacklist_flag": w_black,
                "missing_info_flag": w_missing,
            },
        }

        result = {
            "final_score": round(float(final_score), 4) if final_score is not None else None,
            "action": action,
            "breakdown": breakdown,
        }

        logger.info("compute_final_score finished -> action=%s score=%s", action, result["final_score"])
        return result

    except Exception as e:
        # If anything unexpected fails, return conservative manual review decision
        logger.exception("Unexpected scoring error: %s", e)
        fallback = {
            "final_score": None,
            "action": "manual_review",
            "breakdown": {
                "error": str(e),
                "rule_flags": {},
            },
        }
        return fallback


# Manual test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctx = {
        "anomaly_score": 0.2,
        "fraud_prob": 0.12,
        "similarity_score": 0.05,
        "blacklist_flag": 0,
        "missing_info_flag": 0,
        "normalized": {
            "incident_date": "2024-02-12",
            "policy_id": "POL-123",
            "policy_id_record": "POL-123",
            "claim_amount": 5000,
            "policy_sum_insured": 100000,
        },
    }
    print("SCORE:", compute_final_score(ctx))
