# fraud_detection/decision_engine/explainability.py
"""
Explainability engine for fraud-detection-genai.

Converts scoring output into:
 - summary (1-line result)
 - explanation (multi-line breakdown)
 - reasons (list of short bullets for UI/HITL)

This file integrates:
 - model signals (fraud_prob, anomaly_score, similarity_score)
 - rule-based flags (from rules.extract_rule_flags)
 - metadata (missing info, blacklist hits)
"""

from __future__ import annotations
from typing import Dict, Any, List
from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Human-readable descriptions for rule flags
# -------------------------------------------------------------------
RULE_FLAG_EXPLANATIONS = {
    "policy_expired": "The policy appears expired or outside coverage period.",
    "policy_mismatch": "The claimed policy ID does not match the official record.",
    "claim_too_high": "Claim amount exceeds the allowed threshold relative to policy median.",
    "soft_suspicion": "Claim amount is unusually high (>3× typical median for this policy).",
    "missing_policy_record": "Official policy record is missing or incomplete.",
    "missing_incident_date": "Incident date is missing or invalid.",
    "invalid_incident_date": "Incident date format could not be validated.",
    "incident_date_in_future": "Incident date appears to be in the future.",
    "missing_median_amount": "Median amount for the policy is unavailable; amount comparison limited.",
}


# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    """Format as percent: 0.53 -> '53%'."""
    try:
        return f"{x * 100:.0f}%"
    except Exception:
        return "N/A"


# -------------------------------------------------------------------
# Main explainability builder
# -------------------------------------------------------------------
def build_explanation(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build human-readable explainability from scoring output.

    Input:
      result = {
         "final_score": float,
         "action": "auto_approve"|"manual_review"|"reject",
         "breakdown": { ... }
      }

    Returns:
      {
         "summary": str,
         "explanation": str,
         "reasons": [str, ...]
      }
    """

    action = result.get("action", "manual_review")
    final_score = result.get("final_score", 0.0)
    breakdown = result.get("breakdown", {}) or {}
    rule_flags = breakdown.get("rule_flags", {}) or {}

    fraud_prob = breakdown.get("fraud_prob", 0.0)
    anomaly = breakdown.get("anomaly_score", 0.0)
    similarity = breakdown.get("similarity_score", 0.0)
    blacklist_flag = int(breakdown.get("blacklist_flag", 0))
    missing_info_flag = int(breakdown.get("missing_info_flag", 0))

    # -------------------------------------------------------------------
    # Summary line
    # -------------------------------------------------------------------
    if action == "auto_approve":
        summary = f"Auto-approved (final score {final_score:.2f})."
    elif action == "reject":
        summary = f"High-risk claim — recommended rejecting (final score {final_score:.2f})."
    else:
        summary = f"Requires manual review (final score {final_score:.2f})."

    # -------------------------------------------------------------------
    # Build REASONS list
    # -------------------------------------------------------------------
    reasons: List[str] = []

    # Supervised model explanation
    if fraud_prob >= 0.7:
        reasons.append(f"Supervised model indicates HIGH risk ({_fmt_pct(fraud_prob)} fraud probability).")
    elif fraud_prob >= 0.4:
        reasons.append(f"Supervised model indicates MODERATE risk ({_fmt_pct(fraud_prob)} fraud probability).")
    else:
        reasons.append(f"Supervised model indicates LOW risk ({_fmt_pct(fraud_prob)} fraud probability).")

    # Anomaly score
    if anomaly >= 0.7:
        reasons.append(f"Numeric features are highly unusual (anomaly score {anomaly:.2f}).")
    elif anomaly >= 0.4:
        reasons.append(f"Numeric features show moderate deviation (anomaly score {anomaly:.2f}).")
    else:
        reasons.append(f"Numeric features appear normal (anomaly score {anomaly:.2f}).")

    # Similarity check
    if similarity >= 0.7:
        reasons.append(f"Claim description highly resembles historical fraud patterns (similarity {similarity:.2f}).")
    elif similarity >= 0.4:
        reasons.append(f"Claim description shows some resemblance to known fraud patterns (similarity {similarity:.2f}).")
    else:
        reasons.append(f"Description similarity to known fraud cases is low (similarity {similarity:.2f}).")

    # Blacklist
    if blacklist_flag:
        reasons.append("Entity found in blacklist (customer/vehicle/garage/etc).")
    else:
        reasons.append("No blacklist matches detected.")

    # Missing information
    if missing_info_flag:
        reasons.append("Some required information is missing (phone, garage, amount, etc).")
    else:
        reasons.append("All required information is present.")

    # -------------------------------------------------------------------
    # Add ALL rule-based flags
    # -------------------------------------------------------------------
    for key, val in rule_flags.items():
        if key == "reasons":
            continue  # textual reasons already handled separately

        if key in RULE_FLAG_EXPLANATIONS and bool(val):
            reasons.append(f"Rule triggered: {RULE_FLAG_EXPLANATIONS[key]}")

    # Append rule engine textual reasons (if any)
    rule_text_reasons = rule_flags.get("reasons", [])
    if isinstance(rule_text_reasons, list):
        for r in rule_text_reasons:
            reasons.append(f"Rule: {r}")

    # -------------------------------------------------------------------
    # Build multi-line explanation text
    # -------------------------------------------------------------------
    explanation_text = "\n".join(f"- {r}" for r in reasons)

    return {
        "summary": summary,
        "explanation": explanation_text,
        "reasons": reasons,
    }


# -------------------------------------------------------------------
# Manual test
# -------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    sample = {
        "final_score": 0.78,
        "action": "manual_review",
        "breakdown": {
            "fraud_prob": 0.62,
            "anomaly_score": 0.32,
            "similarity_score": 0.58,
            "blacklist_flag": 0,
            "missing_info_flag": 1,
            "rule_flags": {
                "policy_mismatch": 1,
                "missing_incident_date": 0,
                "missing_policy_record": 1,
                "reasons": ["Official policy record missing", "Mismatch in policy ID"],
            },
        },
    }

    out = build_explanation(sample)
    print("\nSUMMARY:", out["summary"])
    print("\nEXPLANATION:\n", out["explanation"])
