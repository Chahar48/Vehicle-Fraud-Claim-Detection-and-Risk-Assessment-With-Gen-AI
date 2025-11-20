# fraud_detection/decision_engine/explainability.py
"""
Explainability utilities for decision engine.

Converts scoring output (final_score, action, breakdown) into:
 - a short summary string
 - a detailed explanation (multi-line)
 - structured reasons (list) for UI/HITL

No LLM required (PoC): templates + rule-based text.
"""

from __future__ import annotations
from typing import Dict, Any, List
from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)


def build_explanation(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input: result from scoring.compute_final_score
    Output:
      {
        "summary": "Auto-approve / Manual review / Reject explanation",
        "explanation": "Multi-line detailed explanation",
        "reasons": [ ... ]  # list of human-readable reasons
      }
    """
    action = result.get("action", "manual_review")
    final_score = result.get("final_score", 0.0)
    breakdown = result.get("breakdown", {})
    rf = breakdown.get("rule_flags", {}) or {}

    fraud_prob = breakdown.get("fraud_prob", 0.0)
    anomaly = breakdown.get("anomaly_score", 0.0)
    similarity = breakdown.get("similarity_score", 0.0)
    blacklist_flag = int(breakdown.get("blacklist_flag", 0))
    missing_info_flag = int(breakdown.get("missing_info_flag", 0))

    reasons: List[str] = []

    # Basic summary line
    if action == "auto_approve":
        summary = f"Auto-approve (final score {final_score:.2f})"
    elif action == "manual_review":
        summary = f"Requires manual review (final score {final_score:.2f})"
    else:
        summary = f"High-risk — recommended reject (final score {final_score:.2f})"

    # Compose explanation parts
    # Model probability
    if fraud_prob >= 0.7:
        reasons.append(f"Supervised model indicates high fraud probability ({fraud_prob:.2f}).")
    elif fraud_prob >= 0.4:
        reasons.append(f"Supervised model indicates moderate fraud probability ({fraud_prob:.2f}).")
    else:
        reasons.append(f"Supervised model indicates low fraud probability ({fraud_prob:.2f}).")

    # Anomaly
    if anomaly >= 0.7:
        reasons.append(f"Numeric features are highly anomalous (anomaly score {anomaly:.2f}).")
    elif anomaly >= 0.4:
        reasons.append(f"Numeric features show moderate anomaly (anomaly score {anomaly:.2f}).")
    else:
        reasons.append(f"Numeric features are within normal range (anomaly score {anomaly:.2f}).")

    # Similarity
    if similarity >= 0.7:
        reasons.append(f"Description strongly matches previously confirmed fraud cases (similarity {similarity:.2f}).")
    elif similarity >= 0.4:
        reasons.append(f"Description shows some resemblance to known frauds (similarity {similarity:.2f}).")
    else:
        reasons.append(f"Description similarity to past fraud is low (similarity {similarity:.2f}).")

    # Blacklist & missing info
    if blacklist_flag:
        reasons.append("Customer or related entity found in blacklist.")
    else:
        reasons.append("No blacklist matches found.")

    if missing_info_flag:
        reasons.append("Some required information is missing; this increases uncertainty.")
    else:
        reasons.append("Required information is present.")

    # Add rule-based reasons (if any)
    rule_reasons = rf.get("reasons", []) if isinstance(rf, dict) else []
    for r in rule_reasons:
        reasons.append(f"Rule: {r}")

    # Build final multi-line explanation
    explanation_text = "\n".join(f"- {r}" for r in reasons)

    return {
        "summary": summary,
        "explanation": explanation_text,
        "reasons": reasons
    }


# Manual test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = {
        "final_score": 0.82,
        "action": "reject",
        "breakdown": {
            "fraud_prob": 0.75,
            "anomaly_score": 0.60,
            "similarity_score": 0.20,
            "blacklist_flag": 1,
            "missing_info_flag": 0,
            "rule_flags": {
                "reasons": ["Claim amount too high", "Policy expired"]
            }
        }
    }
    out = build_explanation(sample)
    print("SUMMARY:", out["summary"])
    print("EXPLANATION:\n", out["explanation"])
