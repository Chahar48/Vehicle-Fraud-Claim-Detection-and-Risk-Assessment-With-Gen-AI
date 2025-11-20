"""
explanation_generator.py
------------------------
GenAI Layer — Human-friendly Explanation Generator

This module converts scoring output into a structured, easy-to-read
explanation for claims reviewers.

PoC version uses deterministic rule-based templates.
(No LLM needed to avoid costs & latency.)

Input to this module:
{
    "final_score": float,
    "action": "auto_approve" | "manual_review" | "fraud_reject",
    "breakdown": {
        "fraud_prob": float,
        "anomaly_score": float,
        "similarity_score": float,
        "blacklist_flag": 0/1,
        "missing_info_flag": 0/1,
        "rule_override": None / "manual_review"
    }
}

Output:
    str — a readable explanation paragraph
"""

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------
# Explanation generator
# -------------------------------------------------------------
def generate_text_explanation(result: dict) -> str:
    """
    Convert scoring engine result → readable explanation text.
    """

    final_score = result.get("final_score", 0)
    action = result.get("action", "manual_review")
    b = result.get("breakdown", {})

    fraud_prob = b.get("fraud_prob", 0)
    anomaly = b.get("anomaly_score", 0)
    similarity = b.get("similarity_score", 0)
    blacklist_flag = b.get("blacklist_flag", 0)
    missing_flag = b.get("missing_info_flag", 0)
    rule_override = b.get("rule_override", None)

    explanation = []

    # ---------------------------------------------------------
    # 1. Summary line based on decision
    # ---------------------------------------------------------
    if action == "auto_approve":
        explanation.append(
            f"The claim appears to be low risk (final score {final_score:.2f})."
        )
    elif action == "manual_review":
        explanation.append(
            f"The claim requires manual review (final score {final_score:.2f})."
        )
    else:
        explanation.append(
            f"The claim is flagged as high risk (final score {final_score:.2f})."
        )

    # ---------------------------------------------------------
    # 2. Fraud model explanation
    # ---------------------------------------------------------
    if fraud_prob >= 0.7:
        explanation.append(
            f"- Supervised fraud model shows a high probability of fraud ({fraud_prob:.2f})."
        )
    elif fraud_prob >= 0.4:
        explanation.append(
            f"- Fraud model indicates moderate suspicion ({fraud_prob:.2f})."
        )
    else:
        explanation.append(
            f"- Fraud model indicates low risk ({fraud_prob:.2f})."
        )

    # ---------------------------------------------------------
    # 3. Anomaly score explanation
    # ---------------------------------------------------------
    if anomaly >= 0.7:
        explanation.append(
            f"- Numeric patterns in the claim are unusual (anomaly score {anomaly:.2f})."
        )
    elif anomaly >= 0.4:
        explanation.append(
            f"- Some numeric irregularities detected (anomaly score {anomaly:.2f})."
        )
    else:
        explanation.append(
            f"- Numeric patterns appear normal (anomaly score {anomaly:.2f})."
        )

    # ---------------------------------------------------------
    # 4. Text similarity explanation
    # ---------------------------------------------------------
    if similarity >= 0.7:
        explanation.append(
            f"- Description is highly similar to known fraudulent cases ({similarity:.2f})."
        )
    elif similarity >= 0.4:
        explanation.append(
            f"- Description shows some similarity to known fraud patterns ({similarity:.2f})."
        )
    else:
        explanation.append(
            f"- Description does not resemble known fraud patterns ({similarity:.2f})."
        )

    # ---------------------------------------------------------
    # 5. Blacklist signal
    # ---------------------------------------------------------
    if blacklist_flag == 1:
        explanation.append("- One or more blacklist indicators were triggered.")

    else:
        explanation.append("- No blacklist risks detected.")

    # ---------------------------------------------------------
    # 6. Missing info
    # ---------------------------------------------------------
    if missing_flag == 1:
        explanation.append("- Some important fields were missing.")
    else:
        explanation.append("- All required claim information was available.")

    # ---------------------------------------------------------
    # 7. Rule override
    # ---------------------------------------------------------
    if rule_override == "manual_review":
        explanation.append("- Business rules forced manual review.")

    return "\n".join(explanation)

# ------------------------------------------------------------
# Manual Test Block
# ------------------------------------------------------------
if __name__ == "__main__":
    sample_output = {
        "final_score": 0.82,
        "action": "fraud_reject",
        "breakdown": {
            "fraud_prob": 0.75,
            "anomaly_score": 0.60,
            "similarity_score": 0.20,
            "blacklist_flag": 1,
            "missing_info_flag": 0,
            "rule_override": None
        }
    }

    explanation = generate_text_explanation(sample_output)
    print("\n=== Human-Friendly Explanation ===\n")
    print(explanation)
