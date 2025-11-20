# fraud_detection/preprocessing/normalize_fields.py

"""
Field Normalization Module
--------------------------
Updated for CURRENT project architecture (PHASE 1–17).

This file cleans ONLY the allowed claim fields:

    claim_id
    customer_id
    policy_id
    policy_id_record
    claim_amount
    policy_sum_insured
    incident_date
    description
    phone
    garage_id

No VIN, no city, no policy_end_date, no reported_date.

Output ALWAYS contains:
- cleaned numeric fields
- cleaned date (YYYY-MM-DD)
- cleaned phone
- cleaned IDs
- missing_info_flag
- missing_amount_flag
"""

import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional
from dateutil import parser as date_parser

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# Resolve project root safely
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -------------------------------------------------------
# Currency / Numeric
# -------------------------------------------------------
def normalize_currency(val: Optional[Any]) -> Optional[float]:
    if val is None:
        return None
    try:
        # Already numeric
        if isinstance(val, (int, float, Decimal)):
            v = float(val)
            if v != v or v in (float("inf"), float("-inf")):
                return None
            return v

        s = str(val).strip()
        if s == "":
            return None

        # remove currency marks
        s = re.sub(r"(?i)(rs\.?|inr|usd|₹|\$|,)", "", s)

        # extract number
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", s)
        if not m:
            return None

        num = m.group().replace(",", "")
        return float(Decimal(num))

    except Exception as e:
        logger.debug(f"normalize_currency failed for {val}: {e}")
        return None


# -------------------------------------------------------
# Date normalization
# -------------------------------------------------------
def normalize_date(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    try:
        if hasattr(val, "strftime"):
            return val.strftime("%Y-%m-%d")
        s = str(val).strip()
        if s == "":
            return None
        dt = date_parser.parse(s, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


# -------------------------------------------------------
# Phone
# -------------------------------------------------------
def normalize_phone(val: Optional[Any], min_digits: int = 6) -> Optional[str]:
    if val is None:
        return None
    digits = "".join(ch for ch in str(val) if ch.isdigit())
    if len(digits) < min_digits:
        return None
    return digits


# -------------------------------------------------------
# IDs
# -------------------------------------------------------
def normalize_policy_id(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().upper()
    if not s:
        return None
    return re.sub(r"[^A-Z0-9\-_]", "", s) or None


def normalize_simple_id(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    return re.sub(r"\s+", "_", s)


# -------------------------------------------------------
# MAIN NORMALIZATION (FINAL)
# -------------------------------------------------------
def normalize_claims_df(df):
    """
    Accepts a DataFrame with a single row.
    Returns a DataFrame with exactly the normalized fields used in pipeline_runner.
    """

    if df is None or df.empty:
        raise ValueError("normalize_claims_df: input DataFrame is empty")

    row = df.iloc[0].to_dict()
    out = {}

    # Required fields — normalize safely
    out["claim_id"] = row.get("claim_id")

    out["customer_id"] = normalize_simple_id(row.get("customer_id"))
    out["policy_id"] = normalize_policy_id(row.get("policy_id"))
    out["policy_id_record"] = normalize_policy_id(row.get("policy_id_record"))

    out["claim_amount"] = normalize_currency(row.get("claim_amount"))
    out["policy_sum_insured"] = normalize_currency(row.get("policy_sum_insured"))

    out["incident_date"] = normalize_date(row.get("incident_date"))

    out["description"] = (row.get("description") or "").strip()

    out["phone"] = normalize_phone(row.get("phone"))
    out["garage_id"] = normalize_simple_id(row.get("garage_id"))

    # --- Additional pipeline-required flags ---
    out["missing_amount_flag"] = int(out["claim_amount"] is None)
    out["missing_info_flag"] = int(
        not bool(out["phone"])
        or not bool(out["garage_id"])
    )

    logger.debug(f"Normalized claim row => {out}")

    return (
        # return as DataFrame (pipeline_runner expects this)
        df.assign(**out)[list(out.keys())]
    )


# -------------------------------------------------------
# TEST BLOCK
# -------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    sample = pd.DataFrame([{
        "claim_id": "C1001",
        "customer_id": "cust 77",
        "policy_id": "pol-123",
        "policy_id_record": "POL123 ",
        "claim_amount": "Rs. 5,000.25",
        "policy_sum_insured": "100000",
        "incident_date": "02/10/2024",
        "description": "Minor scratch",
        "phone": "+91 98765-43210",
        "garage_id": "GAR 1 ",
    }])

    print(normalize_claims_df(sample))
