# fraud_detection/preprocessing/field_normalizer.py

import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional
from dateutil import parser as date_parser

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# Project root (env-first)
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -------------------------
# Numeric normalization
# -------------------------
def normalize_currency(val: Optional[Any]) -> Optional[float]:
    """
    Convert messy currency strings or numbers to float.
    Returns None on failure.
    """
    if val is None:
        return None

    try:
        if isinstance(val, (int, float, Decimal)):
            # guard against NaN/Inf
            v = float(val)
            if v != v or v == float("inf") or v == float("-inf"):
                return None
            return float(v)
        s = str(val).strip()
        if s == "":
            return None
        # Remove common currency labels
        s = re.sub(r"(?i)(rs\.?|inr|usd|₹|\$|,)", "", s)
        # Extract first numeric pattern
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", s)
        if not m:
            return None
        num = m.group().replace(",", "")
        # use Decimal for safety
        try:
            d = Decimal(num)
            f = float(d)
            return f
        except InvalidOperation:
            return None
    except Exception as e:
        logger.debug("normalize_currency failed for %s: %s", val, e)
        return None


# -------------------------
# Date normalization
# -------------------------
def normalize_date(date_val: Optional[Any]) -> Optional[str]:
    """
    Convert many date inputs to ISO date string YYYY-MM-DD.
    Returns None if parsing fails.
    """
    if date_val is None:
        return None
    try:
        # if already a date/datetime-like object
        if hasattr(date_val, "strftime"):
            return date_val.strftime("%Y-%m-%d")
        s = str(date_val).strip()
        if s == "":
            return None
        # Some OCRs give day/month ambiguous; default dayfirst False (US style).
        dt = date_parser.parse(s, dayfirst=False, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        logger.debug("normalize_date failed for %s: %s", date_val, e)
        return None


# -------------------------
# Phone normalization
# -------------------------
def normalize_phone(phone_val: Optional[Any], min_digits: int = 6) -> Optional[str]:
    """
    Return digits-only phone if it meets min_digits threshold.
    """
    if phone_val is None:
        return None
    s = str(phone_val)
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < min_digits:
        return None
    # If international, keep as-is digits (no +). Business logic can add +91 if needed.
    return digits


# -------------------------
# ID / policy normalization
# -------------------------
def normalize_policy_id(policy_val: Optional[Any]) -> Optional[str]:
    if policy_val is None:
        return None
    s = str(policy_val).strip().upper()
    if s == "":
        return None
    # Keep alphanum and hyphens/underscores only
    s = re.sub(r"[^A-Z0-9\-_]", "", s)
    return s or None


def normalize_simple_id(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    return re.sub(r"\s+", "_", s)


# -------------------------
# Main normalization for a claim dict
# -------------------------
def normalize_claim_dict(claim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a claim dict (possibly from LLM), normalize important fields and
    return a new dict. This is conservative: it does not fill missing values.
    """
    out = dict(claim)  # shallow copy

    # numeric fields
    for fld in ["claim_amount", "policy_sum_insured"]:
        if fld in out:
            out[fld] = normalize_currency(out.get(fld))

    # dates
    for fld in ["incident_date", "reported_date", "claim_date"]:
        if fld in out:
            out[fld] = normalize_date(out.get(fld))

    # phones
    if "phone" in out:
        out["phone"] = normalize_phone(out.get("phone"))

    # policy ids
    for fld in ["policy_id", "policy_id_record"]:
        if fld in out:
            out[fld] = normalize_policy_id(out.get(fld))

    # simple ids
    for fld in ["garage_id", "customer_id", "vin"]:
        if fld in out:
            out[fld] = normalize_simple_id(out.get(fld))

    logger.debug("normalize_claim_dict => %s", {k: v for k, v in out.items() if k in ['claim_id','claim_amount','incident_date','policy_id','phone']})
    return out


# -------------------------
# module test
# -------------------------
if __name__ == "__main__":
    sample = {
        "claim_id": "C1001",
        "claim_amount": "Rs. 5,000.00",
        "policy_sum_insured": "$10,000",
        "incident_date": "2/12/2024",
        "phone": "+91-98765 43210",
        "policy_id": " pol-123 ",
    }
    print(normalize_claim_dict(sample))
