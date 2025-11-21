"""
Improved Field Normalizer for fraud-detection-genai
---------------------------------------------------

Fixes:
 - Adds fuzzy + alias-aware fallback extraction (aligned with csv_ingest & api_ingest)
 - Adds _field_confidence dict for later use by rules/scoring
 - Handles messy keys and values more safely
 - Prevents accidental claim_amount=0 when empty string supplied
 - Ensures consistent normalization pipeline-wide

Produces:
  {
      ...canonical fields...,
      "missing_amount_flag": 0/1,
      "missing_info_flag": 0/1,
      "_field_confidence": {field: confidence_score}
  }
"""

from __future__ import annotations
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dateutil import parser as date_parser

try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("field_normalizer")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


# -----------------------------
# Project root
# -----------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------
# Regex patterns for fallback extraction
# -----------------------------
_amount_re = re.compile(r"[-+]?\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?")
_phone_re = re.compile(r"\+?\d[\d\s\-]{6,}\d")
_date_re = re.compile(r"\b(?:\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b")
_policy_re = re.compile(r"[A-Z0-9\-_]{4,}")


# -----------------------------
# Normalization helpers
# -----------------------------
def normalize_currency(val: Optional[Any]) -> Optional[float]:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float, Decimal)):
            v = float(val)
            if v != v:
                return None
            return v

        s = str(val).strip()
        if s == "":
            return None

        s = re.sub(r"(?i)(rs\.?|inr|₹|\$|usd)", "", s)
        s = s.replace(",", "").strip()

        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        if not m:
            return None

        return float(Decimal(m.group()))
    except Exception:
        return None


def normalize_date(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    try:
        if hasattr(val, "strftime"):
            return val.strftime("%Y-%m-%d")
        s = str(val).strip()
        if not s:
            return None
        dt = date_parser.parse(s, fuzzy=True, dayfirst=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_phone(val: Optional[Any], min_digits: int = 6) -> Optional[str]:
    if val is None:
        return None
    digits = "".join([c for c in str(val) if c.isdigit()])
    return digits if len(digits) >= min_digits else None


def normalize_policy_id(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().upper()
    s = re.sub(r"[^A-Z0-9\-_]", "", s)
    return s if s else None


def normalize_simple_id(val: Optional[Any]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return re.sub(r"\s+", "_", s) if s else None


# -----------------------------
# Heuristic value extraction (fallback)
# -----------------------------
def _extract_from_text(text: str, field: str):
    if not text:
        return None
    s = str(text)

    if field in ("claim_amount", "policy_sum_insured"):
        m = _amount_re.search(s)
        if m:
            return m.group().replace(",", "").strip()

    if field == "phone":
        m = _phone_re.search(s)
        if m:
            return re.sub(r"\D", "", m.group())

    if field == "incident_date":
        m = _date_re.search(s)
        if m:
            return m.group()

    if field in ("policy_id", "policy_id_record", "customer_id", "claim_id", "garage_id"):
        m = _policy_re.search(s)
        if m:
            return m.group()

    return None


# -----------------------------
# DataFrame-level normalizer
# -----------------------------
def normalize_claims_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("normalize_claims_df: empty input df")

    d = df.iloc[0].to_dict()
    out = {
        "claim_id": d.get("claim_id"),
        "customer_id": normalize_simple_id(d.get("customer_id")),
        "policy_id": normalize_policy_id(d.get("policy_id")),
        "policy_id_record": normalize_policy_id(d.get("policy_id_record")),
        "claim_amount": normalize_currency(d.get("claim_amount")),
        "policy_sum_insured": normalize_currency(d.get("policy_sum_insured")),
        "incident_date": normalize_date(d.get("incident_date")),
        "description": (d.get("description") or "").strip(),
        "phone": normalize_phone(d.get("phone")),
        "garage_id": normalize_simple_id(d.get("garage_id")),
    }

    out["missing_amount_flag"] = int(out["claim_amount"] is None)
    out["missing_info_flag"] = int(not bool(out["phone"]) or not bool(out["garage_id"]))

    return pd.DataFrame([out])


# -----------------------------
# Dict-level normalization (FINAL)
# -----------------------------
def normalize_claim_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Most robust normalizer: uses DF-level first, then fallback extraction,
    and adds _field_confidence map.
    """
    import pandas as _pd

    if d is None:
        return {}

    # ------------------------------------------
    # First-pass DF normalization
    # ------------------------------------------
    try:
        df = _pd.DataFrame([d])
        out = normalize_claims_df(df).iloc[0].to_dict()
    except Exception:
        out = {
            "claim_id": d.get("claim_id"),
            "customer_id": normalize_simple_id(d.get("customer_id")),
            "policy_id": normalize_policy_id(d.get("policy_id")),
            "policy_id_record": normalize_policy_id(d.get("policy_id_record")),
            "claim_amount": normalize_currency(d.get("claim_amount")),
            "policy_sum_insured": normalize_currency(d.get("policy_sum_insured")),
            "incident_date": normalize_date(d.get("incident_date")),
            "description": (d.get("description") or "").strip(),
            "phone": normalize_phone(d.get("phone")),
            "garage_id": normalize_simple_id(d.get("garage_id")),
        }

    # ------------------------------------------
    # Build field confidence + fallback extraction
    # ------------------------------------------
    field_conf = {}
    canonical_fields = [
        "claim_id","customer_id","policy_id","policy_id_record",
        "claim_amount","policy_sum_insured","incident_date",
        "description","phone","garage_id"
    ]

    combined_text = " ".join([str(v) for v in d.values() if v])

    for f in canonical_fields:
        if out.get(f) not in (None, "", [], {}):
            field_conf[f] = 1.0
            continue

        # fallback extract from description or all-text
        candidate = _extract_from_text(out.get("description", ""), f)
        if candidate is None:
            candidate = _extract_from_text(combined_text, f)

        if candidate:
            # normalize extracted candidate
            if f in ("claim_amount","policy_sum_insured"):
                out[f] = normalize_currency(candidate)
            elif f == "incident_date":
                out[f] = normalize_date(candidate)
            elif f == "phone":
                out[f] = normalize_phone(candidate)
            else:
                out[f] = str(candidate)

            field_conf[f] = 0.6
        else:
            field_conf[f] = 0.0

    out["missing_amount_flag"] = int(out.get("claim_amount") is None)
    out["missing_info_flag"] = int(not bool(out.get("phone")) or not bool(out.get("garage_id")))

    out["_field_confidence"] = field_conf

    return out
