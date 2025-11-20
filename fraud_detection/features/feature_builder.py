"""
feature_builder.py

Numeric feature engineering (Phase 8). Backwards-compatible API:
 - build_numeric_features(df_or_record, history_df=None) -> pd.DataFrame
 - build_features_from_df(df, history_df=None) -> pd.DataFrame
 - build_features_from_dict(record, history_df=None) -> dict
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# logger
try:
    from fraud_detection.logging.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fraud_detection.features.feature_builder")


def _to_datetime_safe(val) -> Optional[pd.Timestamp]:
    if val is None or (isinstance(val, str) and str(val).strip() == ""):
        return None
    try:
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return None


def build_features_from_dict(claim: Dict[str, Any],
                             history_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Convert a single sanitized claim dict into numeric features.
    Expected keys: claim_id, customer_id, claim_amount, policy_sum_insured, incident_date
    """
    claim_id = claim.get("claim_id")
    customer_id = claim.get("customer_id")

    # Defensive numeric casting
    try:
        claim_amount = None if claim.get("claim_amount") in [None, ""] else float(claim.get("claim_amount"))
    except Exception:
        claim_amount = None

    try:
        psi = None if claim.get("policy_sum_insured") in [None, ""] else float(claim.get("policy_sum_insured"))
    except Exception:
        psi = None

    incident_date = _to_datetime_safe(claim.get("incident_date"))

    # amount_ratio
    amount_ratio = 0.0
    if claim_amount is not None and psi not in (None, 0):
        try:
            amount_ratio = float(claim_amount) / float(psi)
        except Exception:
            amount_ratio = 0.0

    # history-derived features
    claims_last_12m = 0
    days_since_last_claim = -1
    is_new_customer = 1

    if history_df is not None and customer_id:
        try:
            df = history_df.copy()
            # ensure incident_date present in history
            if "incident_date" in df.columns:
                df["incident_date_dt"] = df["incident_date"].apply(_to_datetime_safe)
            else:
                df["incident_date_dt"] = pd.NaT

            if incident_date is not None:
                one_year_ago = incident_date - pd.DateOffset(months=12)
                mask = (
                    (df.get("customer_id") == customer_id)
                    & (df["incident_date_dt"] >= one_year_ago)
                    & (df["incident_date_dt"] < incident_date)
                )
                claims_last_12m = int(mask.sum())

                past_claims = df[(df.get("customer_id") == customer_id) & (df["incident_date_dt"] < incident_date)]
                if not past_claims.empty:
                    last_date = past_claims["incident_date_dt"].max()
                    if pd.notna(last_date):
                        days_since_last_claim = int((incident_date - last_date).days)
            # new customer flag
            try:
                existing_customers = set(df.get("customer_id").dropna().astype(str).tolist())
                is_new_customer = 0 if (customer_id is not None and str(customer_id) in existing_customers) else 1
            except Exception:
                is_new_customer = 1
        except Exception:
            logger.debug("History processing error for customer_id=%s", customer_id, exc_info=True)

    missing_amount_flag = 1 if claim_amount is None else 0
    missing_policy_flag = 1 if psi is None else 0
    missing_date_flag = 1 if incident_date is None else 0

    features = {
        "claim_id": claim_id,
        "customer_id": customer_id,
        "claim_amount": claim_amount if claim_amount is not None else -1.0,
        "policy_sum_insured": psi if psi is not None else -1.0,
        "amount_ratio": float(amount_ratio if amount_ratio is not None else 0.0),
        "claims_last_12m": int(claims_last_12m),
        "days_since_last_claim": int(days_since_last_claim) if days_since_last_claim is not None else -1,
        "is_new_customer": int(is_new_customer),
        "missing_amount_flag": int(missing_amount_flag),
        "missing_policy_flag": int(missing_policy_flag),
        "missing_date_flag": int(missing_date_flag),
    }

    logger.debug("Numeric features built for %s: %s", claim_id, features)
    return features


def build_features_from_df(df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Convert each row of df into numeric features DataFrame.
    """
    rows = []
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "claim_id", "customer_id", "claim_amount", "policy_sum_insured", "amount_ratio",
            "claims_last_12m", "days_since_last_claim", "is_new_customer",
            "missing_amount_flag", "missing_policy_flag", "missing_date_flag"
        ])
    for _, row in df.iterrows():
        try:
            rec = build_features_from_dict(row.to_dict(), history_df=history_df)
            rows.append(rec)
        except Exception:
            logger.exception("Failed building features for row: %s", row.to_dict())
    return pd.DataFrame(rows)


# Backwards-compatible wrapper expected by pipeline_runner
def build_numeric_features(df_like, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Accepts:
      - pandas.DataFrame -> returns features DataFrame
      - dict / single-row -> returns features DataFrame with single row
    """
    if isinstance(df_like, pd.DataFrame):
        return build_features_from_df(df_like, history_df=history_df)
    # if dict or series-like
    try:
        if isinstance(df_like, dict):
            single = pd.DataFrame([df_like])
        else:
            # try to coerce to DataFrame
            single = pd.DataFrame(df_like)
        return build_features_from_df(single, history_df=history_df)
    except Exception:
        logger.exception("build_numeric_features: could not coerce input to DataFrame")
        return pd.DataFrame()
