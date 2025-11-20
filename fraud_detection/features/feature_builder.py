"""
feature_builder.py
------------------
Phase 8 — Numeric Feature Engineering

This module converts a normalized + validated claim dictionary
into a set of numeric features for scoring.

Works in:
- Single-claim mode (dict)
- Batch mode (DataFrame)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------
# Safe date parsing
# ------------------------------------------------------------
def _to_datetime_safe(val) -> Optional[datetime]:
    if val is None or val == "":
        return None
    try:
        return pd.to_datetime(val)
    except Exception:
        return None


# ------------------------------------------------------------
# Build numeric features for ONE claim dict
# ------------------------------------------------------------
def build_features_from_dict(claim: Dict[str, Any],
                             history_df: Optional[pd.DataFrame] = None
                             ) -> Dict[str, Any]:
    """
    Input: sanitized + validated claim dict
    Output: numeric feature dictionary
    """

    claim_id = claim.get("claim_id")
    customer_id = claim.get("customer_id")
    claim_amount = claim.get("claim_amount")
    psi = claim.get("policy_sum_insured")

    incident_date = _to_datetime_safe(claim.get("incident_date"))

    # ------------------------------------------------------------
    # Core numeric features
    # ------------------------------------------------------------
    amount_ratio = None
    if claim_amount is not None and psi not in [None, 0]:
        try:
            amount_ratio = float(claim_amount) / float(psi)
        except Exception:
            amount_ratio = None

    # ------------------------------------------------------------
    # Claims history features (PoC)
    # ------------------------------------------------------------
    claims_last_12m = 0
    days_since_last_claim = None
    is_new_customer = 1

    if history_df is not None and customer_id:
        df = history_df.copy()
        df["incident_date_dt"] = df["incident_date"].apply(_to_datetime_safe)

        one_year_ago = incident_date - pd.DateOffset(months=12) if incident_date else None

        # last 12 months
        if incident_date is not None:
            mask = (
                (df["customer_id"] == customer_id) &
                (df["incident_date_dt"] >= one_year_ago) &
                (df["incident_date_dt"] < incident_date)
            )
            claims_last_12m = int(mask.sum())

        # days since last claim
        past_claims = df[
            (df["customer_id"] == customer_id) &
            (df["incident_date_dt"] < incident_date)
        ]

        if not past_claims.empty:
            last_date = past_claims["incident_date_dt"].max()
            days_since_last_claim = (incident_date - last_date).days if incident_date else None

        # new customer?
        is_new_customer = 0 if customer_id in set(df["customer_id"]) else 1

    # ------------------------------------------------------------
    # Missing flags
    # ------------------------------------------------------------
    missing_amount_flag = 1 if claim_amount is None else 0
    missing_policy_flag = 1 if psi is None else 0
    missing_date_flag = 1 if incident_date is None else 0

    features = {
        "claim_id": claim_id,
        "customer_id": customer_id,
        "claim_amount": claim_amount if claim_amount is not None else -1,
        "policy_sum_insured": psi if psi is not None else -1,

        "amount_ratio": amount_ratio if amount_ratio is not None else 0,

        "claims_last_12m": claims_last_12m,
        "days_since_last_claim": days_since_last_claim if days_since_last_claim is not None else -1,
        "is_new_customer": is_new_customer,

        # Missing flags
        "missing_amount_flag": missing_amount_flag,
        "missing_policy_flag": missing_policy_flag,
        "missing_date_flag": missing_date_flag,
    }

    logger.debug("Numeric features built: %s", features)
    return features


# ------------------------------------------------------------
# Batch mode for DataFrame (optional)
# ------------------------------------------------------------
def build_features_from_df(df: pd.DataFrame,
                           history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Converts each row of DataFrame into numeric features.
    """

    rows = []
    for _, row in df.iterrows():
        features = build_features_from_dict(row.to_dict(), history_df=history_df)
        rows.append(features)

    return pd.DataFrame(rows)
