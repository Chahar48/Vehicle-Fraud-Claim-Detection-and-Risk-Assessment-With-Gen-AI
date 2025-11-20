"""
external_lookup.py
------------------
Simplified Enrichment Layer (PoC Version)

This module performs ONLY:
- Customer blacklist check
- Phone blacklist check
- Garage blacklist check

No VIN lookup
No city-based scoring

In production, these would be replaced by API lookups or databases.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# PROJECT ROOT (env-aware)
# ------------------------------------------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# HARD-CODED BLACKLIST DATA (PoC ONLY)
# ------------------------------------------------------------
BLACKLIST_CUSTOMERS = {"C999", "C777", "C888"}       # high-risk customers
BLACKLIST_PHONES = {"9999999999", "8888888888"}      # suspicious numbers
BLACKLIST_GARAGES = {"GARAGE-FAKE-123", "AUTO-RED"}  # bad garages


# ------------------------------------------------------------
# 1. Blacklist checks
# ------------------------------------------------------------
def check_blacklist(
    customer_id: str = None,
    phone: str = None,
    garage_id: str = None
) -> Tuple[int, List[str]]:
    """
    Check if the claim matches any fraud blacklists.

    Returns:
      (flag, reasons_list)
      flag = 1 → match found
      flag = 0 → clean
    """
    reasons = []

    if customer_id and customer_id.upper() in BLACKLIST_CUSTOMERS:
        reasons.append("Customer found in fraud blacklist.")

    if phone and phone in BLACKLIST_PHONES:
        reasons.append("Phone number is listed in fraud blacklist.")

    if garage_id and garage_id.upper() in BLACKLIST_GARAGES:
        reasons.append("Garage flagged as suspicious in blacklist.")

    flag = 1 if reasons else 0

    logger.debug("Blacklist check result: flag=%s, reasons=%s", flag, reasons)
    return flag, reasons


# ------------------------------------------------------------
# 2. MAIN ENRICHMENT WRAPPER
# ------------------------------------------------------------
def enrich_claim_data(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes extracted claim dictionary and adds blacklist-based signals.

    context example:
    {
        "customer_id": "C001",
        "phone": "9876543210",
        "garage_id": "GA-01"
    }

    Returns:
      {
        "blacklist_flag": 0/1,
        "blacklist_reasons": [...]
      }
    """
    customer_id = context.get("customer_id")
    phone = context.get("phone")
    garage_id = context.get("garage_id")

    logger.info("Running blacklist enrichment for customer=%s, garage=%s", customer_id, garage_id)

    flag, reasons = check_blacklist(
        customer_id=customer_id,
        phone=phone,
        garage_id=garage_id
    )

    result = {
        "blacklist_flag": flag,
        "blacklist_reasons": reasons
    }

    logger.debug("Enrichment result: %s", result)
    return result


# ------------------------------------------------------------
# Manual Test
# ------------------------------------------------------------
if __name__ == "__main__":
    test_context = {
        "customer_id": "C999",
        "phone": "9999999999",
        "garage_id": "GARAGE-FAKE-123"
    }

    out = enrich_claim_data(test_context)
    print("\nEnrichment Output:", out)
