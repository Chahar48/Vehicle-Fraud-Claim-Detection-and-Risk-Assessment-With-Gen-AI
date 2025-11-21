# fraud_detection/ingestion/csv_ingest.py
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import re
import yaml
import pandas as pd
import difflib
import logging

from fraud_detection.logging.logger import get_logger
try:
    # use canonical sanitizers module if present
    from fraud_detection.schema.sanitizers import sanitize_dict
except Exception:
    def sanitize_dict(x): return x

from fraud_detection.schemas.claim_schema import ClaimSchema
from fraud_detection.ingestion.file_saver import save_file

logger = get_logger(__name__)

FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Canonical field names used across pipeline
CANONICAL_FIELDS = [
    "claim_id",
    "customer_id",
    "policy_id",
    "policy_id_record",
    "claim_amount",
    "policy_sum_insured",
    "incident_date",
    "description",
    "phone",
    "garage_id",
]

# Common aliases (lowercase cleaned keys -> canonical)
_COMMON_ALIAS = {
    "claimid": "claim_id", "claim_id": "claim_id", "claim id": "claim_id", "claim": "claim_id",
    "customerid": "customer_id", "customer id": "customer_id", "customer": "customer_id",
    "policyid": "policy_id", "policy id": "policy_id", "policyno": "policy_id",
    "policyidrecord": "policy_id_record", "policy id record": "policy_id_record", "officialpolicyid":"policy_id_record",
    "claimamount": "claim_amount", "claim amount": "claim_amount", "amount": "claim_amount",
    "policysuminsured": "policy_sum_insured", "policy sum insured": "policy_sum_insured",
    "incidentdate": "incident_date", "incident date": "incident_date", "date": "incident_date",
    "description": "description", "desc": "description", "remarks": "description",
    "phone": "phone", "mobile": "phone", "telephone": "phone",
    "garageid": "garage_id", "garage id": "garage_id", "garage": "garage_id",
}

# Threshold for fuzzy matching (0..1). difflib ratio ~ [0..1]
FUZZY_THRESHOLD = 0.7


def _to_key(s: Optional[str]) -> str:
    if s is None:
        return ""
    s2 = str(s).strip().lower()
    # collapse whitespace and non-alnum to single space for alias matching
    s2 = re.sub(r"[^a-z0-9]+", " ", s2).strip()
    return s2


def _fuzzy_map(col: str) -> Optional[str]:
    """
    Map incoming column name to a canonical field.
    1) exact alias lookup
    2) fuzzy match against alias keys and canonical fields (difflib)
    """
    k = _to_key(col)
    if not k:
        return None
    # direct alias
    if k in _COMMON_ALIAS:
        return _COMMON_ALIAS[k]
    # direct canonical
    if k in CANONICAL_FIELDS:
        return k
    # fuzzy against alias keys
    alias_keys = list(_COMMON_ALIAS.keys())
    cand = difflib.get_close_matches(k, alias_keys, n=1, cutoff=FUZZY_THRESHOLD)
    if cand:
        return _COMMON_ALIAS[cand[0]]
    # fuzzy against canonical fields
    cand2 = difflib.get_close_matches(k, CANONICAL_FIELDS, n=1, cutoff=FUZZY_THRESHOLD)
    if cand2:
        return cand2[0]
    # last resort: try removing vowels or small edits — keep as None (will use extractor fallback)
    return None


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        canon = _fuzzy_map(c)
        if canon:
            mapping[c] = canon
        else:
            # canonicalize header to snake like form for downstream traceability
            mapping[c] = re.sub(r"[^a-z0-9]+", "_", _to_key(c)).strip("_")
    return df.rename(columns=mapping)


def _load_required_columns() -> List[str]:
    cfg = PROJECT_ROOT / "fraud_detection" / "configs" / "app.yaml"
    defaults = ["claim_id", "policy_id", "claim_amount", "incident_date", "description"]
    try:
        if cfg.exists():
            with cfg.open("r", encoding="utf-8") as fh:
                conf = yaml.safe_load(fh) or {}
            cols = conf.get("ingestion", {}).get("required_columns")
            if cols and isinstance(cols, list):
                return cols
    except Exception:
        logger.debug("Failed loading required cols from config")
    return defaults


REQUIRED_COLUMNS = _load_required_columns()


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns (post-normalize): {missing}")


# --- row-level pattern extractors (fallback if header mapping failed) ---
_amount_re = re.compile(r"[-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?")
_phone_re = re.compile(r"\+?\d[\d\s\-]{6,}\d")
_date_re = re.compile(r"\b(?:\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b")
_policy_re = re.compile(r"[A-Z0-9\-\_]{4,}")  # simple policy id pattern

def _fallback_extract_value(row: pd.Series, target_field: str) -> Any:
    """
    If mapping didn't produce a value, try to heuristically find it in the row cells.
    """
    # fast map: look for likely cells using regexes
    for _, v in row.items():
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if target_field in ("claim_amount", "policy_sum_insured"):
            m = _amount_re.search(s)
            if m:
                return m.group().replace(",", "").strip()
        if target_field == "phone":
            m = _phone_re.search(s)
            if m:
                return re.sub(r"\D", "", m.group())
        if target_field == "incident_date":
            m = _date_re.search(s)
            if m:
                return m.group()
        if target_field in ("policy_id", "policy_id_record"):
            m = _policy_re.search(s)
            if m and len(m.group()) >= 4:
                return m.group()
        if target_field in ("claim_id", "customer_id", "garage_id"):
            # try short alnum tokens
            m = _policy_re.search(s)
            if m:
                return m.group()
    return None


def load_claims_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(p, dtype=str)
    df = df.fillna("")
    df = _normalize_dataframe_columns(df)
    validate_schema(df)
    logger.info("Loaded CSV (%d rows) with normalized columns: %s", len(df), list(df.columns))
    return df


def _row_to_canonical_dict(row: pd.Series) -> Dict[str, Any]:
    raw = row.to_dict()
    out: Dict[str, Any] = {}
    # prefer already-mapped canonical columns
    for cf in CANONICAL_FIELDS:
        # if column exists exactly, use it
        if cf in row.index:
            val = row.get(cf)
            val = val if (val is not None and str(val).strip() != "") else None
            out[cf] = val
        else:
            # fallback: heuristic search in row for this field
            out[cf] = _fallback_extract_value(row, cf)
    return out


def _validate_row_to_claim(row: pd.Series) -> Dict[str, Any]:
    raw = _row_to_canonical_dict(row)
    try:
        claim_obj = ClaimSchema(**raw)
        # pydantic v2 or v1 handling
        claim_dict = claim_obj.model_dump() if hasattr(claim_obj, "model_dump") else claim_obj.dict()
        claim_dict = sanitize_dict(claim_dict)
        return claim_dict
    except Exception as e:
        logger.debug("Row schema validation failed (trying relaxed sanitize): %s", e)
        partial = sanitize_dict(raw)
        partial["_schema_error"] = str(e)
        return partial


def ingest_file(path: str) -> List[Dict[str, Any]]:
    df = load_claims_csv(path)
    saved = save_file(path)  # persist raw
    logger.info("Persisted raw CSV to: %s", saved)

    claims: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        claims.append(_validate_row_to_claim(row))

    logger.info("CSV ingestion prepared %d claim(s)", len(claims))
    return claims
