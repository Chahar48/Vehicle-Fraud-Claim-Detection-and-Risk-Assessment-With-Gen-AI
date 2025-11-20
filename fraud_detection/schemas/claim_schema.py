# fraud_detection/schemas/claim_schema.py
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional
from datetime import datetime
import re


class ClaimSchema(BaseModel):

    # -----------------------------
    # Pydantic v2 model config
    # -----------------------------
    model_config = ConfigDict(
        str_strip_whitespace=True,     # auto-trim whitespace
        validate_default=True,         # apply validators on default values
        extra="ignore",                # ignore unknown fields from LLM
        arbitrary_types_allowed=True
    )

    # -----------------------------
    # CLAIM FIELDS
    # -----------------------------
    claim_id: Optional[str]
    customer_id: Optional[str]
    policy_id: Optional[str]
    policy_id_record: Optional[str]
    claim_amount: Optional[float]
    policy_sum_insured: Optional[float]
    incident_date: Optional[str]     # stored as YYYY-MM-DD string
    description: Optional[str]
    phone: Optional[str]
    garage_id: Optional[str]

    # -------------------------------------------------------
    # VALIDATORS — converted to Pydantic v2 style
    # -------------------------------------------------------

    @field_validator("incident_date")
    def validate_incident_date(cls, v):
        """Convert many date formats → YYYY-MM-DD"""
        if not v:
            return None

        # If already correct format
        try:
            dt = datetime.strptime(v, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except:
            pass

        # Try multiple formats
        date_formats = ["%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]

        for fmt in date_formats:
            try:
                dt = datetime.strptime(str(v), fmt)
                return dt.strftime("%Y-%m-%d")
            except:
                continue

        return None

    @field_validator("phone")
    def validate_phone(cls, v):
        """Extract digits and ensure >=10 digits."""
        if not v:
            return None
        digits = "".join(ch for ch in str(v) if ch.isdigit())
        return digits if len(digits) >= 10 else None

    @field_validator("claim_amount", "policy_sum_insured")
    def validate_numeric(cls, v):
        """Convert numeric-like values safely."""
        if v is None or str(v).strip().lower() == "nan":
            return None
        try:
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                return None
            return f
        except:
            return None

    @field_validator("policy_id", "policy_id_record")
    def clean_policy_ids(cls, v):
        """Uppercase & remove invalid characters."""
        if not v:
            return None
        v = str(v).strip().upper()
        return re.sub(r"[^A-Z0-9\-_]", "", v)

    @field_validator("garage_id", "customer_id")
    def clean_simple_ids(cls, v):
        """Basic ID cleanup."""
        if not v:
            return None
        return str(v).strip().replace(" ", "_")
