from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class ClaimSchema(BaseModel):
    claim_id: Optional[str]
    customer_id: Optional[str]
    policy_id: Optional[str]
    policy_id_record: Optional[str]
    claim_amount: Optional[float]
    policy_sum_insured: Optional[float]
    incident_date: Optional[datetime]
    description: Optional[str]
    phone: Optional[str]
    garage_id: Optional[str]

    @validator("incident_date", pre=True, always=True)
    def validate_date(cls, v):
        """Convert date strings into datetime objects."""
        if v is None or v == "":
            return None

        date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]

        for fmt in date_formats:
            try:
                return datetime.strptime(str(v), fmt)
            except Exception:
                continue
        return None

    @validator("phone", pre=True, always=True)
    def validate_phone(cls, v):
        """Simple phone validation (India 10-digit)."""
        if v is None:
            return None

        digits = "".join([d for d in str(v) if d.isdigit()])
        return digits if len(digits) >= 10 else None

    @validator("claim_amount", "policy_sum_insured", pre=True, always=True)
    def validate_numeric(cls, v):
        """Convert numeric-like values safely."""
        if v is None or v == "" or str(v).lower() == "nan":
            return None
        try:
            return float(v)
        except:
            return None

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
