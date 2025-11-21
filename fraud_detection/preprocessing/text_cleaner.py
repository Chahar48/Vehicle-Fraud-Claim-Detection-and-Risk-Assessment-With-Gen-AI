"""
Text cleaning utilities used by the pipeline.
- clean_description(text, redact=False) -> str
- summarize_desc(text, max_sentences=2) -> str
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Optional

try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("text_cleaner")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)


# Project root (not required but kept for consistency)
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -------------------------
# Patterns
# -------------------------
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s\-\.\)]*)?(?:\(?\d{2,4}\)?[\s\-\.\)]*)?\d{6,12})")
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")  # simple 12-digit pattern


def remove_control_chars(text: Optional[str]) -> str:
    if not text:
        return ""
    return CONTROL_CHAR_RE.sub("", text)


def normalize_whitespace(text: Optional[str]) -> str:
    if not text:
        return ""
    t = text.replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()


def reduce_ocr_noise(text: Optional[str]) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"\-{3,}", "-", t)   # long dashes
    t = re.sub(r"([^\w\s])\1{2,}", r"\1", t)  # repeated punctuation
    return t


def redact_emails(text: str, mask: str = "[REDACTED_EMAIL]") -> str:
    if not text:
        return ""
    return EMAIL_RE.sub(mask, text)


def redact_phones(text: str, mask: str = "[REDACTED_PHONE]") -> str:
    if not text:
        return ""
    def _mask(m):
        s = re.sub(r"\D", "", m.group(0))
        return mask if len(s) >= 6 else m.group(0)
    return PHONE_RE.sub(_mask, text)


def redact_aadhaar(text: str, mask: str = "[REDACTED_ID]") -> str:
    if not text:
        return ""
    return AADHAAR_RE.sub(mask, text)


def redact_pii(text: str, redact_email: bool = True, redact_phone: bool = True, redact_id: bool = True) -> str:
    t = text or ""
    if redact_email:
        t = redact_emails(t)
    if redact_phone:
        t = redact_phones(t)
    if redact_id:
        t = redact_aadhaar(t)
    return t


def clean_description(text: Optional[str], redact: bool = True, max_length: int = 4000) -> str:
    """
    Clean and normalize a free-text description field.
    - Remove control chars
    - Reduce OCR noise conservatively
    - Normalize whitespace
    - Optionally redact PII
    - Trim to max_length (don't break words badly)
    """
    if not text:
        return ""

    t = remove_control_chars(text)
    t = reduce_ocr_noise(t)
    t = normalize_whitespace(t)

    if redact:
        t = redact_pii(t)

    if len(t) > max_length:
        t = t[:max_length]
        last_space = t.rfind(" ")
        if last_space > int(max_length * 0.5):
            t = t[:last_space]
    return t.strip()


def summarize_desc(text: Optional[str], max_sentences: int = 2) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:max_sentences])


# -------------------------
# module test
# -------------------------
if __name__ == "__main__":
    sample = "Contact: +91 98765-43210, email: test.user@example.com. Claim: Scratch on bumper. CONFIDENTIAL."
    print("Cleaned:", clean_description(sample))
    print("Summ:", summarize_desc(sample))
