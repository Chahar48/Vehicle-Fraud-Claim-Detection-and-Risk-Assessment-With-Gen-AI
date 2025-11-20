# fraud_detection/preprocessing/text_cleaner.py

import os
import re
from pathlib import Path
from typing import Optional, List

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# Project root (env-first)
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -------------------------
# Basic cleaning helpers
# -------------------------
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-\.])?(?:\(?\d{2,4}\)?[\s-\.]?)?\d{6,12})")
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")  # indian 12-digit-ish simple pattern


def remove_control_chars(text: str) -> str:
    if not text:
        return ""
    return CONTROL_CHAR_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    # collapse multiple spaces/newlines to single space, preserve line breaks optionally
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def reduce_ocr_noise(text: str) -> str:
    """
    Remove common OCR artifacts: duplicated characters, broken ligatures, etc.
    This is deliberately conservative to avoid damaging real content.
    """
    if not text:
        return ""
    # Remove repeated hyphens lines like "------"
    text = re.sub(r"\-{3,}", "-", text)
    # Fix common mis-OCR: l vs 1 patterns near words (conservative)
    text = re.sub(r"\b0O\b", "OO", text)  # example
    # Remove isolated non-word characters repeated
    text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)
    return text


# -------------------------
# PII redaction (optional)
# -------------------------
def redact_emails(text: str, mask: str = "[REDACTED_EMAIL]") -> str:
    if not text:
        return ""
    return EMAIL_RE.sub(mask, text)


def redact_phones(text: str, mask: str = "[REDACTED_PHONE]") -> str:
    if not text:
        return ""
    # Conservative: only redact digit sequences >= 6
    def _mask(m):
        s = re.sub(r"\D", "", m.group(0))
        return mask if len(s) >= 6 else m.group(0)
    return PHONE_RE.sub(_mask, text)


def redact_aadhaar(text: str, mask: str = "[REDACTED_ID]") -> str:
    if not text:
        return ""
    return AADHAAR_RE.sub(mask, text)


def redact_pii(text: str, redact_email=True, redact_phone=True, redact_id=True) -> str:
    t = text
    if redact_email:
        t = redact_emails(t)
    if redact_phone:
        t = redact_phones(t)
    if redact_id:
        t = redact_aadhaar(t)
    return t


# -------------------------
# High-level API
# -------------------------
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

    # Trim to max_length preserving words
    if len(t) > max_length:
        t = t[:max_length]
        # try to cut at last space
        last_space = t.rfind(" ")
        if last_space > int(max_length * 0.5):
            t = t[:last_space]
    return t.strip()


def summarize_desc(text: Optional[str], max_sentences: int = 2) -> str:
    """
    Simple deterministic summary: return first max_sentences sentences.
    This avoids calling LLM for small summaries.
    """
    if not text:
        return ""
    # split on sentence-like punctuation (simple)
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
