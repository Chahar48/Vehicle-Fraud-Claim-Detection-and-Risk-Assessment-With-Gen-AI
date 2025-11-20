"""
text_extractor.py
-----------------
Utility functions for cleaning and preparing OCR text
before sending it to the LLM for structured extraction.

Responsibilities:
- Clean raw OCR text
- Normalize spacing, punctuation
- Remove header/footer noise
- Remove repeated lines
- Split text into chunks for LLM (if needed)
- Return cleaned, LLM-ready text

Aligned with:
- FD_PROJECT_ROOT environment system
- Logging architecture
"""

import os
import re
from pathlib import Path
from typing import List

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------
# Project Root (Environment-first)
# ---------------------------------------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
#  BASIC TEXT CLEANING
# =========================================================

def remove_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" .", ".").replace(" ,", ",")
    text = text.replace(" :", ":").replace(" ;", ";")
    return text.strip()


def remove_repeated_lines(lines: List[str]) -> List[str]:
    """
    Removes duplicated lines often produced by OCR.
    """
    seen = set()
    clean_lines = []
    for line in lines:
        if line.strip() and line not in seen:
            clean_lines.append(line)
            seen.add(line)
    return clean_lines


def remove_header_footer(lines: List[str]) -> List[str]:
    """
    Very simple heuristic that removes:
    - page headers
    - page numbers
    - repeated footer text
    - watermark lines
    """
    cleaned = []
    for line in lines:
        line_strip = line.strip()

        # typical patterns to remove
        if re.match(r"^\s*page\s+\d+", line_strip, re.I):
            continue
        if re.match(r"^\s*\d+/\d+$", line_strip):  # e.g. 1/4
            continue
        if "confidential" in line_strip.lower():
            continue
        if "do not distribute" in line_strip.lower():
            continue
        if len(line_strip) < 3:  # skip tiny noise
            continue

        cleaned.append(line)

    return cleaned


# =========================================================
#  MAIN TEXT CLEAN FUNCTION
# =========================================================

def clean_text(raw_text: str) -> str:
    """
    Full pipeline:
    - remove control characters
    - split lines
    - remove repeated lines
    - remove typical header/footer noise
    - normalize whitespace
    """

    if not raw_text:
        logger.warning("clean_text called with empty input")
        return ""

    # 1. Control characters
    text = remove_control_chars(raw_text)

    # 2. Split into lines
    lines = text.splitlines()

    # 3. Remove repeated lines
    lines = remove_repeated_lines(lines)

    # 4. Remove footer/header noise
    lines = remove_header_footer(lines)

    # 5. Reassemble
    cleaned = "\n".join(lines)

    # 6. Whitespace normalization
    cleaned = normalize_whitespace(cleaned)

    logger.debug("Cleaned OCR text (%d chars → %d chars)",
                 len(raw_text), len(cleaned))

    return cleaned


# =========================================================
#  OPTIONAL — CHUNKING FOR LONG TEXTS
# =========================================================

def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """
    Splits text into safe LLM chunks.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start: start + max_chars]
        chunks.append(chunk)
        start += max_chars

    logger.info("chunk_text: split into %d chunks", len(chunks))
    return chunks


# =========================================================
#  MODULE END
# =========================================================

if __name__ == "__main__":
    sample = """
        PAGE 1
        CLAIM FORM
        Claim ID: C1001
        Claim ID: C1001

        Customer ID: CUST001

        Confidential
    """

    cleaned = clean_text(sample)
    print("\n--- Cleaned Text ---\n", cleaned)
