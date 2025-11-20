"""
unified_extractor.py
--------------------
This file performs the full extraction pipeline:

1. Auto-detect file type (csv, pdf, png, jpg, jpeg, txt)
2. OCR for PDF/Images
3. Clean text using text_extractor
4. Send text to Groq LLM with strong JSON extraction prompt
5. Clean & parse JSON safely (handles ```json … ``` formats)
6. Convert empty strings to None → required for Pydantic v2
7. Validate with ClaimSchema (Pydantic v2)
8. Sanitize values to avoid NaN/inf errors

This is the core engine that converts ANY raw document
into a normalized, validated claim dictionary.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any

from groq import Groq

# OCR + Cleaning
from fraud_detection.extraction.ocr_extractor import extract_text
from fraud_detection.extraction.text_extractor import clean_text

# Schema + Sanitizers
from fraud_detection.schemas.claim_schema import ClaimSchema
from fraud_detection.utils.sanitizers import sanitize_dict

# Logging
from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# LOAD MODEL CONFIG
# ---------------------------------------------------------
def _load_model_config():
    cfg_path = PROJECT_ROOT / "fraud_detection" / "configs" / "model.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("Missing configs/model.yaml")

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("extraction_model", {})
    except Exception as e:
        logger.error("Failed loading model.yaml: %s", e)
        raise


MODEL_CONFIG = _load_model_config()

# default Groq model
GROQ_MODEL = MODEL_CONFIG.get("model_name", "llama-3.1-70b-versatile")
TEMPERATURE = MODEL_CONFIG.get("temperature", 0.0)
MAX_TOKENS = MODEL_CONFIG.get("max_tokens", 1024)


# ---------------------------------------------------------
# CREATE GROQ CLIENT
# ---------------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------------
PROMPT_TEMPLATE = """
You are an AI assistant that extracts structured insurance claim information from raw OCR text.

### Task:
Read the following text and extract all possible insurance claim fields.

### Required JSON Format:
Return ONLY a JSON object with exactly these keys:
{
  "claim_id": null or "...",
  "customer_id": null or "...",
  "policy_id": null or "...",
  "policy_id_record": null or "...",
  "claim_amount": null or "...",
  "policy_sum_insured": null or "...",
  "incident_date": null or "...",
  "description": null or "...",
  "phone": null or "...",
  "garage_id": null or "..."
}

### Critical Rules:
- If a field is missing, return null.
- STRICTLY DO NOT hallucinate.
- JSON must be valid and contain ONLY the required keys.
- No markdown, no comments, no ```json fences.

### Text:
{input_text}

Return ONLY valid JSON.
"""


# ---------------------------------------------------------
# LLM JSON Extraction
# ---------------------------------------------------------
def llm_extract_claim_fields(text: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.replace("{input_text}", text)

    logger.info("Sending OCR text to Groq LLM…")

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw_output = response.choices[0].message["content"]
        logger.debug("Raw LLM output: %s", raw_output)

    except Exception as e:
        logger.exception("Groq LLM request failed: %s", e)
        raise ValueError(f"LLM extraction error: {e}")

    # -----------------------------------------------------
    # FIX JSON: remove ```json fences, whitespace, artifacts
    # -----------------------------------------------------
    clean_json = (
        raw_output.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    # -----------------------------------------------------
    # Parse JSON strictly
    # -----------------------------------------------------
    try:
        extracted = json.loads(clean_json)
        return extracted
    except Exception as e:
        logger.error("Failed parsing LLM JSON: %s", clean_json)
        raise ValueError(f"Invalid JSON returned by LLM: {e}")


# ---------------------------------------------------------
# FILE TYPE DETECTION
# ---------------------------------------------------------
def _detect_file_type(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return "pdf"
    if ext in [".jpg", ".jpeg", ".png"]:
        return "image"
    if ext == ".csv":
        raise ValueError("CSV ingestion must be done via ingestion/csv_ingest.py")
    if ext == ".txt":
        return "text"

    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------
# MAIN EXTRACTION PIPELINE
# ---------------------------------------------------------
def extract_claim_from_file(file_path: str) -> Dict[str, Any]:
    """
    Pipeline:
    - Detect file type
    - OCR or text extraction
    - Clean text
    - LLM → structured JSON
    - Convert "" → None
    - Validate with Pydantic v2 ClaimSchema
    - Sanitize dict (remove NaN, Inf, objects)
    """
    ftype = _detect_file_type(file_path)
    logger.info("Extracting claim from file: %s (type=%s)", file_path, ftype)

    # 1️⃣ OCR or plain text
    raw_text = extract_text(file_path)
    if not raw_text or not raw_text.strip():
        raise ValueError("OCR returned empty text")

    # 2️⃣ Clean text and normalize Unicode
    cleaned_text = clean_text(raw_text)
    cleaned_text = cleaned_text.encode("utf-8", "ignore").decode("utf-8")

    if not cleaned_text.strip():
        raise ValueError("Cleaned OCR text is empty")

    # 3️⃣ LLM Extraction
    extracted = llm_extract_claim_fields(cleaned_text)

    # 4️⃣ Convert empty strings → None (required for Pydantic v2)
    extracted = {
        k: (v if v not in ["", " ", "null", "None"] else None)
        for k, v in extracted.items()
    }

    # 5️⃣ Validate with ClaimSchema (Pydantic v2)
    try:
        claim_obj = ClaimSchema(**extracted)
    except Exception as e:
        logger.error("Schema validation failed: %s", e)
        raise

    # 6️⃣ Convert to plain dict & sanitize JSON
    from fraud_detection.preprocessing.schema_validator import to_plain_dict

    final_claim = sanitize_dict(to_plain_dict(claim_obj))

    logger.info("Extraction completed successfully for: %s", file_path)
    return final_claim


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------
if __name__ == "__main__":
    test_file = "sample_doc.pdf"  # change to a real file

    try:
        out = extract_claim_from_file(test_file)
        print("\n--- Final Extracted Claim ---")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("ERROR:", e)
