"""
unified_extractor.py
--------------------
This file performs the full extraction pipeline:

1. Auto-detect file type (csv, pdf, png, jpg, jpeg, txt)
2. OCR for PDF/Images
3. Clean text using text_extractor
4. Send text to LLM (ChatGroq)
5. Parse structured JSON into claim_dict
6. Validate with ClaimSchema
7. Sanitize dict values (avoid NaN/inf float errors)

This is the core engine that converts ANY raw document
into a normalized claim dictionary.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any

# OCR + Cleaning
from fraud_detection.extraction.ocr_extractor import extract_text
from fraud_detection.extraction.text_extractor import clean_text

# Schema + Sanitizers
from fraud_detection.schemas.claim_schema import ClaimSchema
from fraud_detection.utils.sanitizers import sanitize_dict

# Logging
from fraud_detection.logging.logger import get_logger

# Groq SDK
from groq import Groq
#from openai import OpenAI

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

GROQ_MODEL = MODEL_CONFIG.get("model_name", "llama-3.1-70b-versatile")
#MODEL = MODEL_CONFIG.get("model_name", "gpt-4.1")
TEMPERATURE = MODEL_CONFIG.get("temperature", 0.0)
MAX_TOKENS = MODEL_CONFIG.get("max_tokens", 1024)


# ---------------------------------------------------------
# CREATE GROQ CLIENT
# ---------------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------
# PROMPT TEMPLATE FOR LLM
# ---------------------------------------------------------
PROMPT_TEMPLATE = """
You are an AI assistant that extracts structured insurance claim information from raw OCR text.

### Task:
Read the following text and extract all possible insurance claim fields.

### Required Output Format:
Return ONLY a JSON object with these keys:
{
  "claim_id": "...",
  "customer_id": "...",
  "policy_id": "...",
  "policy_id_record": "...",
  "claim_amount": "...",
  "policy_sum_insured": "...",
  "incident_date": "...",
  "description": "...",
  "phone": "...",
  "garage_id": "..."
}

### Important Rules:
- If a field is missing, return null.
- Do NOT hallucinate values — only use information present in the text.
- JSON must be valid and strictly follow the required keys.

### Text:
{input_text}

Return only the JSON.
"""


# ---------------------------------------------------------
# FUNCTION: Ask Groq LLM for structured extraction
# ---------------------------------------------------------
def llm_extract_claim_fields(text: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.replace("{input_text}", text)

    logger.info("Sending text to Groq LLM for extraction...")

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,#gpt-4.1
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw_output = response.choices[0].message["content"]
        logger.debug("LLM Output: %s", raw_output)

        # Parse strict JSON
        return json.loads(raw_output)

    except Exception as e:
        logger.exception("LLM extraction failed: %s", e)
        raise ValueError(f"LLM extraction error: {e}")


# ---------------------------------------------------------
# AUTO-DETECT FILE TYPE
# ---------------------------------------------------------
def _detect_file_type(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext in [".pdf"]:
        return "pdf"
    if ext in [".png", ".jpg", ".jpeg"]:
        return "image"
    if ext in [".csv"]:
        return "csv"
    if ext in [".txt"]:
        return "text"

    raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------
# MAIN FUNCTION: Unified Extraction
# ---------------------------------------------------------
def extract_claim_from_file(file_path: str) -> Dict[str, Any]:
    """
    Converts ANY file into a structured claim dictionary.

    Steps:
    1. Detect file type
    2. Extract raw text (OCR for PDF/IMG)
    3. Clean text
    4. Send cleaned text to Groq LLM
    5. Validate with ClaimSchema
    6. Sanitize dict values
    """

    ftype = _detect_file_type(file_path)

    # SPECIAL CASE: CSV
    if ftype == "csv":
        raise ValueError(
            "CSV extraction must be performed via ingestion/csv_ingest.py"
        )

    logger.info("Extracting file: %s (type=%s)", file_path, ftype)

    # 1️⃣ OCR / Text extraction
    raw_text = extract_text(file_path)

    if not raw_text.strip():
        raise ValueError("OCR returned empty text")

    # 2️⃣ Clean text
    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        raise ValueError("Cleaned OCR text is empty")

    # 3️⃣ LLM extraction
    extracted = llm_extract_claim_fields(cleaned_text)

    # 4️⃣ Schema Validation
    try:
        claim_obj = ClaimSchema(**extracted)
    except Exception as e:
        logger.error("Schema validation failed: %s", e)
        raise

    # 5️⃣ Sanitize for JSON compatibility
    from fraud_detection.preprocessing.schema_validator import to_plain_dict
    final_claim = sanitize_dict(to_plain_dict(claim_obj))

    logger.info("Extraction completed successfully for: %s", file_path)
    return final_claim


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------
if __name__ == "__main__":
    test_file = "sample_doc.pdf"  # change to test

    try:
        out = extract_claim_from_file(test_file)
        print("\n--- Final Extracted Claim ---")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("ERROR:", e)
