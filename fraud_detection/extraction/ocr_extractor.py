"""
ocr_extractor.py
----------------
Responsible for:
- Extracting text from PDF files (digital PDFMiner + scanned OCR fallback)
- Extracting text from images (JPG/PNG)
- Preprocessing images to improve OCR accuracy
- Returning clean raw text for LLM extraction

Integrated with:
- FD_PROJECT_ROOT (environment-based project root)
- Central logging system
- Config-driven behavior (OCR DPI, languages, etc.)
"""

import os
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image, ImageFilter
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdf2image import convert_from_path

from fraud_detection.logging.logger import get_logger
import yaml

logger = get_logger(__name__)

# ---------------------------------------------------------
# PROJECT ROOT (env first, fallback)
# ---------------------------------------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------
# LOAD OCR CONFIG
# ---------------------------------------------------------
def _load_ocr_config():
    cfg_path = PROJECT_ROOT / "fraud_detection" / "configs" / "ocr.yaml"
    default_cfg = {
        "dpi": 200,
        "language": "eng",
        "min_text_length": 30,  # PDFMiner fallback threshold
    }

    if not cfg_path.exists():
        return default_cfg

    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return {**default_cfg, **cfg}
    except Exception as e:
        logger.warning("Failed loading ocr.yaml config: %s", e)
        return default_cfg


OCR_CONFIG = _load_ocr_config()


# =========================================================
#  IMAGE OCR FUNCTIONS
# =========================================================

def _preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocesses image for better OCR accuracy:
    - Convert to grayscale
    - Upscale (helps OCR)
    - Apply sharpen filter
    - Binarize (threshold)
    """
    try:
        img = img.convert("L")  # grayscale

        w, h = img.size
        img = img.resize((w * 2, h * 2))

        img = img.filter(ImageFilter.SHARPEN)

        img = img.point(lambda x: 0 if x < 150 else 255)

        return img
    except Exception as e:
        logger.warning("Image preprocessing failed: %s", e)
        return img  # fallback


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image (jpg/png).
    """
    p = Path(image_path)

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(p)
        img = _preprocess_image(img)

        text = pytesseract.image_to_string(
            img, lang=OCR_CONFIG.get("language", "eng")
        )

        logger.info("Image OCR completed: %s", image_path)
        return text.strip()

    except Exception as e:
        logger.exception("OCR failed for image: %s", e)
        return ""


# =========================================================
# PDF OCR FUNCTIONS (digital + scanned)
# =========================================================

def _extract_pdf_pdfminer(pdf_path: str) -> str:
    """
    Extract text from digital PDF using PDFMiner.
    """
    try:
        text = pdfminer_extract_text(pdf_path)
        text = text.strip()
        if text:
            logger.debug("PDFMiner extracted %d chars from %s", len(text), pdf_path)
        return text
    except Exception as e:
        logger.warning("PDFMiner failed on %s: %s", pdf_path, e)
        return ""


def _pdf_to_images(pdf_path: str) -> List[Image.Image]:
    try:
        return convert_from_path(pdf_path, dpi=OCR_CONFIG["dpi"])
    except Exception as e:
        logger.error("Failed converting PDF to images: %s", e)
        return []


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Full PDF extraction pipeline:
    1. Attempt digital text extraction using PDFMiner
    2. If too little text, fallback to OCR (scanned PDF)
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Extracting PDF text: %s", pdf_path)

    # Step 1 — Digital extraction
    pdfminer_text = _extract_pdf_pdfminer(pdf_path)

    if len(pdfminer_text) >= OCR_CONFIG["min_text_length"]:
        return pdfminer_text

    # Step 2 — Scanned PDF OCR fallback
    logger.info("PDF seems scanned. Using OCR fallback...")

    pages = _pdf_to_images(pdf_path)
    if not pages:
        return ""

    ocr_text = []

    for idx, page in enumerate(pages):
        try:
            processed = _preprocess_image(page)
            page_text = pytesseract.image_to_string(
                processed, lang=OCR_CONFIG.get("language", "eng")
            )
            ocr_text.append(page_text)
        except Exception as e:
            logger.error("OCR failed for PDF page %d: %s", idx, e)

    full_text = "\n".join(ocr_text).strip()
    logger.info("Scanned PDF OCR completed: %s", pdf_path)

    return full_text


# =========================================================
# GENERAL ENTRYPOINT
# =========================================================

def extract_text(file_path: str) -> str:
    """
    General-purpose entrypoint:
    - Auto-detect file extension
    - Route to proper extractor
    - Return raw text for LLM extraction
    """
    ext = Path(file_path).suffix.lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    raise ValueError(f"OCR not supported for file type: {ext}")
