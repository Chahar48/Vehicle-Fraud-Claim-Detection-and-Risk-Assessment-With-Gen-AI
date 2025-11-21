"""
file_router.py

Lightweight, config-driven routing helper.

Responsibilities:
- Validate extension using allowed_extensions from configs/app.yaml
- Detect MIME type (guessed)
- Return routing metadata (original_path, extension, mime, category)

Does NOT move/copy files. Use file_saver.save_file() for persistence.
"""

from __future__ import annotations
import mimetypes
import os
from pathlib import Path
from typing import Dict
import yaml

# defensive logger
try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("file_router")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

# Project root
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load allowed extensions from config
def _load_allowed_extensions() -> set:
    cfg = PROJECT_ROOT / "configs" / "app.yaml"
    if cfg.exists():
        try:
            with cfg.open("r", encoding="utf-8") as fh:
                conf = yaml.safe_load(fh)
            exts = conf.get("files", {}).get("allowed_extensions")
            if exts:
                return {e.lower().lstrip(".") for e in exts}
        except Exception as e:
            logger.warning("Failed to read allowed_extensions from config: %s", e)
    return {"csv", "pdf", "jpg", "jpeg", "png"}

ALLOWED_EXTENSIONS = _load_allowed_extensions()


def get_extension(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


def validate_file_type(path: str) -> str:
    ext = get_extension(path)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    return ext


def detect_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def route_metadata(path: str) -> Dict[str, str]:
    """
    Returns metadata dict:
      - original_path (abs)
      - extension (without dot)
      - mime
      - category ('image'|'csv'|'document')
    """
    p = Path(path)
    if not p.exists():
        logger.warning("route_metadata: file does not exist: %s", path)

    ext = validate_file_type(path)
    mime = detect_mime(path)

    if ext in {"jpg", "jpeg", "png"} or (mime and mime.startswith("image/")):
        category = "image"
    elif ext == "csv" or (mime and mime == "text/csv"):
        category = "csv"
    else:
        category = "document"

    meta = {
        "original_path": str(p.resolve()) if p.exists() else str(Path(path).absolute()),
        "extension": ext,
        "mime": mime,
        "category": category,
    }
    logger.debug("route_metadata: %s", meta)
    return meta
