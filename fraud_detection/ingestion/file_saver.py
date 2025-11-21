"""
file_saver.py

Centralized saving of uploaded or local files into project's raw data folder.

Returns pathlib.Path objects (absolute).
"""

from __future__ import annotations
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Union

import yaml

# logger
try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("file_saver")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

# Project root
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# optional storage abstraction
try:
    from fraud_detection.storage import store as storage_store  # type: ignore
except Exception:
    storage_store = None

def _load_raw_data_path() -> Path:
    cfg = PROJECT_ROOT / "configs" / "app.yaml"
    default = PROJECT_ROOT / "data" / "raw"
    try:
        if cfg.exists():
            with cfg.open("r", encoding="utf-8") as fh:
                conf = yaml.safe_load(fh)
            raw = conf.get("storage", {}).get("raw_data_path")
            if raw:
                p = Path(raw)
                if not p.is_absolute():
                    p = (PROJECT_ROOT / p).resolve()
                return p
    except Exception as e:
        logger.warning("Unable to read raw_data_path from config: %s", e)
    return default

RAW_FOLDER = _load_raw_data_path()
RAW_FOLDER.mkdir(parents=True, exist_ok=True)


def _unique_filename(original_name: str) -> str:
    stem = Path(original_name).stem
    ext = Path(original_name).suffix
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:8]
    return f"{stem}_{ts}_{uid}{ext}"


def _save_local_copy(src: Union[str, Path], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))
    return dest


def _save_uploadfile(upload, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        upload.file.seek(0)
    except Exception:
        pass
    with open(dest, "wb") as out_f:
        while True:
            chunk = upload.file.read(1024 * 64)
            if not chunk:
                break
            out_f.write(chunk)
    return dest


def save_file(source: Union[str, Path, object]) -> Path:
    """
    Save a single file. 'source' can be:
      - filesystem path (string or Path) to an existing file
      - an UploadFile-like object with .filename and .file (FastAPI)
    Returns absolute Path to saved file.
    """
    # existing local path
    if isinstance(source, (str, Path)) and Path(source).exists():
        src_path = Path(source).resolve()
        dest_name = _unique_filename(src_path.name)
        dest = RAW_FOLDER / dest_name

        if storage_store and hasattr(storage_store, "save_raw_file"):
            try:
                final = storage_store.save_raw_file(str(src_path), dest_folder=str(RAW_FOLDER))
                logger.info("Saved via storage abstraction: %s -> %s", src_path, final)
                return Path(final)
            except Exception as e:
                logger.warning("storage.save_raw_file failed, falling back to local: %s", e)

        saved = _save_local_copy(src_path, dest)
        logger.info("Saved local file: %s", saved)
        return saved

    # UploadFile-like (FastAPI)
    if hasattr(source, "filename") and hasattr(source, "file"):
        filename = getattr(source, "filename", "upload")
        dest_name = _unique_filename(filename)
        dest = RAW_FOLDER / dest_name

        if storage_store and hasattr(storage_store, "save_raw_upload"):
            try:
                final = storage_store.save_raw_upload(source, dest_folder=str(RAW_FOLDER))
                logger.info("Saved upload via storage abstraction: %s", final)
                return Path(final)
            except Exception as e:
                logger.warning("storage.save_raw_upload failed, falling back to local save: %s", e)

        saved = _save_uploadfile(source, dest)
        logger.info("Saved uploaded file: %s", saved)
        return saved

    raise ValueError("Unsupported 'source' type for save_file. Provide file path or UploadFile-like object.")


def save_files(sources: Iterable[Union[str, Path, object]]) -> List[Path]:
    saved: List[Path] = []
    for s in sources or []:
        try:
            p = save_file(s)
            saved.append(p)
        except Exception as e:
            logger.exception("Failed to save file %s: %s", getattr(s, "filename", s), e)
    return saved
