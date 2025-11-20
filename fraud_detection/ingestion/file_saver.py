"""
file_saver.py

Centralized saving of uploaded or local files into project's raw data folder.

Behavior:
- Uses FD_PROJECT_ROOT environment variable (preferred) for project root.
- Falls back to automatic detection if env var missing.
- Reads raw folder path from fraud_detection/configs/app.yaml if present.
- Supports saving:
    - local filesystem paths (str / Path)
    - UploadFile-like objects (FastAPI UploadFile)
- Tries to use fraud_detection.storage.store abstraction if available; falls back to local storage.
- Returns pathlib.Path objects (absolute).

If you move the project root, set FD_PROJECT_ROOT env var to new location.
"""

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Union

import yaml

from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# ---------------------------
# Project root resolution
# ---------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    # fallback - assume package is at <project_root>/fraud_detection/...
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------
# Optional storage abstraction (S3/local)
# ---------------------------
try:
    from fraud_detection.storage import store as storage_store  # type: ignore
except Exception:
    storage_store = None

# ---------------------------
# Load raw_data_path from config (if present)
# ---------------------------
def _load_raw_data_path() -> Path:
    cfg = PROJECT_ROOT / "fraud_detection" / "configs" / "app.yaml"
    default = PROJECT_ROOT / "fraud_detection" / "data" / "raw"
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


# ---------------------------
# Helpers
# ---------------------------
def _unique_filename(original_name: str) -> str:
    """
    Build a collision-resistant filename preserving extension.
    """
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
    """
    Saves a file-like UploadFile (.file, .filename) to dest path in chunks.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # ensure file pointer at start
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


# ---------------------------
# Public API
# ---------------------------
def save_file(source: Union[str, Path, object]) -> Path:
    """
    Save a single file. 'source' can be:
      - a filesystem path (string or Path) to an existing file
      - an UploadFile-like object with .filename and .file (FastAPI)
    Returns absolute Path to saved file.
    """
    # Local file path
    if isinstance(source, (str, Path)) and Path(source).exists():
        src_path = Path(source).resolve()
        dest_name = _unique_filename(src_path.name)
        dest = RAW_FOLDER / dest_name

        # Prefer storage abstraction when available
        if storage_store and hasattr(storage_store, "save_raw_file"):
            try:
                # Expected interface: save_raw_file(src_path: str, dest_folder: str) -> str/Path
                final = storage_store.save_raw_file(str(src_path), dest_folder=str(RAW_FOLDER))
                logger.info("Saved via storage abstraction: %s -> %s", src_path, final)
                return Path(final)
            except Exception as e:
                logger.warning("storage.save_raw_file failed, falling back to local: %s", e)

        saved = _save_local_copy(src_path, dest)
        logger.info("Saved local file: %s", saved)
        return saved

    # UploadFile-like object (FastAPI)
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
    """
    Save multiple files and return list of Paths for successfully saved files.
    Errors are logged and problematic files are skipped.
    """
    saved = []
    for s in sources or []:
        try:
            p = save_file(s)
            saved.append(p)
        except Exception as e:
            logger.exception("Failed to save file %s: %s", getattr(s, "filename", s), e)
    return saved
