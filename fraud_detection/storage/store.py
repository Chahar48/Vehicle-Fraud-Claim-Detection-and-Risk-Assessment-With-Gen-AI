"""
store.py
--------
Unified, safe storage layer used across:
- Extraction (OCR / PDF parser)
- Preprocessing
- Feature engineering
- ML models (save/load)
- Decision engine outputs
- Human-in-the-loop labels
- Pipeline runner

All paths are resolved relative to FD_PROJECT_ROOT
to avoid hard-coded errors.
"""

import os
import shutil
import pandas as pd
import joblib

# -------------------------------------------------------------------
# Resolve project root
# -------------------------------------------------------------------
FD_PROJECT_ROOT = os.getenv("FD_PROJECT_ROOT", os.getcwd())

def _abs(path: str) -> str:
    """Convert project-relative path → absolute path safely."""
    return os.path.join(FD_PROJECT_ROOT, path)


# -------------------------------------------------------------------
# Ensure folder exists
# -------------------------------------------------------------------
def ensure_dir(path: str):
    path = _abs(path)
    os.makedirs(path, exist_ok=True)
    return path


# -------------------------------------------------------------------
# Save RAW FILE (PDF / IMAGE / CSV)
# -------------------------------------------------------------------
def save_raw_file(src_path: str, dest_folder="data/raw"):
    """
    Copy any uploaded file into /data/raw.
    Returns the absolute destination path.
    """
    dest_folder = ensure_dir(dest_folder)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_folder, filename)

    shutil.copy2(src_path, dest_path)
    print(f"[STORE] Saved raw file → {dest_path}")
    return dest_path


# -------------------------------------------------------------------
# Save Extracted TEXT
# -------------------------------------------------------------------
def save_text(claim_id: str, text: str, folder="data/processed/texts"):
    folder = ensure_dir(folder)

    safe_id = str(claim_id).replace("/", "_")
    path = os.path.join(folder, f"{safe_id}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")

    print(f"[STORE] Saved extracted text → {path}")
    return path


# -------------------------------------------------------------------
# Save DataFrame (CSV or Parquet)
# -------------------------------------------------------------------
def save_df(df: pd.DataFrame, path: str):
    abs_path = _abs(path)
    ensure_dir(os.path.dirname(abs_path))

    if abs_path.endswith(".csv"):
        df.to_csv(abs_path, index=False)
    elif abs_path.endswith(".parquet"):
        df.to_parquet(abs_path, index=False)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet")

    print(f"[STORE] Saved DataFrame → {abs_path}")
    return abs_path


def load_df(path: str):
    abs_path = _abs(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    if abs_path.endswith(".csv"):
        return pd.read_csv(abs_path)
    elif abs_path.endswith(".parquet"):
        return pd.read_parquet(abs_path)
    else:
        raise ValueError("Unsupported DataFrame format")


# -------------------------------------------------------------------
# Save / Load Model Object
# -------------------------------------------------------------------
def save_model_obj(model, path: str):
    abs_path = _abs(path)
    ensure_dir(os.path.dirname(abs_path))
    joblib.dump(model, abs_path)
    print(f"[STORE] Saved model → {abs_path}")
    return abs_path


def load_model_obj(path: str):
    abs_path = _abs(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model not found: {abs_path}")
    print(f"[STORE] Loaded model ← {abs_path}")
    return joblib.load(abs_path)


# -------------------------------------------------------------------
# Append HITL Label
# -------------------------------------------------------------------
def append_label(label_dict: dict, path="data/labels/labels.csv"):
    abs_path = _abs(path)
    ensure_dir(os.path.dirname(abs_path))

    df_new = pd.DataFrame([label_dict])

    if os.path.exists(abs_path):
        try:
            df_old = pd.read_csv(abs_path)
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_out = df_new  # fallback if corrupted
    else:
        df_out = df_new

    df_out.to_csv(abs_path, index=False)
    print(f"[STORE] Appended label → {abs_path}")
    return abs_path


# -------------------------------------------------------------------
# Manual Test Block
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Testing storage/store.py ===")

    # 1. Save raw file
    # (Requires a real local file path. Uncomment during real execution.)
    # save_raw_file("sample.pdf")

    # 2. Text save/load
    save_text("TEST_123", "Sample extracted claim text.")

    # 3. DF save/load
    df = pd.DataFrame({"claim_id": ["C1", "C2"], "amount": [1000, 2000]})
    save_df(df, "data/test/claims.csv")
    loaded = load_df("data/test/claims.csv")
    print(loaded)

    # 4. Model save/load
    dummy = {"a": 1}
    save_model_obj(dummy, "models/dummy_model.joblib")
    print(load_model_obj("models/dummy_model.joblib"))

    # 5. Append label
    append_label({
        "claim_id": "TEST_123",
        "label": 1,
        "reviewer_id": "rev001",
        "notes": "suspicious",
        "timestamp": "2025-01-01T12:00:00"
    })
