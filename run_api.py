"""
run_api.py
----------
Project entrypoint for running the Fraud Claim Detection API.

Instead of typing long uvicorn commands, you can simply run:
    python run_api.py

Features:
- Loads environment variables (FD_PROJECT_ROOT, API_HOST, API_PORT)
- Starts FastAPI API defined in fraud_detection.serving.api_server
- Production-safe defaults
- Recruiter-friendly single-entry startup
"""

import os
import uvicorn
from pathlib import Path

# ------------------------------
# 1. Resolve FD_PROJECT_ROOT
# ------------------------------
DEFAULT_ROOT = Path(__file__).resolve().parent

FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if not FD_PROJECT_ROOT:
    # auto-set for convenience
    FD_PROJECT_ROOT = str(DEFAULT_ROOT)
    os.environ["FD_PROJECT_ROOT"] = FD_PROJECT_ROOT

print(f"[INFO] FD_PROJECT_ROOT = {FD_PROJECT_ROOT}")


# ------------------------------
# 2. API host/port configuration
# ------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))


# ------------------------------
# 3. Run the API server
# ------------------------------
if __name__ == "__main__":
    print("=============================================")
    print("🚀 Starting Fraud Detection API")
    print("📌 Host:", API_HOST)
    print("📌 Port:", API_PORT)
    print("📌 Root:", FD_PROJECT_ROOT)
    print("=============================================")

    # Load FastAPI app from serving/api_server.py
    uvicorn.run(
        "fraud_detection.serving.api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,           # Set True only in development
        workers=1,              # For POC, 1 worker is enough
        log_level="info"
    )
