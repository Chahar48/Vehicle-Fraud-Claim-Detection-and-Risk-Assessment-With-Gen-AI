# fraud_detection/serving/api_server.py
"""
FastAPI API server for Fraud Detection (final architecture, stable).

Supports:
 - /health
 - /score  (JSON body)
 - /score_claim (multipart: claim_json + files)
 - /batch/score (CSV)
 - /hitl/* endpoints (review queue, next, submit)

All file saving uses ingestion/file_saver.py
All scoring uses orchestration/pipeline_runner.py
"""

import os
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Resolve ROOT
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = os.path.abspath(FD_PROJECT_ROOT)
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Logging
try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("api_server")


# ------------------------------------------------------
# Safe import helper
# ------------------------------------------------------
def _safe_import(module: str):
    try:
        mod = __import__(module, fromlist=["*"])
        return mod
    except Exception as e:
        logger.debug(f"Optional import failed for {module}: {e}")
        return None


# ------------------------------------------------------
# Import pipeline runner
# ------------------------------------------------------
_runner = _safe_import("fraud_detection.orchestration.pipeline_runner")
run_single_claim = getattr(_runner, "run_single_claim", None)
run_batch = getattr(_runner, "run_batch", None)

# ------------------------------------------------------
# Ingestion modules
# ------------------------------------------------------
_ing_api = _safe_import("fraud_detection.ingestion.api_ingest")
handle_api_payload = getattr(_ing_api, "handle_api_payload", None)

_ing_csv = _safe_import("fraud_detection.ingestion.csv_ingest")
load_claims_csv = getattr(_ing_csv, "load_claims_csv", None)

# File saving
_file_saver = _safe_import("fraud_detection.ingestion.file_saver")
save_file = getattr(_file_saver, "save_file", None)
save_files = getattr(_file_saver, "save_files", None)

# ------------------------------------------------------
# HITL Modules
# ------------------------------------------------------
_hitl = _safe_import("fraud_detection.hitl.review_queue")
enqueue_for_review = getattr(_hitl, "enqueue_for_review", None)
list_pending_reviews = getattr(_hitl, "list_pending_reviews", None)
get_next_review = getattr(_hitl, "get_next_review", None)
mark_review_completed = getattr(_hitl, "mark_review_completed", None)

_feedback = _safe_import("fraud_detection.hitl.feedback_processor")
feedback_submit = getattr(_feedback, "save_review_result", None)

# ------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------
class HitlSubmitIn(BaseModel):
    claim_id: str
    label: int
    reviewer_id: str
    notes: Optional[str] = ""
    queue_id: Optional[str] = None


# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI(title="Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)


# ------------------------------------------------------
# Health
# ------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "root": PROJECT_ROOT}


# ------------------------------------------------------
# Helper: parse claim JSON from request
# ------------------------------------------------------
async def _extract_claim(request: Request, claim_json_form_name="claim_json") -> Dict[str, Any]:

    # Try multipart field
    try:
        form = await request.form()
        if claim_json_form_name in form:
            raw = form.get(claim_json_form_name)
            return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        pass

    # Try JSON body
    try:
        body = await request.json()
        return body if isinstance(body, dict) else json.loads(body)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing JSON. Send JSON body or form-data with 'claim_json'."
        )


# ------------------------------------------------------
# /score and /score_claim
# ------------------------------------------------------
@app.post("/score")
@app.post("/score_claim")
async def score_claim(
    request: Request,
    claim_json: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Accept:
      - Raw JSON (POST /score)
      - Multipart with claim_json + files (POST /score_claim)
    """

    # Parse claim payload
    try:
        if claim_json:
            claim = json.loads(claim_json)
        else:
            claim = await _extract_claim(request)
    except Exception as e:
        raise HTTPException(400, f"Invalid claim payload: {e}")

    # Save files (uses ingestion.file_saver)
    saved_paths: List[str] = []
    if files and save_file:
        try:
            saved = save_files(files)
            saved_paths = [str(p) for p in saved]
        except Exception as e:
            logger.exception("Failed saving attachments")
            saved_paths = []

    # Run pipeline
    if not run_single_claim:
        raise HTTPException(503, "Pipeline not available on server.")

    try:
        result = run_single_claim(claim, attachments=saved_paths)
    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise HTTPException(500, f"Pipeline failed: {e}")

    # HITL enqueue if needed
    try:
        if result.get("action") == "manual_review" and enqueue_for_review:
            if "hitl_queue_id" not in result:
                qid = enqueue_for_review(result)
                result["hitl_queue_id"] = qid
    except Exception:
        logger.exception("HITL enqueue failed (ignored).")

    return JSONResponse(result)


# ------------------------------------------------------
# /batch/score (CSV → batch scoring)
# ------------------------------------------------------
@app.post("/batch/score")
async def batch_score(csv_file: UploadFile = File(...)):
    if csv_file.content_type not in ("text/csv", "application/vnd.ms-excel", "text/plain"):
        raise HTTPException(400, "Upload must be a CSV file.")

    # Save CSV to raw folder
    if not save_file:
        raise HTTPException(503, "file_saver not available.")

    try:
        tmp_path = save_file(csv_file)
    except Exception as e:
        raise HTTPException(500, f"CSV save failed: {e}")

    if not load_claims_csv or not run_batch:
        raise HTTPException(503, "Batch components unavailable.")

    # Load CSV → DataFrame
    try:
        df = load_claims_csv(str(tmp_path))
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {e}")

    # Run pipeline batch
    try:
        out_df = run_batch(df)
    except Exception as e:
        raise HTTPException(500, f"Batch pipeline failed: {e}")

    preview = out_df.head(10).to_dict(orient="records")
    return {
        "status": "ok",
        "rows": len(out_df),
        "preview": preview
    }


# ------------------------------------------------------
# HITL: pending
# ------------------------------------------------------
@app.get("/hitl/pending")
@app.get("/review_queue")
def hitl_pending(limit: int = 100):
    if not list_pending_reviews:
        raise HTTPException(404, "HITL module not available.")
    try:
        items = list_pending_reviews(limit=limit)
        return {"pending": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------------
# HITL: next
# ------------------------------------------------------
@app.get("/hitl/next")
def hitl_next():
    if not get_next_review:
        raise HTTPException(404, "HITL module not available.")
    try:
        item = get_next_review()
        return {"status": "ok", "item": item} if item else {"status": "empty"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ------------------------------------------------------
# HITL: submit review
# ------------------------------------------------------
@app.post("/hitl/submit")
@app.post("/submit_review")
def hitl_submit(payload: HitlSubmitIn):
    rec = {
        "claim_id": payload.claim_id,
        "label": payload.label,
        "reviewer_id": payload.reviewer_id,
        "notes": payload.notes or "",
    }

    # Prefer feedback_processor.save_review_result
    try:
        if feedback_submit:
            feedback_submit(
                payload.claim_id,
                payload.label,
                payload.reviewer_id,
                payload.notes or "",
                queue_id=payload.queue_id
            )
    except Exception:
        logger.exception("feedback_processor failed.")

    # Mark queue completed
    if mark_review_completed and payload.queue_id:
        try:
            mark_review_completed(payload.queue_id, payload.reviewer_id, {"label": payload.label})
        except Exception:
            logger.exception("mark_review_completed failed.")

    return {"status": "ok", "record": rec}
