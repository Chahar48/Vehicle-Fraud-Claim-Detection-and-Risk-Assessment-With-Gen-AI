"""
fraud_detection.serving.api_server
----------------------------------
FastAPI server exposing endpoints for:
 - /health
 - /score (single claim JSON or multipart with files)
 - /batch/score (upload CSV of claims)
 - /hitl/pending (list)
 - /hitl/next (claim assignment)
 - /hitl/submit (submit reviewer label)

This server uses the pipeline_runner.run_single_claim() to score claims,
and integrates with storage.store, ingestion, and hitl modules when available.

Environment:
 - FD_PROJECT_ROOT (optional) sets project root paths
 - .env is supported (see README). If you add .env in project root, use python-dotenv.load_dotenv()
"""

import os
import json
import logging
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# optional: load .env if present at project root
try:
    from dotenv import load_dotenv
    # Load .env placed in PROJECT_ROOT (if FD_PROJECT_ROOT set use it, else repo root)
    FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
    if FD_PROJECT_ROOT:
        load_dotenv(os.path.join(FD_PROJECT_ROOT, ".env"))
    else:
        load_dotenv()
except Exception:
    pass

# Project root fallback
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = FD_PROJECT_ROOT
else:
    # two levels up from this file -> project root `your_project_root/`
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Logging helper (use your project's logger)
try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fraud_api")

# imports for pipeline + helpers (defensive)
try:
    from fraud_detection.orchestration.pipeline_runner import run_single_claim, run_batch
except Exception:
    run_single_claim = None
    run_batch = None
    logger.warning("pipeline_runner not available; scoring endpoints will fail until implemented.")

try:
    from fraud_detection.ingestion.api_ingest import handle_api_payload, validate_incoming_payload
except Exception:
    handle_api_payload = None
    validate_incoming_payload = None
    logger.info("api_ingest not available; files will be saved via storage.store/file_saver if present.")

try:
    from fraud_detection.ingestion.csv_ingest import ingest_file as ingest_csv_file, load_claims_csv
except Exception:
    ingest_csv_file = None
    load_claims_csv = None

try:
    from fraud_detection.storage.store import save_raw_file, save_text, append_label, save_df
except Exception:
    save_raw_file = None
    save_text = None
    append_label = None
    save_df = None

try:
    from fraud_detection.hitl.review_queue import enqueue_for_review, list_pending_reviews, get_next_review, mark_review_completed
except Exception:
    enqueue_for_review = None
    list_pending_reviews = None
    get_next_review = None
    mark_review_completed = None

# pydantic request model for JSON claim input
class ClaimIn(BaseModel):
    claim_id: Optional[str] = None
    customer_id: Optional[str] = None
    policy_id: Optional[str] = None
    policy_id_record: Optional[str] = None
    claim_amount: Optional[float] = None
    policy_sum_insured: Optional[float] = None
    incident_date: Optional[str] = None
    description: Optional[str] = None
    phone: Optional[str] = None
    garage_id: Optional[str] = None

app = FastAPI(
    title="Fraud Detection POC API",
    version="0.1",
    description="Scoring API for Fraud Detection & Risk Assessment PoC"
)

# Allow CORS for demo (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "project_root": PROJECT_ROOT}

# ---------------------------------------------------------------------
# Score single claim (JSON body) OR multipart with files.
# - If JSON-only: provide claim fields as JSON.
# - If files included: pass files and JSON payload string in form field 'payload'
# ---------------------------------------------------------------------
@app.post("/score")
async def score_claim(
    request: Request,
    payload: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """
    POST /score
    - JSON-only: send JSON with claim fields
    - multipart form: 'payload' = JSON string, 'files' = list of files
    """
    # Parse payload (either from form or raw JSON body)
    claim = None
    try:
        if payload:
            claim_json = payload
        else:
            # try to read json body
            try:
                claim_json = await request.json()
                # if request.json() returned dict, convert to json string
                if isinstance(claim_json, dict):
                    claim_dict = claim_json
                else:
                    claim_dict = json.loads(claim_json)
            except Exception:
                claim_dict = None

        # If form payload provided as JSON string
        if payload:
            try:
                claim_dict = json.loads(payload)
            except Exception as e:
                logger.exception("Failed to parse payload form JSON: %s", e)
                raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

        if claim_dict is None:
            raise HTTPException(status_code=400, detail="No JSON payload provided.")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Payload parse error: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to parse payload: {e}")

    # Validate payload basic required fields if helper exists
    if validate_incoming_payload:
        ok, missing = validate_incoming_payload(claim_dict)
        if not ok:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")

    # Save attachments (if any)
    saved_paths = []
    if files:
        for f in files:
            try:
                # Save to temporary raw folder: use storage.save_raw_file if available
                tmp_path = os.path.join(PROJECT_ROOT, "data", "raw", f.filename)
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                with open(tmp_path, "wb") as fh:
                    fh.write(await f.read())
                # if centralized store available, use it
                if save_raw_file:
                    final = save_raw_file(tmp_path, dest_folder=os.path.join(PROJECT_ROOT, "data", "raw"))
                else:
                    final = tmp_path
                saved_paths.append(final)
            except Exception as e:
                logger.exception("Failed saving uploaded file %s: %s", getattr(f, "filename", "<unknown>"), e)
                # continue saving other files

    # Run pipeline
    if run_single_claim is None:
        raise HTTPException(status_code=503, detail="Scoring pipeline not available on server.")

    try:
        result = run_single_claim(claim_dict, attachments=saved_paths)
    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")

    return JSONResponse(status_code=200, content=result)

# ---------------------------------------------------------------------
# Batch scoring via CSV upload (multipart/form-data)
# Accepts a CSV file. Returns a summary CSV saved path and a small JSON preview.
# ---------------------------------------------------------------------
@app.post("/batch/score")
async def batch_score(csv_file: UploadFile = File(...)):
    if csv_file.content_type not in ("text/csv", "application/vnd.ms-excel"):
        raise HTTPException(status_code=400, detail="Only CSV upload supported for batch scoring.")

    tmp_path = os.path.join(PROJECT_ROOT, "data", "raw", csv_file.filename)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(await csv_file.read())
    except Exception as e:
        logger.exception("Failed to save CSV file: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save CSV: {e}")

    if load_claims_csv is None or run_batch is None:
        raise HTTPException(status_code=503, detail="Batch ingestion or runner unavailable.")

    try:
        df = load_claims_csv(tmp_path)
    except Exception as e:
        logger.exception("CSV load error: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    try:
        batch_df = run_batch(df)
    except Exception as e:
        logger.exception("Batch pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")

    # save summary CSV if possible
    summary_path = os.path.join(PROJECT_ROOT, "data", "results", f"batch_summary_{os.path.basename(tmp_path)}")
    try:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        if save_df:
            save_df(batch_df, summary_path)
        else:
            batch_df.to_csv(summary_path, index=False)
    except Exception:
        logger.exception("Failed to save batch summary, continuing.")

    return {"status": "ok", "rows_processed": len(batch_df), "summary_path": summary_path, "preview": batch_df.head(5).to_dict(orient="records")}

# ---------------------------------------------------------------------
# HITL endpoints: list pending, next, and submit
# ---------------------------------------------------------------------
@app.get("/hitl/pending")
def hitl_pending(limit: int = 100):
    if list_pending_reviews is None:
        raise HTTPException(status_code=404, detail="HITL module not available.")
    try:
        items = list_pending_reviews(limit=limit)
        return {"count": len(items), "items": items}
    except Exception as e:
        logger.exception("hitl_pending error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hitl/next")
def hitl_next(assign_to: Optional[str] = None):
    if get_next_review is None:
        raise HTTPException(status_code=404, detail="HITL module not available.")
    try:
        item = get_next_review(assign_to=assign_to)
        if item is None:
            return {"status": "empty"}
        return {"status": "ok", "item": item}
    except Exception as e:
        logger.exception("hitl_next error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

class HitlSubmitIn(BaseModel):
    claim_id: str
    label: int
    reviewer_id: str
    notes: Optional[str] = ""
    queue_id: Optional[str] = None

@app.post("/hitl/submit")
def hitl_submit(payload: HitlSubmitIn):
    if append_label is None:
        raise HTTPException(status_code=404, detail="HITL/labels storage not available.")
    try:
        # Save label via storage or append_label helper if available
        rec = {
            "claim_id": payload.claim_id,
            "label": int(payload.label),
            "reviewer_id": payload.reviewer_id,
            "notes": payload.notes or "",
            "timestamp": __import__("datetime").datetime.utcnow().isoformat()
        }
        if append_label:
            append_label(rec, path=os.path.join(PROJECT_ROOT, "data", "labels", "labels.csv"))
        # Mark queue as reviewed if available
        try:
            if mark_review_completed and payload.queue_id:
                mark_review_completed(payload.queue_id, payload.reviewer_id, result_summary={"label": payload.label, "notes": payload.notes})
        except Exception:
            logger.exception("Failed to mark review completed (non-fatal).")
        return {"status": "ok", "record": rec}
    except Exception as e:
        logger.exception("hitl_submit error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "fraud-detection-poc", "version": "0.1", "endpoints": ["/health", "/score", "/batch/score", "/hitl/pending", "/hitl/next", "/hitl/submit"]}
