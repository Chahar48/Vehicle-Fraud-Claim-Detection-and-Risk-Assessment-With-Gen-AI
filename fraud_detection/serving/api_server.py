# fraud_detection/serving/api_server.py
"""
FastAPI server for Fraud Detection POC (aligned with current architecture).

Exposes:
 - GET  /health
 - POST /score, /score_claim   (single claim JSON or multipart with files)
 - POST /batch/score           (CSV)
 - GET  /hitl/pending, /review_queue
 - GET  /hitl/next
 - POST /hitl/submit, /submit_review

This implementation is defensive: it uses pipeline_runner when available,
and falls back to helpful HTTP errors when modules are missing.
"""
import os
import json
import logging
from typing import List, Optional, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional .env loader
try:
    from dotenv import load_dotenv
    FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
    if FD_PROJECT_ROOT:
        load_dotenv(os.path.join(FD_PROJECT_ROOT, ".env"))
    else:
        load_dotenv()
except Exception:
    pass

# Resolve project root
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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fraud_api")

# Defensive imports - try to import modules that may or may not exist
def _try_import(name: str):
    try:
        mod = __import__(name, fromlist=["*"])
        logger.debug("Imported %s", name)
        return mod
    except Exception as e:
        logger.debug("Optional import failed: %s -> %s", name, e)
        return None

# pipeline runner
_pr = _try_import("fraud_detection.orchestration.pipeline_runner")
run_single_claim = getattr(_pr, "run_single_claim", None) if _pr is not None else None
run_batch = getattr(_pr, "run_batch", None) if _pr is not None else None

# ingestion helpers (optional)
_ing_api = _try_import("fraud_detection.ingestion.api_ingest")
handle_api_payload = getattr(_ing_api, "handle_api_payload", None) if _ing_api is not None else None
validate_incoming_payload = getattr(_ing_api, "validate_incoming_payload", None) if _ing_api is not None else None

_ing_csv = _try_import("fraud_detection.ingestion.csv_ingest")
load_claims_csv = getattr(_ing_csv, "load_claims_csv", None) if _ing_csv is not None else None

# storage (store module)
_store = _try_import("fraud_detection.storage.store")
# prefer whole module to call functions with proper signatures
store = _store

# HITL queue
_hitl = _try_import("fraud_detection.hitl.review_queue")
enqueue_for_review = getattr(_hitl, "enqueue_for_review", None) if _hitl is not None else None
list_pending_reviews = getattr(_hitl, "list_pending_reviews", None) if _hitl is not None else None
get_next_review = getattr(_hitl, "get_next_review", None) if _hitl is not None else None
mark_review_completed = getattr(_hitl, "mark_review_completed", None) if _hitl is not None else None

# backup feedback processor (labels/append)
_feedback = _try_import("fraud_detection.hitl.feedback_processor")
feedback_save = getattr(_feedback, "save_review_result", None) if _feedback is not None else None

# Pydantic models
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

class HitlSubmitIn(BaseModel):
    claim_id: str
    label: int
    reviewer_id: str
    notes: Optional[str] = ""
    queue_id: Optional[str] = None

app = FastAPI(title="Fraud Detection POC API", version="0.1")

# CORS for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "project_root": PROJECT_ROOT}

# Root (list endpoints)
@app.get("/")
def root():
    return {
        "service": "fraud-detection-poc",
        "version": "0.1",
        "endpoints": ["/health", "/score", "/score_claim", "/batch/score", "/hitl/pending", "/review_queue", "/hitl/next", "/hitl/submit", "/submit_review"]
    }

# ----------------------------
# Internal helper: save uploaded files to data/raw
# ----------------------------
def _save_uploaded_file(upload: UploadFile) -> str:
    """Save incoming UploadFile to data/raw and return path. Prefer store.save_raw_file if available."""
    target_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = os.path.join(target_dir, upload.filename)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(upload.file.read())
    except Exception:
        # fallback: async read
        try:
            import asyncio
            data = asyncio.get_event_loop().run_until_complete(upload.read())
            with open(tmp_path, "wb") as fh:
                fh.write(data)
        except Exception as e:
            logger.exception("Failed to write upload to %s: %s", tmp_path, e)
            raise
    # If store.save_raw_file exists and is callable, use to move/copy
    if store is not None and hasattr(store, "save_raw_file"):
        try:
            # store.save_raw_file(src_path, dest_folder)
            final = store.save_raw_file(tmp_path, dest_folder=os.path.join(PROJECT_ROOT, "data", "raw"))
            return final
        except Exception:
            logger.exception("store.save_raw_file failed; using tmp path %s", tmp_path)
            return tmp_path
    return tmp_path

# ----------------------------
# Score: accept JSON body OR multipart with claim_json Form field + files
# Provide two endpoints for compatibility: /score and /score_claim
# ----------------------------
async def _parse_claim_from_request(request: Request, claim_form_field: Optional[str] = "claim_json") -> dict:
    """
    Parse a claim dict from either:
     - request.json() (application/json)
     - form field named `claim_json` containing JSON string
    Returns dict or raises HTTPException(400).
    """
    # try form field first
    form = None
    try:
        form = await request.form()
    except Exception:
        form = None

    if form and claim_form_field in form:
        s = form.get(claim_form_field)
        if isinstance(s, dict):
            return s
        try:
            return json.loads(s)
        except Exception as e:
            logger.exception("Failed to parse claim_json form field: %s", e)
            raise HTTPException(status_code=400, detail=f"Invalid JSON in form field '{claim_form_field}': {e}")

    # try raw JSON body
    try:
        content = await request.json()
        if isinstance(content, dict):
            return content
        # maybe it's a JSON string
        return json.loads(content)
    except Exception:
        # no JSON found
        raise HTTPException(status_code=400, detail="No JSON payload provided (either send JSON body or use 'claim_json' form field).")

@app.post("/score")
@app.post("/score_claim")   # alias for older UI
async def score_claim(request: Request, files: Optional[List[UploadFile]] = File(None), claim_json: Optional[str] = Form(None)):
    """
    Accept either:
      - application/json body with claim dict
      - multipart/form-data with form field 'claim_json' (string) and files list
    The UI historically sends form field 'claim_json' and files under name 'attachments'.
    """
    # parse claim
    try:
        # If claim_json form field was provided, prefer it
        if claim_json:
            try:
                claim_dict = json.loads(claim_json) if isinstance(claim_json, str) else claim_json
            except Exception as e:
                logger.exception("Invalid claim_json form: %s", e)
                raise HTTPException(status_code=400, detail=f"Invalid claim_json: {e}")
        else:
            # parse from full request body
            claim_dict = await _parse_claim_from_request(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Claim parse failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to parse claim: {e}")

    # optional validation helper
    if validate_incoming_payload is not None:
        try:
            ok, missing = validate_incoming_payload(claim_dict)
            if not ok:
                raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")
        except HTTPException:
            raise
        except Exception:
            logger.exception("validate_incoming_payload failed; continuing without validation.")

    # save files if any
    saved_paths = []
    # NOTE: some clients may post files under field name 'attachments' or others.
    # FastAPI gives us 'files' arg; also we can inspect request.form() for upload fields
    if files:
        for up in files:
            try:
                saved = _save_uploaded_file(up)
                saved_paths.append(saved)
            except Exception as e:
                logger.exception("Failed to save uploaded file %s: %s", getattr(up, "filename", "<no-name>"), e)
                # continue with other files

    # run pipeline
    if run_single_claim is None:
        raise HTTPException(status_code=503, detail="Scoring pipeline not available on server (run_single_claim missing).")

    try:
        # run_single_claim is synchronous per your runner; call directly
        result = run_single_claim(claim_dict, attachments=saved_paths)
    except Exception as e:
        logger.exception("Pipeline execution failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")

    # If pipeline decided manual_review and we have enqueue helper, ensure it's enqueued (pipeline_runner may already enqueue)
    try:
        if result and isinstance(result, dict) and result.get("action") == "manual_review":
            # If pipeline didn't include hitl_queue_id, attempt enqueue now (idempotent when allow_duplicates handled)
            if enqueue_for_review is not None and not result.get("hitl_queue_id"):
                try:
                    qid = enqueue_for_review(result)
                    # attach queue id for UI convenience
                    if isinstance(result, dict):
                        result["hitl_queue_id"] = qid
                except Exception:
                    logger.exception("Failed to enqueue result into HITL queue (non-fatal).")
    except Exception:
        logger.exception("Post-pipeline HITL enqueue attempt failed (non-fatal).")

    return JSONResponse(status_code=200, content=result)

# ----------------------------
# Batch scoring (CSV)
# ----------------------------
@app.post("/batch/score")
async def batch_score(csv_file: UploadFile = File(...)):
    if csv_file.content_type not in ("text/csv", "application/vnd.ms-excel", "text/plain"):
        raise HTTPException(status_code=400, detail="Only CSV upload supported for batch scoring.")

    target_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = os.path.join(target_dir, csv_file.filename)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(await csv_file.read())
    except Exception as e:
        logger.exception("Failed to save CSV: %s", e)
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

    # save summary CSV
    summary_path = os.path.join(PROJECT_ROOT, "data", "results", f"batch_summary_{os.path.basename(tmp_path)}")
    try:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        if store is not None and hasattr(store, "save_df"):
            store.save_df(batch_df, summary_path)
        else:
            batch_df.to_csv(summary_path, index=False)
    except Exception:
        logger.exception("Failed to save batch summary (non-fatal).")

    return {"status": "ok", "rows_processed": len(batch_df), "summary_path": summary_path, "preview": batch_df.head(5).to_dict(orient="records")}

# ----------------------------
# HITL endpoints
# Provide both /hitl/* and alias endpoints /review_queue and /submit_review for UI compatibility
# ----------------------------
@app.get("/hitl/pending")
@app.get("/review_queue")   # alias older UI
def hitl_pending(limit: int = 100):
    if list_pending_reviews is None:
        raise HTTPException(status_code=404, detail="HITL module not available.")
    try:
        items = list_pending_reviews(limit=limit)
        # UI historically expected either a list or {"pending": [...]}. Provide both shapes.
        return {"count": len(items), "items": items, "pending": items}
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

@app.post("/hitl/submit")
@app.post("/submit_review")   # alias for older UI
def hitl_submit(payload: HitlSubmitIn):
    # Append label to labels CSV via store or feedback processor
    rec = {
        "claim_id": payload.claim_id,
        "label": int(payload.label),
        "reviewer_id": payload.reviewer_id,
        "notes": payload.notes or "",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
    }
    try:
        # prefer centralized store.append_label if present
        if store is not None and hasattr(store, "append_label"):
            try:
                store.append_label(rec, path=os.path.join(PROJECT_ROOT, "data", "labels", "labels.csv"))
            except Exception:
                logger.exception("store.append_label failed; falling back to feedback processor if available.")

        # fallback: feedback processor
        if feedback_save is not None:
            try:
                feedback_save(payload.claim_id, payload.label, payload.reviewer_id, payload.notes or "", queue_id=payload.queue_id)
            except Exception:
                logger.exception("feedback_processor.save_review_result failed (non-fatal).")

        # mark queue item reviewed
        if mark_review_completed is not None and payload.queue_id:
            try:
                mark_review_completed(payload.queue_id, payload.reviewer_id, result_summary={"label": payload.label, "notes": payload.notes})
            except Exception:
                logger.exception("mark_review_completed failed (non-fatal).")
    except Exception as e:
        logger.exception("hitl_submit error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok", "record": rec}
