# ui/reviewer_app.py
"""
Streamlit UI for Fraud Claim Detection (compatible with PHASE 1-17 architecture)

Place this file at: <repo>/ui/reviewer_app.py
Run: streamlit run ui/reviewer_app.py
"""

import os
import json
import requests
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

# ---------------------------------------------------------------------
# PROJECT ROOT (prefer FD_PROJECT_ROOT; fallback to repo root)
# ---------------------------------------------------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if not FD_PROJECT_ROOT:
    # assume ui is at <repo>/ui/reviewer_app.py -> project root is parent
    FD_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["FD_PROJECT_ROOT"] = FD_PROJECT_ROOT

RESULTS_FOLDER = os.path.join(FD_PROJECT_ROOT, "data", "results")
HITL_FOLDER = os.path.join(FD_PROJECT_ROOT, "data", "hitl")
LABELS_FOLDER = os.path.join(FD_PROJECT_ROOT, "data", "labels")

API_BASE = os.environ.get("FD_API_BASE", "http://localhost:8000")

# ---------------------------------------------------------------------
# Try to import local pipeline & hitl modules for fallback
# ---------------------------------------------------------------------
LOCAL_AVAILABLE = True
try:
    from fraud_detection.orchestration import pipeline_runner
    from fraud_detection.hitl import review_queue as local_queue
    from fraud_detection.hitl import feedback_processor as local_feedback
    from fraud_detection.storage import store as local_store
except Exception:
    LOCAL_AVAILABLE = False
    pipeline_runner = None
    local_queue = None
    local_feedback = None
    local_store = None

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_float(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


def api_ok():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


USE_API = api_ok()

# ---------------------------------------------------------------------
# API Wrappers
# ---------------------------------------------------------------------
def api_score(claim: dict, upload_files):
    files = []
    for fname, content, ctype in upload_files:
        files.append(("attachments", (fname, BytesIO(content), ctype)))
    try:
        resp = requests.post(
            f"{API_BASE}/score_claim",
            data={"claim_json": json.dumps(claim)},
            files=files,
            timeout=30,
        )
        return resp.json()
    except Exception as e:
        return {"error": f"API scoring failed: {e}"}


def api_submit_review(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/submit_review", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": f"API submit review failed: {e}"}


def api_pending():
    try:
        r = requests.get(f"{API_BASE}/review_queue?limit=200", timeout=5)
        return r.json().get("pending", [])
    except Exception:
        return []


# ---------------------------------------------------------------------
# UI CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Fraud Claim Detection UI", layout="wide")

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("System Status")
    st.write("Project root:", FD_PROJECT_ROOT)
    st.write("API base:", API_BASE)
    if USE_API:
        st.success("FastAPI reachable")
    else:
        if LOCAL_AVAILABLE:
            st.warning("API unreachable — using local pipeline fallback")
        else:
            st.error("API unreachable and local pipeline unavailable")

    st.markdown("---")
    reviewer_id = st.text_input("Reviewer ID", value="reviewer_001")
    st.caption("UTC Timestamp")
    st.write(datetime.utcnow().isoformat())

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_upload, tab_hitl, tab_results, tab_labels = st.tabs(
    ["Upload & Score", "Pending Reviews", "Saved Results", "Labeled Data"]
)

# =====================================================================
# TAB 1 — UPLOAD & SCORE
# =====================================================================
with tab_upload:
    st.header("Submit Claim for Scoring")

    with st.form("form_claim", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            claim_id = st.text_input("Claim ID")
            customer_id = st.text_input("Customer ID")
            policy_id = st.text_input("Policy ID (Claimed)")
            policy_id_record = st.text_input("Policy ID (Official)")
            claim_amount = safe_float(st.text_input("Claim Amount", value="0"))
            policy_sum_insured = safe_float(st.text_input("Policy Sum Insured", value="0"))

        with col2:
            incident_date = st.date_input("Incident Date")
            description = st.text_area("Description")
            phone = st.text_input("Phone")
            garage_id = st.text_input("Garage ID")
            upload_files_streamlit = st.file_uploader(
                "Upload PDF/Images (optional)", accept_multiple_files=True
            )

        submitted = st.form_submit_button("Submit for Scoring")

    if submitted:
        st.info("Submitting for scoring — please wait...")

        claim_payload = {
            "claim_id": claim_id,
            "customer_id": customer_id,
            "policy_id": policy_id,
            "policy_id_record": policy_id_record,
            "claim_amount": claim_amount,
            "policy_sum_insured": policy_sum_insured,
            "incident_date": incident_date.isoformat() if incident_date else None,
            "description": description,
            "phone": phone,
            "garage_id": garage_id,
        }

        files_for_api = []
        for f in upload_files_streamlit:
            try:
                content = f.read()
                files_for_api.append((f.name, content, f.type))
            except Exception as e:
                st.warning(f"Failed to read uploaded file {getattr(f, 'name', '')}: {e}")

        # Choose API or local pipeline
        result = None
        if USE_API:
            result = api_score(claim_payload, files_for_api)
        else:
            # local fallback
            if not LOCAL_AVAILABLE:
                result = {"error": "No scoring backend available (API down and local pipeline missing)."}
            else:
                # save attachments to data/raw and call pipeline_runner
                saved_paths = []
                raw_dir = os.path.join(FD_PROJECT_ROOT, "data", "raw")
                os.makedirs(raw_dir, exist_ok=True)
                for fname, content, ctype in files_for_api:
                    path = os.path.join(raw_dir, fname)
                    try:
                        with open(path, "wb") as fh:
                            fh.write(content)
                        saved_paths.append(path)
                    except Exception as e:
                        st.warning(f"Failed to save attachment {fname}: {e}")

                try:
                    result = pipeline_runner.run_single_claim(claim_payload, attachments=saved_paths)
                except Exception as e:
                    result = {"error": f"Local pipeline_runner failed: {e}"}

        # Display result robustly
        if not isinstance(result, dict):
            st.error("Unexpected pipeline response format (expected dict).")
        elif "error" in result:
            st.error(f"Pipeline error: {result.get('error')}")
            # show debug info if available
            if result.get("steps"):
                st.json(result.get("steps"))
            if result.get("errors"):
                st.json(result.get("errors"))
        else:
            # use .get to avoid KeyError
            action = result.get("action", "unknown")
            score = result.get("final_score", None)

            if action in [None, "error", "unknown"]:
                st.error("Pipeline did not produce a valid decision.")
                # show raw payload to debug
                st.subheader("Raw Pipeline Output (debug)")
                st.json(result)
            else:
                st.success(f"Decision: {action}  (score={score})")

            st.subheader("Explanation")
            st.info(result.get("explanation", ""))

            st.subheader("Breakdown")
            st.json(result.get("breakdown", {}))

            st.subheader("Normalized Fields")
            st.json(result.get("normalized", {}))

            st.subheader("Features")
            st.json(result.get("features", {}))

            st.subheader("Rule Flags")
            st.json(result.get("rule_flags", {}))

            # If manual review, ensure it is enqueued and display queue info
            try:
                if action == "manual_review":
                    # If the pipeline already returned a hitl_queue_id show it
                    qid = result.get("hitl_queue_id")
                    if qid:
                        st.warning(f"Claim is in HITL queue (queue_id={qid}).")
                    else:
                        # otherwise try to enqueue here (best-effort)
                        if LOCAL_AVAILABLE and local_queue is not None:
                            try:
                                qid = local_queue.enqueue_for_review(result, allow_duplicates=False)
                                st.warning(f"Enqueued locally for HITL (queue_id={qid}).")
                            except Exception as e:
                                st.error(f"Failed to enqueue locally: {e}")
                        elif USE_API:
                            # assume API enqueues automatically; display note
                            st.info("API mode: HITL should be managed by server (verify /review_queue endpoint).")
                        else:
                            st.info("HITL unavailable; please check review queue manually.")
            except Exception as e:
                st.error(f"HITL enqueue handling error: {e}")

# =====================================================================
# TAB 2 — PENDING HITL REVIEWS
# =====================================================================
with tab_hitl:
    st.header("Pending Human Reviews")
    pending = []
    if USE_API:
        pending = api_pending()
    else:
        if LOCAL_AVAILABLE and local_queue is not None:
            try:
                pending = local_queue.list_pending_reviews(limit=200)
            except Exception as e:
                st.error(f"Failed to load local HITL queue: {e}")
                pending = []

    if not pending:
        st.info("No pending reviews.")
    else:
        st.success(f"{len(pending)} items waiting for review")
        # build friendly options
        options = [
            f"{p.get('queue_id','<no-id>')} — {p.get('claim_id', '<no-claim>')}"
            for p in pending
        ]
        choice = st.selectbox("Select item", options)
        idx = options.index(choice)
        item = pending[idx]

        st.subheader(f"Queue ID: {item.get('queue_id')}")
        st.markdown(f"**Claim ID:** {item.get('claim_id')}")
        st.markdown(f"**Enqueued at:** {item.get('enqueued_at')}")

        payload = item.get("payload") or {}
        # payload may be a JSON string in older implementations
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}

        st.write("### AI Summary")
        st.json({
            "final_score": payload.get("final_score"),
            "action": payload.get("action"),
            "fraud_prob": payload.get("fraud_prob"),
            "anomaly_score": payload.get("anomaly_score"),
            "similarity_score": payload.get("similarity_score"),
        })

        st.write("### Explanation")
        st.info(payload.get("explanation", ""))

        st.write("### Normalized Fields")
        st.json(payload.get("normalized", {}))

        st.write("### Features")
        st.json(payload.get("features", {}))

        st.markdown("---")
        st.subheader("Submit Review")
        notes = st.text_area("Reviewer notes", value="")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Mark as FRAUD"):
                review_payload = {
                    "claim_id": item.get("claim_id"),
                    "label": 1,
                    "reviewer_id": reviewer_id,
                    "notes": notes,
                    "queue_id": item.get("queue_id"),
                }
                if USE_API:
                    resp = api_submit_review(review_payload)
                    st.json(resp)
                else:
                    try:
                        rec = local_feedback.save_review_result(**review_payload)
                        st.success("Saved review locally.")
                        st.json(rec)
                    except Exception as e:
                        st.error(f"Failed to save review locally: {e}")

        with col2:
            if st.button("Mark as NOT Fraud"):
                review_payload = {
                    "claim_id": item.get("claim_id"),
                    "label": 0,
                    "reviewer_id": reviewer_id,
                    "notes": notes,
                    "queue_id": item.get("queue_id"),
                }
                if USE_API:
                    resp = api_submit_review(review_payload)
                    st.json(resp)
                else:
                    try:
                        rec = local_feedback.save_review_result(**review_payload)
                        st.success("Saved review locally.")
                        st.json(rec)
                    except Exception as e:
                        st.error(f"Failed to save review locally: {e}")

# =====================================================================
# TAB 3 — SAVED RESULTS
# =====================================================================
with tab_results:
    st.header("Saved Results Browser")
    summary_path = os.path.join(FD_PROJECT_ROOT, "data", "results", "results_summary.csv")
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Failed to load results summary CSV: {e}")
    else:
        st.info("No results_summary.csv found.")

    st.markdown("---")
    claim_lookup = st.text_input("Enter Claim ID to load full result")
    if st.button("Load Result JSON"):
        if not claim_lookup:
            st.warning("Please enter a claim id")
        else:
            path = os.path.join(FD_PROJECT_ROOT, "data", "results", f"result_{claim_lookup}.json")
            if os.path.exists(path):
                try:
                    st.json(json.load(open(path, "r", encoding="utf-8")))
                except Exception as e:
                    st.error(f"Failed to read JSON result: {e}")
            else:
                st.error("Result JSON not found")

# =====================================================================
# TAB 4 — LABELS / RETRAIN
# =====================================================================
with tab_labels:
    st.header("Labeled Data")
    labels_path = os.path.join(FD_PROJECT_ROOT, "data", "labels", "labels.csv")
    if os.path.exists(labels_path):
        try:
            df_labels = pd.read_csv(labels_path)
            st.dataframe(df_labels)
        except Exception as e:
            st.error(f"Failed to load labels.csv: {e}")
    else:
        st.info("No labeled data found.")

    st.markdown("---")
    if st.button("Export merged dataset (for retraining)"):
        if LOCAL_AVAILABLE and local_feedback is not None:
            try:
                merged = local_feedback.export_for_retraining()
                if merged is None or merged.empty:
                    st.warning("No merged dataset available.")
                else:
                    csv_bytes = merged.to_csv(index=False).encode("utf-8")
                    st.download_button("Download merged CSV", csv_bytes, file_name="merged_dataset.csv")
            except Exception as e:
                st.error(f"Failed to export merged dataset: {e}")
        else:
            st.error("Local feedback/export_for_retraining not available.")
