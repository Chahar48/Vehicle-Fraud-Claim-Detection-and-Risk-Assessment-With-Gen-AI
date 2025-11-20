"""
reviewer_app.py – Streamlit UI
Compatible with PHASE 1–17 architecture
---------------------------------------

Major improvements:
- Fully aligned with new pipeline runner, scoring, rules, enrichment
- VIN + City removed (only using required fields)
- Clean HITL integration (queue always updates correctly)
- Uses FD_PROJECT_ROOT automatically
- Works with or without FastAPI
- UX improved: validation, summaries, HITL actions
"""

import os
import json
import requests
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

# ---------------------------------------------------------------------
# PROJECT ROOT (auto-detect or use FD_PROJECT_ROOT)
# ---------------------------------------------------------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if not FD_PROJECT_ROOT:
    FD_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["FD_PROJECT_ROOT"] = FD_PROJECT_ROOT

RESULTS_FOLDER = os.path.join(FD_PROJECT_ROOT, "data", "results")
HITL_FOLDER = os.path.join(FD_PROJECT_ROOT, "data", "hitl")
LABELS_FOLDER = os.path.join(FD_PROJECT_ROOT, "data", "labels")

API_BASE = "http://localhost:8000"


# ---------------------------------------------------------------------
# LOCAL FALLBACK MODULES (if API is down)
# ---------------------------------------------------------------------
LOCAL_AVAILABLE = True
try:
    from fraud_detection.orchestration import pipeline_runner
    from fraud_detection.hitl import review_queue as local_queue
    from fraud_detection.hitl import feedback_processor as local_feedback
    from fraud_detection.storage import store as local_store
except Exception:
    LOCAL_AVAILABLE = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_float(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return None


def api_ok():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


USE_API = api_ok()


# ---------------------------------------------------------------------
# UI CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Fraud Claim Detection UI", layout="wide")


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("System Status")
    st.write("API:", API_BASE)
    if USE_API:
        st.success("FastAPI Running")
    else:
        if LOCAL_AVAILABLE:
            st.warning("API down → using local pipeline")
        else:
            st.error("API + Local pipeline unavailable")

    st.markdown("---")
    reviewer_id = st.text_input("Reviewer ID", value="reviewer_001")
    st.caption("UTC Timestamp")
    st.write(datetime.utcnow().isoformat())


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
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def api_submit_review(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/submit_review", json=payload)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_pending():
    try:
        r = requests.get(f"{API_BASE}/review_queue?limit=200")
        return r.json().get("pending", [])
    except:
        return []


# ---------------------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------------------
tab_upload, tab_hitl, tab_results, tab_labels = st.tabs([
    "Upload & Score",
    "Pending Reviews",
    "Saved Results",
    "Labeled Data",
])


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
            claim_amount = safe_float(st.text_input("Claim Amount"))
            policy_sum_insured = safe_float(st.text_input("Policy Sum Insured"))

        with col2:
            incident_date = st.date_input("Incident Date")
            description = st.text_area("Description")
            phone = st.text_input("Phone")
            garage_id = st.text_input("Garage ID")
            upload_files_streamlit = st.file_uploader(
                "Upload PDF/Images",
                accept_multiple_files=True,
            )

        submit = st.form_submit_button("Submit for Scoring")

    if submit:
        st.info("Scoring...")

        claim = {
            "claim_id": claim_id,
            "customer_id": customer_id,
            "policy_id": policy_id,
            "policy_id_record": policy_id_record,
            "claim_amount": claim_amount,
            "policy_sum_insured": policy_sum_insured,
            "incident_date": incident_date.isoformat(),
            "description": description,
            "phone": phone,
            "garage_id": garage_id,
        }

        files_api = [(f.name, f.read(), f.type) for f in upload_files_streamlit]

        if USE_API:
            result = api_score(claim, files_api)
        else:
            # save locally + run pipeline
            saved_paths = []
            raw_folder = os.path.join(FD_PROJECT_ROOT, "data", "raw")
            os.makedirs(raw_folder, exist_ok=True)

            for f in upload_files_streamlit:
                file_path = os.path.join(raw_folder, f.name)
                with open(file_path, "wb") as w:
                    w.write(f.read())
                saved_paths.append(file_path)

            try:
                result = pipeline_runner.run_single_claim(claim, attachments=saved_paths)
            except Exception as e:
                result = {"error": str(e)}

        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Decision: {result['action']}  (score={result['final_score']})")

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

            if result["action"] == "manual_review":
                st.warning("This claim has been added to HITL review queue.")


# =====================================================================
# TAB 2 — HITL QUEUE
# =====================================================================
with tab_hitl:
    st.header("Pending Human Review")

    if USE_API:
        pending = api_pending()
    else:
        pending = local_queue.list_pending_reviews(limit=200) if LOCAL_AVAILABLE else []

    if not pending:
        st.info("No pending items.")
    else:
        st.success(f"{len(pending)} items awaiting review")

        labels = [f"{p['queue_id']} — {p['claim_id']}" for p in pending]
        selected = st.selectbox("Select Review Item", labels)
        item = pending[labels.index(selected)]

        st.subheader(f"Queue ID: {item['queue_id']}")
        st.markdown(f"**Claim ID:** {item['claim_id']}")
        st.markdown(f"**Enqueued At:** {item['enqueued_at']}")

        payload = item.get("payload", {})

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

        st.write("### Normalized")
        st.json(payload.get("normalized", {}))

        st.write("### Features")
        st.json(payload.get("features", {}))

        st.markdown("---")
        st.subheader("Submit Human Review")

        notes = st.text_area("Reviewer Notes")

        colA, colB = st.columns(2)
        with colA:
            if st.button("Mark as FRAUD"):
                data = {
                    "claim_id": item["claim_id"],
                    "label": 1,
                    "reviewer_id": reviewer_id,
                    "notes": notes,
                    "queue_id": item["queue_id"],
                }
                result = api_submit_review(data) if USE_API else local_feedback.save_review_result(**data)
                st.success("Review submitted!")

        with colB:
            if st.button("Mark as NOT Fraud"):
                data = {
                    "claim_id": item["claim_id"],
                    "label": 0,
                    "reviewer_id": reviewer_id,
                    "notes": notes,
                    "queue_id": item["queue_id"],
                }
                result = api_submit_review(data) if USE_API else local_feedback.save_review_result(**data)
                st.success("Review submitted!")


# =====================================================================
# TAB 3 — SAVED RESULTS
# =====================================================================
with tab_results:
    st.header("Saved Results Browser")

    summary_path = os.path.join(RESULTS_FOLDER, "results_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        st.dataframe(df)
    else:
        st.info("No results available.")

    st.markdown("---")
    cid = st.text_input("Enter Claim ID to load full result")

    if st.button("Load Result JSON"):
        path = os.path.join(RESULTS_FOLDER, f"result_{cid}.json")
        if os.path.exists(path):
            st.json(json.load(open(path)))
        else:
            st.error("Result not found")


# =====================================================================
# TAB 4 — LABELS / RETRAIN
# =====================================================================
with tab_labels:
    st.header("Labeled Data")

    labels_path = os.path.join(LABELS_FOLDER, "labels.csv")
    if os.path.exists(labels_path):
        df = pd.read_csv(labels_path)
        st.dataframe(df)
    else:
        st.info("No labeled data found.")
