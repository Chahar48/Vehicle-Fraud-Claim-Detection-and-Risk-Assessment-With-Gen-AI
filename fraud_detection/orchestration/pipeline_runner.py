"""
pipeline_runner.py
------------------
Orchestrates the PoC pipeline for one claim or a batch:
ingest -> preprocess -> features -> models -> decision -> explain -> save -> (HITL)

Designed to be imported & used by the API server (run_api.py).
Uses FD_PROJECT_ROOT environment variable first; falls back to repo-relative paths.

Key guarantees:
- Consistent use of fraud_detection.store for saving artifacts
- Conservative fallback behavior: missing scoring modules -> prefer manual_review
- When action == "manual_review" we enqueue to HITL and store queue_id in result
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

# logger
from fraud_detection.logging.logger import get_logger

logger = get_logger(__name__)

# Project root resolution (env-first)
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Results folder under project root
RESULTS_FOLDER = PROJECT_ROOT / "data" / "results"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Defensive imports (modules may or may not be implemented)
def _safe_import(name: str):
    try:
        module = __import__(name, fromlist=["*"])
        logger.debug("Imported module %s", name)
        return module
    except Exception as e:
        logger.warning("Import failed for %s: %s", name, e)
        return None

# prefer explicit imports where available
file_router = _safe_import("fraud_detection.ingestion.file_router")
# Preprocessing components
field_normalizer = _safe_import("fraud_detection.preprocessing.field_normalizer")
text_cleaner = _safe_import("fraud_detection.preprocessing.text_cleaner")
schema_validator = _safe_import("fraud_detection.preprocessing.schema_validator")

# feature modules
feature_builder = _safe_import("fraud_detection.features.feature_builder")
similarity_index = _safe_import("fraud_detection.features.similarity_index")

# generative ai
embedder = _safe_import("fraud_detection.generative_ai.embedder")
explain_generator = _safe_import("fraud_detection.generative_ai.explanation_generator")

# models
anomaly_detector = _safe_import("fraud_detection.models.anomaly_detector")
fraud_classifier = _safe_import("fraud_detection.models.fraud_classifier")
model_utils = _safe_import("fraud_detection.models.model_utils")

# enrichment
external_lookup = _safe_import("fraud_detection.enrichment.external_lookup")

# decision engine
scoring = _safe_import("fraud_detection.decision_engine.scoring")
rules = _safe_import("fraud_detection.decision_engine.rules")

# storage
store = _safe_import("fraud_detection.storage.store")

# HITL
review_queue = _safe_import("fraud_detection.hitl.review_queue")
HITL_AVAILABLE = review_queue is not None

logger.info("Pipeline runner initialized. PROJECT_ROOT=%s HITL_AVAILABLE=%s", PROJECT_ROOT, HITL_AVAILABLE)

# ---------------------------
# Helpers
# ---------------------------
def _now_iso():
    return datetime.utcnow().isoformat()

def _abs_rel(path: str) -> str:
    """
    Convert project-relative path (starting without /) to absolute on disk.
    Used only for direct file writes when needed; prefer store.save_df/text instead.
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / path).resolve())

def _safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

def _first_or_scalar(x):
    """
    Accept possibly iterable or scalar. Return scalar float or 0.0.
    """
    try:
        if x is None:
            return 0.0
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            if len(x) == 0:
                return 0.0
            return float(x[0])
        return float(x)
    except Exception:
        return 0.0

def _save_result_json(claim_id: str, result: Dict[str, Any]) -> str:
    path = RESULTS_FOLDER / f"result_{claim_id}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved result JSON for %s -> %s", claim_id, path)
    except Exception:
        logger.exception("Failed to save result JSON for %s", claim_id)
        raise
    return str(path)

def _append_result_summary(result: Dict[str, Any], csv_rel: str = "data/results/results_summary.csv") -> str:
    # Use store.save_df when available; otherwise write directly to PROJECT_ROOT
    csv_path = csv_rel
    rec = {
        "claim_id": result.get("claim_id"),
        "final_score": result.get("final_score"),
        "action": result.get("action"),
        "fraud_prob": result.get("fraud_prob"),
        "anomaly_score": result.get("anomaly_score"),
        "similarity_score": result.get("similarity_score"),
        "blacklist_flag": _safe_get(result.get("enrichment", {}), "blacklist_flag", 0),
        "timestamp": _now_iso()
    }
    df_new = pd.DataFrame([rec])
    try:
        if store is not None and hasattr(store, "save_df"):
            # store.save_df expects a project-relative path like "data/.../file.csv"
            # use csv_rel as-is
            abs_csv = _abs_rel(csv_rel)
            # if file exists, load and concat then save via store.save_df
            if os.path.exists(abs_csv):
                df_exist = pd.read_csv(abs_csv)
                df_out = pd.concat([df_exist, df_new], ignore_index=True)
            else:
                df_out = df_new
            store.save_df(df_out, csv_rel)
            logger.info("Appended result summary via store.save_df -> %s", csv_rel)
            return str(Path(PROJECT_ROOT) / csv_rel)
        else:
            # fallback: write directly to resolved path
            abs_csv = _abs_rel(csv_rel)
            if os.path.exists(abs_csv):
                df_exist = pd.read_csv(abs_csv)
                df_out = pd.concat([df_exist, df_new], ignore_index=True)
            else:
                df_out = df_new
            os.makedirs(os.path.dirname(abs_csv), exist_ok=True)
            df_out.to_csv(abs_csv, index=False)
            logger.warning("storage.store unavailable, saved results_summary directly to %s", abs_csv)
            return abs_csv
    except Exception:
        logger.exception("Failed to append result summary CSV")
        raise

def _ensure_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

# ---------------------------
# Main pipeline: single claim
# ---------------------------
def run_single_claim(claim_record: Dict[str, Any], attachments: Optional[List[str]] = None, history_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    claim_id = str(claim_record.get("claim_id") or f"noid_{int(datetime.utcnow().timestamp())}")
    logger.info("Running pipeline for claim_id=%s", claim_id)

    result: Dict[str, Any] = {
        "claim_id": claim_id,
        "input": claim_record,
        "steps": {},
        "errors": []
    }

    # 1) Normalization (field_normalizer.normalize_claims_df)
    norm_row: Dict[str, Any] = {}
    try:
        if field_normalizer is None or not hasattr(field_normalizer, "normalize_claims_df"):
            raise RuntimeError("field_normalizer.normalize_claims_df not available")
        df_norm = field_normalizer.normalize_claims_df(pd.DataFrame([claim_record]))
        norm_row = df_norm.iloc[0].to_dict()
        # ensure missing flags present
        norm_row.setdefault("missing_amount_flag", int(norm_row.get("claim_amount") in [None, -1]))
        norm_row.setdefault("missing_info_flag", int(not bool(norm_row.get("phone")) or not bool(norm_row.get("garage_id")) or not bool(norm_row.get("vin"))))
        result["normalized"] = norm_row
        result["steps"]["normalize"] = "ok"
    except Exception as e:
        logger.exception("Normalization failed: %s", e)
        result["errors"].append(f"normalize_error:{e}")
        # fallback to raw claim values
        norm_row = claim_record.copy()
        norm_row.setdefault("description", claim_record.get("description", ""))
        norm_row.setdefault("claim_amount", claim_record.get("claim_amount"))
        norm_row.setdefault("policy_sum_insured", claim_record.get("policy_sum_insured"))
        norm_row.setdefault("missing_info_flag", int(not bool(claim_record.get("phone")) or not bool(claim_record.get("garage_id")) or not bool(claim_record.get("vin"))))
        result["normalized"] = norm_row
        result["steps"]["normalize"] = "fallback"

    # 2) Process attachments (route + extract text)
    attachments = _ensure_list(attachments)
    extracted_texts: List[str] = []
    try:
        for p in attachments:
            routed = p
            try:
                if file_router is not None and hasattr(file_router, "route_file"):
                    routed = file_router.route_file(p)
            except Exception:
                logger.warning("file_router.route_file failed for %s; using original path", p)
                routed = p

            ext = Path(routed).suffix.lower()
            try:
                if ext == ".pdf":
                    # pdf extractor may be in extraction package; try known names
                    pdf_mod = _safe_import("fraud_detection.extraction.ocr_extractor")
                    if pdf_mod and hasattr(pdf_mod, "extract_text_from_pdf"):
                        txt, summary = pdf_mod.extract_text_from_pdf(routed)
                        extracted_texts.append(txt or "")
                    else:
                        logger.debug("PDF extractor not available for %s", routed)
                elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
                    img_mod = _safe_import("fraud_detection.extraction.ocr_extractor")
                    if img_mod and hasattr(img_mod, "extract_text_from_image"):
                        txt = img_mod.extract_text_from_image(routed)
                        extracted_texts.append(txt or "")
                    else:
                        logger.debug("Image extractor not available for %s", routed)
                else:
                    logger.debug("Unsupported attachment extension %s for %s", ext, routed)
            except Exception as ex:
                logger.exception("Attachment extraction failed for %s: %s", routed, ex)
        result["steps"]["attachments"] = f"extracted_{len(extracted_texts)}"
    except Exception as e:
        logger.exception("Attachment processing error: %s", e)
        result["errors"].append(f"attachments_error:{e}")
        result["steps"]["attachments"] = "error"

    # 3) Assemble full_text
    try:
        desc = str(norm_row.get("description", "") or "")
        full_text = desc
        if extracted_texts:
            full_text = desc + "\n\n" + "\n\n".join(extracted_texts)
        # Save extracted text via store if available
        try:
            if store is not None and hasattr(store, "save_text"):
                store.save_text(claim_id, full_text)
        except Exception:
            logger.exception("Failed to save extracted text via store")
        result["full_text"] = full_text[:10000]
        result["steps"]["text"] = "ok"
    except Exception as e:
        logger.exception("Text assembly failed: %s", e)
        result["errors"].append(f"text_error:{e}")
        result["full_text"] = norm_row.get("description", "")
        result["steps"]["text"] = "error"

    # 4) Numeric features
    features_df = None
    feature_row: Dict[str, Any] = {}
    try:
        if feature_builder is None or not hasattr(feature_builder, "build_numeric_features"):
            raise RuntimeError("feature_builder.build_numeric_features not available")
        features_df = feature_builder.build_numeric_features(pd.DataFrame([norm_row]), history_df)
        feature_row = features_df.iloc[0].to_dict()
        result["features"] = feature_row
        result["steps"]["numeric_features"] = "ok"
    except Exception as e:
        logger.exception("Numeric features build failed: %s", e)
        result["errors"].append(f"numeric_features_error:{e}")
        result["steps"]["numeric_features"] = "error"

    # 5) Embedding + similarity
    similarity_score = 0.0
    emb_vector = None
    try:
        if embedder is not None and hasattr(embedder, "embed_text"):
            try:
                emb_vector = embedder.embed_text(full_text)
            except Exception as e:
                logger.exception("embedder.embed_text failed: %s", e)
                emb_vector = None
        # similarity: use similarity_index if available, else try feature-level quick similarity
        if emb_vector is not None and similarity_index is not None and hasattr(similarity_index, "load_embeddings"):
            try:
                emb_path = "data/processed/embeddings.npy"
                abs_emb = _abs_rel(emb_path)
                if os.path.exists(abs_emb):
                    hist_emb = similarity_index.load_embeddings(emb_path)
                    index = similarity_index.build_faiss_index(hist_emb)
                    dists, idxs = similarity_index.similarity_search(index, emb_vector, k=1)
                    # convert L2 distance -> similarity in range 0..1 (simple mapping)
                    similarity_score = float(np.clip(1.0 / (1.0 + float(dists[0])), 0.0, 1.0))
                else:
                    # fallback to quick_similarity if exists
                    if similarity_index is not None and hasattr(similarity_index, "quick_similarity"):
                        similarity_score = float(similarity_index.quick_similarity(full_text))
            except Exception as e:
                logger.exception("Similarity calculation failed: %s", e)
                similarity_score = 0.0
        else:
            # fallback
            if similarity_index is not None and hasattr(similarity_index, "quick_similarity"):
                similarity_score = float(similarity_index.quick_similarity(full_text))
            else:
                similarity_score = 0.0
        similarity_score = float(np.clip(similarity_score, 0.0, 1.0))
        result["similarity_score"] = similarity_score
        result["steps"]["text_features"] = "ok"
    except Exception as e:
        logger.exception("Embedding/similarity error: %s", e)
        result["errors"].append(f"similarity_error:{e}")
        result["similarity_score"] = 0.0
        result["steps"]["text_features"] = "error"

    # 6) Enrichment (blacklist-only simplified)
    enrichment = {}
    try:
        if external_lookup is not None and hasattr(external_lookup, "check_blacklist"):
            bl_flag, bl_reasons = external_lookup.check_blacklist(
                customer_id=norm_row.get("customer_id"),
                phone=norm_row.get("phone"),
                garage_id=norm_row.get("garage_id")
            )
            enrichment = {"blacklist_flag": int(bl_flag), "blacklist_reasons": bl_reasons}
        else:
            enrichment = {"blacklist_flag": 0, "blacklist_reasons": []}
        result["enrichment"] = enrichment
        result["steps"]["enrichment"] = "ok"
    except Exception as e:
        logger.exception("Enrichment failed: %s", e)
        result["errors"].append(f"enrichment_error:{e}")
        result["enrichment"] = {"blacklist_flag": 0}
        result["steps"]["enrichment"] = "error"

    # 7) Anomaly scoring
    anomaly_score = 0.0
    try:
        if anomaly_detector is not None and hasattr(anomaly_detector, "predict_anomaly_score") and model_utils is not None:
            # model_utils.get_numeric_matrix expects a DataFrame of numeric features
            X_num = None
            try:
                if features_df is not None:
                    X_num = model_utils.get_numeric_matrix(features_df)
                else:
                    X_num = model_utils.get_numeric_matrix(pd.DataFrame([feature_row]))
            except Exception:
                # fallback to numeric vector if get_numeric_matrix unavailable
                X_num = None
            raw_anom = anomaly_detector.predict_anomaly_score(X_num if X_num is not None else feature_row)
            anomaly_score = float(_first_or_scalar(raw_anom))
        else:
            anomaly_score = 0.0
            logger.info("Anomaly detector not available; anomaly_score=0.0")
        result["anomaly_score"] = anomaly_score
        result["steps"]["anomaly"] = "ok"
    except Exception as e:
        logger.exception("Anomaly scoring error: %s", e)
        result["errors"].append(f"anomaly_error:{e}")
        # heuristic fallback
        try:
            anomaly_score = float(feature_row.get("amount_ratio") or 0.0)
        except Exception:
            anomaly_score = 0.0
        result["anomaly_score"] = anomaly_score
        result["steps"]["anomaly"] = "fallback"

    # 8) Supervised fraud classifier
    fraud_prob = 0.0
    try:
        if fraud_classifier is not None and hasattr(fraud_classifier, "predict_proba") and model_utils is not None:
            X_num = None
            try:
                if features_df is not None:
                    X_num = model_utils.get_numeric_matrix(features_df)
                else:
                    X_num = model_utils.get_numeric_matrix(pd.DataFrame([feature_row]))
            except Exception:
                X_num = None
            raw_prob = fraud_classifier.predict_proba(X_num if X_num is not None else feature_row)
            fraud_prob = float(_first_or_scalar(raw_prob))
        else:
            fraud_prob = 0.0
            logger.info("fraud_classifier not available; fraud_prob=0.0")
        result["fraud_prob"] = fraud_prob
        result["steps"]["fraud_classifier"] = "ok"
    except Exception as e:
        logger.exception("Fraud classifier error: %s", e)
        result["errors"].append(f"fraud_classifier_error:{e}")
        result["fraud_prob"] = 0.0
        result["steps"]["fraud_classifier"] = "error"

    # 9) Business rules
    rule_flags = {}
    try:
        rule_ctx = {
            "policy_end_date": norm_row.get("policy_end_date"),
            "incident_date": norm_row.get("incident_date"),
            "policy_id_claim": norm_row.get("policy_id"),
            "policy_id_record": claim_record.get("policy_id_record"),
            "claim_amount": norm_row.get("claim_amount"),
            "median_amount_for_policy": norm_row.get("policy_sum_insured")
        }
        if rules is not None and hasattr(rules, "extract_rule_flags"):
            rule_flags = rules.extract_rule_flags(rule_ctx)
        else:
            rule_flags = {}
        result["rule_flags"] = rule_flags
        result["steps"]["rules"] = "ok"
    except Exception as e:
        logger.exception("Rules processing failed: %s", e)
        result["errors"].append(f"rules_error:{e}")
        result["steps"]["rules"] = "error"

    # 10) Scoring (compose context)
    try:
        missing_info_flag = int(norm_row.get("missing_info_flag", 0))
        scoring_context = {
            "fraud_prob": fraud_prob,
            "anomaly_score": anomaly_score,
            "similarity_score": similarity_score,
            "blacklist_flag": int(enrichment.get("blacklist_flag", 0)),
            "missing_info_flag": missing_info_flag,
            "normalized": norm_row,
            "input": claim_record
        }
        if scoring is None or not hasattr(scoring, "compute_final_score"):
            # conservative fallback: require manual review if scoring not available
            logger.warning("Scoring module unavailable: forcing manual_review for safety")
            final_decision = {"final_score": None, "action": "manual_review", "breakdown": {}}
        else:
            final_decision = scoring.compute_final_score(scoring_context)
        result["final_score"] = final_decision.get("final_score")
        result["action"] = final_decision.get("action")
        result["breakdown"] = final_decision.get("breakdown", {})
        result["steps"]["scoring"] = "ok"
        logger.info("Scoring result: action=%s score=%s", result["action"], result["final_score"])
    except Exception as e:
        logger.exception("Scoring error: %s", e)
        result["errors"].append(f"scoring_error:{e}")
        result["final_score"] = None
        result["action"] = "manual_review"
        result["breakdown"] = {}
        result["steps"]["scoring"] = "error"

    # 11) Explanation generation
    try:
        explain_payload = {
            "final_score": result.get("final_score"),
            "action": result.get("action"),
            "breakdown": result.get("breakdown"),
            "claim_id": claim_id,
            "brief": (result.get("full_text") or "")[:500]
        }
        if explain_generator is not None and hasattr(explain_generator, "generate_text_explanation"):
            explanation = explain_generator.generate_text_explanation(explain_payload)
        else:
            # fallback human-friendly explanation generation
            bd = result.get("breakdown") or {}
            explanation = f"Action={result.get('action')}. Score={result.get('final_score')}. Rules: {bd.get('rule_flags', {})}"
        result["explanation"] = explanation
        result["steps"]["explain"] = "ok"
    except Exception as e:
        logger.exception("Explanation generation failed: %s", e)
        result["errors"].append(f"explain_error:{e}")
        result["explanation"] = ""
        result["steps"]["explain"] = "error"

    # 12) Save results (JSON + CSV summary)
    try:
        json_path = _save_result_json(claim_id, result)
        csv_path = _append_result_summary(result)
        result["saved_paths"] = {"json": json_path, "summary_csv": csv_path}
        result["steps"]["save"] = "ok"
    except Exception as e:
        logger.exception("Saving results failed: %s", e)
        result["errors"].append(f"save_error:{e}")
        result["steps"]["save"] = "error"

    # 13) HITL enqueue if manual_review
    try:
        if result.get("action") == "manual_review":
            if HITL_AVAILABLE:
                try:
                    qid = review_queue.enqueue_for_review(result)
                    result["steps"]["hitl_enqueue"] = "ok"
                    result["hitl_queue_id"] = qid
                    logger.info("Enqueued for HITL: claim_id=%s queue_id=%s", claim_id, qid)
                except Exception as e:
                    logger.exception("Failed to enqueue HITL: %s", e)
                    result["errors"].append(f"hitl_enqueue_error:{e}")
                    result["steps"]["hitl_enqueue"] = "error"
            else:
                # as a fallback, append to labels or a fallback file so the UI can surface it
                logger.info("HITL unavailable: will mark hitl_enqueue=skipped and save record for later")
                result["steps"]["hitl_enqueue"] = "skipped"
        else:
            result["steps"]["hitl_enqueue"] = "not_required"
    except Exception as e:
        logger.exception("HITL integration error: %s", e)
        result["errors"].append(f"hitl_integration_error:{e}")

    logger.info("Pipeline completed for claim_id=%s action=%s score=%s", claim_id, result.get("action"), result.get("final_score"))
    return result

# ---------------------------
# Batch runner
# ---------------------------
def run_batch(claims_df: pd.DataFrame, attachments_map: Optional[Dict[str, List[str]]] = None, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    attachments_map = attachments_map or {}
    results = []
    for _, row in claims_df.iterrows():
        claim = row.to_dict()
        att = attachments_map.get(str(claim.get("claim_id")), [])
        try:
            res = run_single_claim(claim, attachments=att, history_df=history_df)
            results.append({
                "claim_id": res.get("claim_id"),
                "final_score": res.get("final_score"),
                "action": res.get("action"),
                "fraud_prob": res.get("fraud_prob"),
                "anomaly_score": res.get("anomaly_score"),
                "similarity_score": res.get("similarity_score")
            })
        except Exception as e:
            logger.exception("Error processing claim %s: %s", claim.get("claim_id"), e)
            results.append({
                "claim_id": claim.get("claim_id"),
                "final_score": None,
                "action": "error"
            })
    return pd.DataFrame(results)

# ---------------------------
# Manual test
# ---------------------------
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    sample_claims = pd.DataFrame([{
        "claim_id": "C1001",
        "customer_id": "C001",
        "policy_id": "POL-123",
        "policy_id_record": "POL-123",
        "claim_amount": 12000,
        "policy_sum_insured": 50000,
        "incident_date": "2023-07-10",
        "description": "Rear bumper damage due to minor collision.",
        "phone": "9999999999",
        "garage_id": None,
        "vin": None,
        "city": "Pune"
    }, {
        "claim_id": "C1002",
        "customer_id": "C999",
        "policy_id": "POL-999",
        "policy_id_record": "POL-000",
        "claim_amount": 750000,
        "policy_sum_insured": 50000,
        "incident_date": "2023-08-01",
        "description": "Total loss - vehicle stolen, no police report filed.",
        "phone": "8888888888",
        "garage_id": "GARAGE-FAKE-123",
        "vin": "MH12AB1234",
        "city": "Mumbai"
    }])

    out = run_batch(sample_claims)
    logger.info("Batch results:\n%s", out.to_string(index=False))
