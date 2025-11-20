# fraud_detection/orchestration/pipeline_runner.py
"""
Pipeline runner (robust, defensive).
Orchestrates:
  normalize -> attachments/extract -> features -> models -> decision -> explain -> save -> (HITL)

Adapted to current architecture where:
  fraud_detection/features/feature_builder.py
  fraud_detection/features/similarity_index.py
exist and expose functions used below.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

# Logging helper: simple fallback if project logger missing
try:
    from fraud_detection.logging.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("pipeline_runner")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

# Resolve project root (env-first)
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]  # go up to repo root

RESULTS_FOLDER = PROJECT_ROOT / "data" / "results"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Defensive import helper
# ---------------------------
def _safe_import(module_path: str):
    try:
        mod = __import__(module_path, fromlist=["*"])
        logger.debug("Imported %s", module_path)
        return mod
    except Exception as e:
        logger.debug("Optional import failed %s: %s", module_path, e)
        return None


# Preferred modules (may be None)
normalize_mod = _safe_import("fraud_detection.preprocessing.normalize_fields")
pdf_extractor_mod = _safe_import("fraud_detection.extraction.pdf_extractor")
ocr_extractor_mod = _safe_import("fraud_detection.extraction.ocr_extractor")

# === UPDATED feature modules ===
feature_builder_mod = _safe_import("fraud_detection.features.feature_builder")
similarity_mod = _safe_import("fraud_detection.features.similarity_index")

embedder_mod = _safe_import("fraud_detection.generative_ai.embedder")
explain_gen_mod = _safe_import("fraud_detection.generative_ai.explain_generator")
anomaly_mod = _safe_import("fraud_detection.models.anomaly_detector")
fraud_clf_mod = _safe_import("fraud_detection.models.fraud_classifier")
model_utils_mod = _safe_import("fraud_detection.models.model_utils")
external_lookup_mod = _safe_import("fraud_detection.enrichment.external_lookup")
scoring_mod = _safe_import("fraud_detection.decision_engine.scoring")
rules_mod = _safe_import("fraud_detection.decision_engine.rules")
store_mod = _safe_import("fraud_detection.storage.store")
review_queue_mod = _safe_import("fraud_detection.hitl.review_queue")
HITL_AVAILABLE = review_queue_mod is not None


# ---------------------------
# Helpers
# ---------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _abs_rel(p: str) -> str:
    pth = Path(p)
    if pth.is_absolute():
        return str(pth)
    return str((PROJECT_ROOT / p).resolve())


def _first_or_scalar(x):
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
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        logger.info("Saved result JSON -> %s", path)
    except Exception:
        logger.exception("Failed to save result JSON")
    return str(path)


def _append_result_summary(result: Dict[str, Any], csv_rel: str = "data/results/results_summary.csv") -> str:
    rec = {
        "claim_id": result.get("claim_id"),
        "final_score": result.get("final_score"),
        "action": result.get("action"),
        "fraud_prob": result.get("fraud_prob"),
        "anomaly_score": result.get("anomaly_score"),
        "similarity_score": result.get("similarity_score"),
        "blacklist_flag": (result.get("enrichment") or {}).get("blacklist_flag", 0),
        "timestamp": _now_iso()
    }
    df_new = pd.DataFrame([rec])
    abs_csv = _abs_rel(csv_rel)
    try:
        if store_mod is not None and hasattr(store_mod, "save_df"):
            # store.save_df expects project-relative path in this codebase
            if os.path.exists(abs_csv):
                df_exist = pd.read_csv(abs_csv)
                df_out = pd.concat([df_exist, df_new], ignore_index=True)
            else:
                df_out = df_new
            store_mod.save_df(df_out, csv_rel)
            logger.info("Appended results via store.save_df -> %s", csv_rel)
            return str((PROJECT_ROOT / csv_rel).resolve())
        else:
            # fallback write directly
            if os.path.exists(abs_csv):
                df_exist = pd.read_csv(abs_csv)
                df_out = pd.concat([df_exist, df_new], ignore_index=True)
            else:
                df_out = df_new
            os.makedirs(os.path.dirname(abs_csv), exist_ok=True)
            df_out.to_csv(abs_csv, index=False)
            logger.info("Saved results_summary directly -> %s", abs_csv)
            return abs_csv
    except Exception:
        logger.exception("Failed to append result summary")
        # try best-effort direct write
        try:
            os.makedirs(os.path.dirname(abs_csv), exist_ok=True)
            df_new.to_csv(abs_csv, index=False)
            return abs_csv
        except Exception:
            logger.exception("Final fallback failed writing summary")
            return abs_csv


def _ensure_list(x: Optional[List[str]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


# ---------------------------
# Main pipeline for single claim
# ---------------------------
def run_single_claim(claim_record: Dict[str, Any], attachments: Optional[List[str]] = None, history_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Process one claim dict end-to-end and return result dict.
    Always ensures 'action' and 'final_score' keys exist.
    """
    claim_id = str(claim_record.get("claim_id") or f"noid_{int(datetime.utcnow().timestamp())}")
    logger.info("Pipeline start claim_id=%s", claim_id)

    result: Dict[str, Any] = {
        "claim_id": claim_id,
        "input": claim_record,
        "steps": {},
        "errors": []
    }

    # -------------------------
    # 1) Normalization
    # -------------------------
    norm_row: Dict[str, Any] = {}
    try:
        if normalize_mod is None:
            raise RuntimeError("normalize_fields module unavailable")
        # prefer DataFrame-style normalizer function names
        if hasattr(normalize_mod, "normalize_claims_df"):
            df_norm = normalize_mod.normalize_claims_df(pd.DataFrame([claim_record]))
            norm_row = df_norm.iloc[0].to_dict()
        elif hasattr(normalize_mod, "normalize_claim_dict"):
            norm_row = normalize_mod.normalize_claim_dict(claim_record)
        else:
            # fallback: try generic function named "normalize"
            if hasattr(normalize_mod, "normalize"):
                maybe = normalize_mod.normalize(claim_record)
                if isinstance(maybe, dict):
                    norm_row = maybe
                else:
                    norm_row = dict(claim_record)
            else:
                raise RuntimeError("normalize_fields has no usable API")
        # ensure expected flags
        norm_row.setdefault("missing_amount_flag", int(norm_row.get("claim_amount") in [None, -1]))
        norm_row.setdefault("missing_info_flag", int(not bool(norm_row.get("phone")) or not bool(norm_row.get("garage_id"))))
        result["normalized"] = norm_row
        result["steps"]["normalize"] = "ok"
    except Exception as e:
        logger.exception("Normalization failed: %s", e)
        result["errors"].append(f"normalize_error:{e}")
        # safe fallback to claim_record copy
        norm_row = dict(claim_record)
        norm_row.setdefault("description", claim_record.get("description", ""))
        norm_row.setdefault("claim_amount", claim_record.get("claim_amount"))
        norm_row.setdefault("policy_sum_insured", claim_record.get("policy_sum_insured"))
        norm_row.setdefault("missing_info_flag", int(not bool(claim_record.get("phone")) or not bool(claim_record.get("garage_id"))))
        result["normalized"] = norm_row
        result["steps"]["normalize"] = "fallback"

    # -------------------------
    # 2) Attachments: route & extract
    # -------------------------
    attachments = _ensure_list(attachments)
    extracted_texts: List[str] = []
    try:
        for p in attachments:
            routed = p
            # file_router optional (may be in ingestion.file_router)
            try:
                file_router_mod = _safe_import("fraud_detection.ingestion.file_router")
                if file_router_mod and hasattr(file_router_mod, "route_file"):
                    routed = file_router_mod.route_file(p)
            except Exception:
                routed = p

            suffix = Path(routed).suffix.lower()
            try:
                if suffix == ".pdf" and pdf_extractor_mod is not None and hasattr(pdf_extractor_mod, "extract_text_from_pdf"):
                    txt, summary = pdf_extractor_mod.extract_text_from_pdf(routed)
                    extracted_texts.append(txt or "")
                elif suffix in [".png", ".jpg", ".jpeg", ".tiff"] and ocr_extractor_mod is not None and hasattr(ocr_extractor_mod, "extract_text_from_image"):
                    txt = ocr_extractor_mod.extract_text_from_image(routed)
                    extracted_texts.append(txt or "")
                else:
                    logger.debug("No extractor for %s (ext=%s)", routed, suffix)
            except Exception as ex:
                logger.exception("Attachment extraction error %s: %s", routed, ex)
        result["steps"]["attachments"] = f"extracted_{len(extracted_texts)}"
    except Exception as e:
        logger.exception("Attachments processing top-level error: %s", e)
        result["errors"].append(f"attachments_error:{e}")
        result["steps"]["attachments"] = "error"

    # -------------------------
    # 3) Build full_text and save
    # -------------------------
    try:
        desc = str(norm_row.get("description", "") or "")
        full_text = desc
        if extracted_texts:
            full_text = desc + "\n\n" + "\n\n".join(extracted_texts)
        # try saving via store
        try:
            if store_mod is not None and hasattr(store_mod, "save_text"):
                store_mod.save_text(claim_id, full_text)
        except Exception:
            logger.debug("store.save_text failed")
        result["full_text"] = full_text[:10000]
        result["steps"]["text"] = "ok"
    except Exception as e:
        logger.exception("Text assembly failed: %s", e)
        result["errors"].append(f"text_error:{e}")
        result["full_text"] = norm_row.get("description", "")
        result["steps"]["text"] = "error"

    # -------------------------
    # 4) Numeric features (USE feature_builder_mod)
    # -------------------------
    features_df = None
    feature_row: Dict[str, Any] = {}
    try:
        if feature_builder_mod is None:
            raise RuntimeError("feature_builder module missing (fraud_detection.features.feature_builder)")
        # Try DataFrame API first
        if hasattr(feature_builder_mod, "build_features_from_df"):
            try:
                features_df = feature_builder_mod.build_features_from_df(pd.DataFrame([norm_row]), history_df=history_df)
            except Exception:
                # try dict-level API next
                features_df = None
        if features_df is None and hasattr(feature_builder_mod, "build_features_from_dict"):
            fr = feature_builder_mod.build_features_from_dict(norm_row, history_df=history_df)
            # convert single dict to DataFrame-like for consistency
            features_df = pd.DataFrame([fr])
        # Final fallback: if module exposes build_numeric_features (compat)
        if features_df is None and hasattr(feature_builder_mod, "build_numeric_features"):
            features_df = feature_builder_mod.build_numeric_features(pd.DataFrame([norm_row]), history_df)
        if features_df is None:
            raise RuntimeError("feature_builder provided no usable build function")
        feature_row = features_df.iloc[0].to_dict()
        result["features"] = feature_row
        result["steps"]["numeric_features"] = "ok"
    except Exception as e:
        logger.exception("Numeric features failed: %s", e)
        result["errors"].append(f"numeric_features_error:{e}")
        result["steps"]["numeric_features"] = "error"
        feature_row = {}
        result["features"] = {}

    # -------------------------
    # 5) Embedding + similarity (best-effort) using similarity_mod
    # -------------------------
    similarity_score = 0.0
    try:
        # Try FAISS-based historical similarity if embeddings file exists
        emb_vec = None
        # Build query vector using similarity_mod.get_embeddings or embedder_mod
        if similarity_mod is not None and hasattr(similarity_mod, "get_embeddings"):
            try:
                qvecs = similarity_mod.get_embeddings([result.get("full_text", "") or ""])
                if isinstance(qvecs, (list, np.ndarray)) and len(qvecs) > 0:
                    emb_vec = np.array(qvecs[0], dtype="float32")
            except Exception:
                emb_vec = None

        # If no embedding model above, try generic embedder_mod
        if emb_vec is None and embedder_mod is not None and hasattr(embedder_mod, "embed_text"):
            try:
                emb = embedder_mod.embed_text(result.get("full_text", "") or "")
                if emb is not None:
                    emb_vec = np.array(emb, dtype="float32")
            except Exception:
                emb_vec = None

        # Now, if we have a similarity module with FAISS helpers and a saved embeddings file, compute similarity
        emb_file = _abs_rel("data/processed/embeddings.npy")
        if emb_vec is not None and similarity_mod is not None and hasattr(similarity_mod, "build_faiss_index") and os.path.exists(emb_file):
            try:
                hist_emb = np.load(emb_file)
                if hist_emb is not None and getattr(hist_emb, "shape", None) and hist_emb.shape[0] > 0:
                    idx = similarity_mod.build_faiss_index(hist_emb)
                    similarity_score = float(similarity_mod.compute_similarity_score(idx, emb_vec))
            except Exception as e:
                logger.debug("similarity via embeddings.npy failed: %s", e)
                similarity_score = 0.0
        else:
            # fallback: if similarity_mod exposes compute_similarity_score and an index loader
            try:
                # if similarity_mod provides quick_similarity(text) use it
                if similarity_mod is not None and hasattr(similarity_mod, "compute_similarity_score"):
                    # but compute_similarity_score expects index+vector; skip unless index available
                    similarity_score = 0.0
                elif similarity_mod is not None and hasattr(similarity_mod, "quick_similarity"):
                    try:
                        similarity_score = float(similarity_mod.quick_similarity(result.get("full_text", "")))
                    except Exception:
                        similarity_score = 0.0
                else:
                    similarity_score = 0.0
            except Exception:
                similarity_score = 0.0

        similarity_score = float(np.clip(similarity_score, 0.0, 1.0))
        result["similarity_score"] = similarity_score
        result["steps"]["text_features"] = "ok"
    except Exception as e:
        logger.exception("Similarity/embedding failed: %s", e)
        result["errors"].append(f"similarity_error:{e}")
        result["similarity_score"] = 0.0
        result["steps"]["text_features"] = "error"

    # -------------------------
    # 6) Enrichment (blacklist simplified)
    # -------------------------
    enrichment = {}
    try:
        if external_lookup_mod is not None and hasattr(external_lookup_mod, "check_blacklist"):
            try:
                bl_flag, bl_reasons = external_lookup_mod.check_blacklist(
                    customer_id=norm_row.get("customer_id"),
                    phone=norm_row.get("phone"),
                    garage_id=norm_row.get("garage_id")
                )
                enrichment = {"blacklist_flag": int(bool(bl_flag)), "blacklist_reasons": bl_reasons or []}
            except Exception as e:
                logger.debug("external_lookup.check_blacklist failed: %s", e)
                enrichment = {"blacklist_flag": 0, "blacklist_reasons": []}
        else:
            enrichment = {"blacklist_flag": 0, "blacklist_reasons": []}
        result["enrichment"] = enrichment
        result["steps"]["enrichment"] = "ok"
    except Exception as e:
        logger.exception("Enrichment error: %s", e)
        result["errors"].append(f"enrichment_error:{e}")
        result["enrichment"] = {"blacklist_flag": 0}
        result["steps"]["enrichment"] = "error"

    # -------------------------
    # 7) Anomaly scoring
    # -------------------------
    anomaly_score = 0.0
    try:
        if anomaly_mod is not None and hasattr(anomaly_mod, "predict_anomaly_score") and model_utils_mod is not None:
            X_num = None
            try:
                if features_df is not None:
                    X_num = model_utils_mod.get_numeric_matrix(features_df)
                else:
                    X_num = model_utils_mod.get_numeric_matrix(pd.DataFrame([feature_row]))
            except Exception:
                X_num = None
            raw_anom = anomaly_mod.predict_anomaly_score(X_num if X_num is not None else feature_row)
            anomaly_score = float(_first_or_scalar(raw_anom))
        else:
            logger.info("Anomaly detector unavailable -> anomaly_score=0.0")
            anomaly_score = 0.0
        result["anomaly_score"] = anomaly_score
        result["steps"]["anomaly"] = "ok"
    except Exception as e:
        logger.exception("Anomaly error: %s", e)
        result["errors"].append(f"anomaly_error:{e}")
        try:
            anomaly_score = float(feature_row.get("amount_ratio") or 0.0)
        except Exception:
            anomaly_score = 0.0
        result["anomaly_score"] = anomaly_score
        result["steps"]["anomaly"] = "fallback"

    # -------------------------
    # 8) Supervised fraud classifier
    # -------------------------
    fraud_prob = 0.0
    try:
        if fraud_clf_mod is not None and hasattr(fraud_clf_mod, "predict_proba") and model_utils_mod is not None:
            X_num = None
            try:
                if features_df is not None:
                    X_num = model_utils_mod.get_numeric_matrix(features_df)
                else:
                    X_num = model_utils_mod.get_numeric_matrix(pd.DataFrame([feature_row]))
            except Exception:
                X_num = None
            raw_prob = fraud_clf_mod.predict_proba(X_num if X_num is not None else feature_row)
            fraud_prob = float(_first_or_scalar(raw_prob))
        else:
            logger.info("fraud_classifier unavailable -> fraud_prob=0.0")
            fraud_prob = 0.0
        result["fraud_prob"] = fraud_prob
        result["steps"]["fraud_classifier"] = "ok"
    except Exception as e:
        logger.exception("Fraud classifier error: %s", e)
        result["errors"].append(f"fraud_classifier_error:{e}")
        result["fraud_prob"] = 0.0
        result["steps"]["fraud_classifier"] = "error"

    # -------------------------
    # 9) Business rules (prepare context)
    # -------------------------
    rule_flags = {}
    try:
        rule_ctx = {
            "incident_date": norm_row.get("incident_date"),
            "policy_id": norm_row.get("policy_id") or claim_record.get("policy_id"),
            "policy_id_record": claim_record.get("policy_id_record") or norm_row.get("policy_id_record"),
            "claim_amount": norm_row.get("claim_amount"),
            "median_amount_for_policy": norm_row.get("policy_sum_insured") or norm_row.get("policy_sum_insured")
        }
        if rules_mod is not None and hasattr(rules_mod, "extract_rule_flags"):
            rule_flags = rules_mod.extract_rule_flags(rule_ctx)
        else:
            rule_flags = {}
        result["rule_flags"] = rule_flags
        result["steps"]["rules"] = "ok"
    except Exception as e:
        logger.exception("Rules extraction failed: %s", e)
        result["errors"].append(f"rules_error:{e}")
        result["rule_flags"] = {}
        result["steps"]["rules"] = "error"

    # -------------------------
    # 10) Scoring: compute final decision
    # -------------------------
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

        if scoring_mod is None or not hasattr(scoring_mod, "compute_final_score"):
            logger.warning("Scoring module missing -> conservative manual_review")
            final_decision = {"final_score": None, "action": "manual_review", "breakdown": {}}
        else:
            final_decision = scoring_mod.compute_final_score(scoring_context)

        # ensure keys exist
        result["final_score"] = final_decision.get("final_score")
        result["action"] = final_decision.get("action") or "manual_review"
        result["breakdown"] = final_decision.get("breakdown", {})
        result["steps"]["scoring"] = "ok"
        logger.info("Scoring done action=%s score=%s", result["action"], result["final_score"])
    except Exception as e:
        logger.exception("Scoring error: %s", e)
        result["errors"].append(f"scoring_error:{e}")
        result["final_score"] = None
        result["action"] = "manual_review"
        result["breakdown"] = {}
        result["steps"]["scoring"] = "error"

    # -------------------------
    # 11) Explanation (best-effort)
    # -------------------------
    try:
        explain_payload = {
            "final_score": result.get("final_score"),
            "action": result.get("action"),
            "breakdown": result.get("breakdown"),
            "claim_id": claim_id,
            "brief": (result.get("full_text") or "")[:500]
        }
        if explain_gen_mod is not None and hasattr(explain_gen_mod, "generate_text_explanation"):
            try:
                explanation = explain_gen_mod.generate_text_explanation(explain_payload)
            except Exception:
                explanation = f"Action={result.get('action')}. Score={result.get('final_score')}."
        else:
            # human-friendly fallback
            bd = result.get("breakdown") or {}
            explanation = f"Action={result.get('action')}. Score={result.get('final_score')}. Rules: {bd.get('rule_flags', {})}"
        result["explanation"] = explanation
        result["steps"]["explain"] = "ok"
    except Exception as e:
        logger.exception("Explanation failed: %s", e)
        result["errors"].append(f"explain_error:{e}")
        result["explanation"] = ""
        result["steps"]["explain"] = "error"

    # -------------------------
    # 12) Save results (JSON + CSV summary)
    # -------------------------
    try:
        json_path = _save_result_json(claim_id, result)
        csv_path = _append_result_summary(result)
        result["saved_paths"] = {"json": json_path, "summary_csv": csv_path}
        result["steps"]["save"] = "ok"
    except Exception as e:
        logger.exception("Save error: %s", e)
        result["errors"].append(f"save_error:{e}")
        result["steps"]["save"] = "error"

    # -------------------------
    # 13) HITL enqueue if manual_review
    # -------------------------
    try:
        if result.get("action") == "manual_review":
            if HITL_AVAILABLE and hasattr(review_queue_mod, "enqueue_for_review"):
                try:
                    qid = review_queue_mod.enqueue_for_review(result)
                    result["steps"]["hitl_enqueue"] = "ok"
                    result["hitl_queue_id"] = qid
                    logger.info("Enqueued for HITL queue_id=%s", qid)
                except Exception as e:
                    logger.exception("Enqueue HITL failed: %s", e)
                    result["errors"].append(f"hitl_enqueue_error:{e}")
                    result["steps"]["hitl_enqueue"] = "error"
            else:
                # store skipped indicator so UI shows it's waiting but HITL system not available
                logger.info("HITL unavailable, marking hitl_enqueue as skipped")
                result["steps"]["hitl_enqueue"] = "skipped"
        else:
            result["steps"]["hitl_enqueue"] = "not_required"
    except Exception as e:
        logger.exception("HITL integration error: %s", e)
        result["errors"].append(f"hitl_integration_error:{e}")

    logger.info("Pipeline finished claim_id=%s action=%s", claim_id, result.get("action"))
    # final safety: guarantee action and final_score keys (avoid KeyError in UI)
    result.setdefault("action", "manual_review")
    result.setdefault("final_score", result.get("final_score", None))
    return result


# ---------------------------
# Batch runner
# ---------------------------
def run_batch(claims_df: pd.DataFrame, attachments_map: Optional[Dict[str, List[str]]] = None, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    attachments_map = attachments_map or {}
    rows = []
    for _, r in claims_df.iterrows():
        claim = r.to_dict()
        att = attachments_map.get(str(claim.get("claim_id")), [])
        try:
            res = run_single_claim(claim, attachments=att, history_df=history_df)
            rows.append({
                "claim_id": res.get("claim_id"),
                "final_score": res.get("final_score"),
                "action": res.get("action"),
                "fraud_prob": res.get("fraud_prob"),
                "anomaly_score": res.get("anomaly_score"),
                "similarity_score": res.get("similarity_score")
            })
        except Exception as e:
            logger.exception("Batch row processing failed: %s", e)
            rows.append({
                "claim_id": claim.get("claim_id"),
                "final_score": None,
                "action": "error"
            })
    return pd.DataFrame(rows)


# ---------------------------
# Manual test block
# ---------------------------
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)

    example = {
        "claim_id": "T1001",
        "customer_id": "C001",
        "policy_id": "POL-111",
        "policy_id_record": "POL-111",
        "claim_amount": 4500,
        "policy_sum_insured": 50000,
        "incident_date": "2024-06-10",
        "description": "Minor scratch on left door while parking.",
        "phone": "9876543210",
        "garage_id": "GAR-001"
    }
    res = run_single_claim(example, attachments=[])
    print(json.dumps(res, indent=2, default=str))
