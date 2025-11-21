# fraud_detection/orchestration/pipeline_runner.py
"""
Robust pipeline runner for Fraud Detection & Risk Assessment (full orchestration).

This implementation follows OPTION B (Full Orchestration):
 - keeps unified_extractor + pdf/ocr/text fallbacks
 - keeps multi-path feature builder fallbacks
 - keeps external lookups and enrichment
 - keeps embedding fallbacks and similarity fallbacks
 - integrates anomaly & fraud classifier models (with model_utils helper)
 - runs rules -> scoring (decision_engine.scoring) -> explainability (LLM or rule-based)
 - persists results and enqueues HITL when required

The runner is defensive: any optional module may be None and the pipeline will still
return a best-effort result (with errors logged inside result["errors"]).
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import yaml

# -----------------------
# Logging (project logger preferred)
# -----------------------
try:
    from fraud_detection.logging.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pipeline_runner")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

# -----------------------
# Project root resolution
# -----------------------
FD_PROJECT_ROOT = os.environ.get("FD_PROJECT_ROOT")
if FD_PROJECT_ROOT:
    PROJECT_ROOT = Path(FD_PROJECT_ROOT).resolve()
else:
    # repository root assumed 3 parents up from this file
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -----------------------
# Load config (try multiple expected locations)
# -----------------------
CONFIG: Dict[str, Any] = {}
for p in (
    PROJECT_ROOT / "configs" / "model.yaml",
    PROJECT_ROOT / "fraud_detection" / "configs" / "model.yaml",
):
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                CONFIG = yaml.safe_load(fh) or {}
                logger.info("Loaded config: %s", p)
                break
    except Exception:
        logger.exception("Failed to read config %s", p)
if not CONFIG:
    logger.info("No model config found; using defaults")

# -----------------------
# Results folder
# -----------------------
RESULTS_FOLDER = PROJECT_ROOT / "data" / "results"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# -----------------------
# Defensive import helper
# -----------------------
def _safe_import(module_path: str):
    try:
        mod = __import__(module_path, fromlist=["*"])
        logger.debug("Imported %s", module_path)
        return mod
    except Exception as e:
        logger.debug("Optional import failed %s: %s", module_path, e)
        return None


# -----------------------
# Optional modules (may be None)
# -----------------------
# Preprocessing
field_normalizer_mod = _safe_import("fraud_detection.preprocessing.field_normalizer")
schema_validator_mod = _safe_import("fraud_detection.preprocessing.schema_validator")
text_cleaner_mod = _safe_import("fraud_detection.preprocessing.text_cleaner")

# Extraction (prefer unified_extractor)
unified_extractor_mod = _safe_import("fraud_detection.extraction.unified_extractor")
ocr_extractor_mod = _safe_import("fraud_detection.extraction.ocr_extractor")
text_extractor_mod = _safe_import("fraud_detection.extraction.text_extractor")

# Features & similarity
feature_builder_mod = _safe_import("fraud_detection.features.feature_builder")
similarity_mod = _safe_import("fraud_detection.features.similarity_index")

# Generative AI (embeddings / explain LLM)
embedder_mod = _safe_import("fraud_detection.generative_ai.embedder")
explain_gen_mod = _safe_import("fraud_detection.generative_ai.explain_generator")

# Models
anomaly_mod = _safe_import("fraud_detection.models.anomaly_detector")
fraud_clf_mod = _safe_import("fraud_detection.models.fraud_classifier")
model_utils_mod = _safe_import("fraud_detection.models.model_utils")

# Enrichment / external lookups
external_lookup_mod = _safe_import("fraud_detection.enrichment.external_lookup")

# Decision engine
rules_mod = _safe_import("fraud_detection.decision_engine.rules")
scoring_mod = _safe_import("fraud_detection.decision_engine.scoring")
explainability_mod = _safe_import("fraud_detection.decision_engine.explainability")

# Storage & HITL
store_mod = _safe_import("fraud_detection.storage.store")
review_queue_mod = _safe_import("fraud_detection.hitl.review_queue")
feedback_mod = _safe_import("fraud_detection.hitl.feedback_processor")

HITL_AVAILABLE = review_queue_mod is not None

# -----------------------
# Helpers
# -----------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()

def _abs_rel(rel: str) -> str:
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / rel).resolve())

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
    safe_id = str(claim_id).replace(os.sep, "_")[:120]
    path = RESULTS_FOLDER / f"result_{safe_id}.json"
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
        "timestamp": _now_iso(),
    }
    df_new = pd.DataFrame([rec])
    abs_csv = _abs_rel(csv_rel)
    try:
        if store_mod is not None and hasattr(store_mod, "save_df"):
            try:
                if os.path.exists(abs_csv):
                    df_exist = pd.read_csv(abs_csv)
                    df_out = pd.concat([df_exist, df_new], ignore_index=True)
                else:
                    df_out = df_new
                store_mod.save_df(df_out, csv_rel)
                logger.info("Appended results via store.save_df -> %s", csv_rel)
                return str((PROJECT_ROOT / csv_rel).resolve())
            except Exception:
                logger.exception("store_mod.save_df failed; falling back to local write")

        os.makedirs(os.path.dirname(abs_csv), exist_ok=True)
        if os.path.exists(abs_csv):
            df_exist = pd.read_csv(abs_csv)
            df_out = pd.concat([df_exist, df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_csv(abs_csv, index=False)
        logger.info("Saved results_summary directly -> %s", abs_csv)
        return abs_csv
    except Exception:
        logger.exception("Failed to append result summary")
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

# -----------------------
# Main pipeline: single claim
# -----------------------
def run_single_claim(claim: Dict[str, Any], attachments: Optional[List[str]] = None, history_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    claim_id = str(claim.get("claim_id") or f"noid_{int(datetime.utcnow().timestamp())}")
    logger.info("Pipeline start claim_id=%s", claim_id)

    result: Dict[str, Any] = {
        "claim_id": claim_id,
        "input": claim,
        "normalized": {},
        "features": {},
        "extracted_texts": [],
        "enrichment": {},
        "steps": {},
        "errors": []
    }

    # -------------------------
    # 1) Normalization
    # -------------------------
    try:
        norm_row: Dict[str, Any] = {}

        # Prefer dict-level normalizer
        if field_normalizer_mod is not None:
            if hasattr(field_normalizer_mod, "normalize_claim_dict"):
                try:
                    norm_row = field_normalizer_mod.normalize_claim_dict(claim)
                except Exception:
                    logger.debug("normalize_claim_dict failed; will try df-level")
                    norm_row = {}
            # DF-level fallback
            if (not norm_row) and hasattr(field_normalizer_mod, "normalize_claims_df"):
                try:
                    df = pd.DataFrame([claim])
                    norm_row = field_normalizer_mod.normalize_claims_df(df).iloc[0].to_dict()
                except Exception:
                    logger.debug("normalize_claims_df fallback failed")
                    norm_row = {}

        # If normalizer missing or failed, conservative shallow copy
        if not norm_row:
            norm_row = dict(claim)

        # Schema validation & sanitization (optional)
        if schema_validator_mod is not None and hasattr(schema_validator_mod, "validate_and_sanitize"):
            try:
                ok, sanitized = schema_validator_mod.validate_and_sanitize(norm_row)
                if ok and isinstance(sanitized, dict) and sanitized:
                    norm_row = sanitized
                else:
                    logger.debug("Schema validation returned partial/invalid sanitized; using normalized row")
            except Exception:
                logger.exception("Schema validator raised; continuing with normalized row")

        # Ensure expected flags
        norm_row.setdefault("missing_amount_flag", int(norm_row.get("claim_amount") in [None, -1, ""]))
        norm_row.setdefault("missing_info_flag", int(not bool(norm_row.get("phone")) or not bool(norm_row.get("garage_id"))))

        result["normalized"] = norm_row
        result["steps"]["normalize"] = "ok"
    except Exception as e:
        logger.exception("Normalization failed: %s", e)
        result["errors"].append(f"normalize_error:{e}")
        result["normalized"] = dict(claim)
        result["steps"]["normalize"] = "error"

    # -------------------------
    # 2) Attachments extraction (unified preferred, then pdf/text/ocr)
    # -------------------------
    attachments = _ensure_list(attachments)
    extracted_texts: List[str] = []
    try:
        for p in attachments:
            try:
                if unified_extractor_mod is not None and hasattr(unified_extractor_mod, "extract_text"):
                    txt = unified_extractor_mod.extract_text(p)
                    extracted_texts.append(txt or "")
                    continue

                suffix = Path(str(p)).suffix.lower()
                if suffix == ".pdf" and text_extractor_mod is not None and hasattr(text_extractor_mod, "extract_text_from_pdf"):
                    txt = text_extractor_mod.extract_text_from_pdf(p)
                    extracted_texts.append(txt or "")
                elif suffix in {".png", ".jpg", ".jpeg", ".tiff"} and ocr_extractor_mod is not None and hasattr(ocr_extractor_mod, "extract_text_from_image"):
                    txt = ocr_extractor_mod.extract_text_from_image(p)
                    extracted_texts.append(txt or "")
                else:
                    logger.debug("No extractor for attachment %s (suffix=%s)", p, suffix)
            except Exception as ex:
                logger.exception("Attachment extraction error %s: %s", p, ex)
        result["extracted_texts"] = extracted_texts
        result["steps"]["attachments"] = f"extracted_{len(extracted_texts)}"
    except Exception as e:
        logger.exception("Attachments processing top-level error: %s", e)
        result["errors"].append(f"attachments_error:{e}")
        result["steps"]["attachments"] = "error"

    # -------------------------
    # 3) Text assembly & cleaning
    # -------------------------
    try:
        desc = str(result["normalized"].get("description", "") or "")
        full_text = desc
        if extracted_texts:
            full_text = desc + "\n\n" + "\n\n".join(extracted_texts)
        if text_cleaner_mod is not None and hasattr(text_cleaner_mod, "clean_description"):
            try:
                full_text = text_cleaner_mod.clean_description(full_text, redact=False)
            except Exception:
                logger.debug("text_cleaner.clean_description failed; using raw assembled text")
        result["full_text"] = (full_text or "")[:10000]
        result["steps"]["text"] = "ok"
    except Exception as e:
        logger.exception("Text assembly failed: %s", e)
        result["errors"].append(f"text_error:{e}")
        result["full_text"] = result["normalized"].get("description", "")
        result["steps"]["text"] = "error"

    # -------------------------
    # 4) Numeric / derived features
    # -------------------------
    features_df = None
    feature_row: Dict[str, Any] = {}
    try:
        if feature_builder_mod is None:
            raise RuntimeError("feature_builder module missing (fraud_detection.features.feature_builder)")

        # prefer DataFrame API
        if hasattr(feature_builder_mod, "build_features_from_df"):
            try:
                features_df = feature_builder_mod.build_features_from_df(pd.DataFrame([result["normalized"]]), history_df=history_df)
            except Exception:
                logger.debug("build_features_from_df failed; will try other APIs")
                features_df = None

        # dict-level API
        if features_df is None and hasattr(feature_builder_mod, "build_features_from_dict"):
            try:
                fr = feature_builder_mod.build_features_from_dict(result["normalized"], history_df=history_df)
                features_df = pd.DataFrame([fr])
            except Exception:
                logger.debug("build_features_from_dict failed")
                features_df = None

        # old compatibility name
        if features_df is None and hasattr(feature_builder_mod, "build_numeric_features"):
            try:
                features_df = feature_builder_mod.build_numeric_features(pd.DataFrame([result["normalized"]]), history_df)
            except Exception:
                logger.debug("build_numeric_features failed")
                features_df = None

        if features_df is None:
            raise RuntimeError("feature_builder provided no usable build function")
        feature_row = features_df.iloc[0].to_dict()
        result["features"] = feature_row
        result["steps"]["numeric_features"] = "ok"
    except Exception as e:
        logger.exception("Numeric features failed: %s", e)
        result["errors"].append(f"numeric_features_error:{e}")
        result["steps"]["numeric_features"] = "error"
        result["features"] = {}

    # -------------------------
    # 5) Embedding + similarity
    # -------------------------
    similarity_score = 0.0
    try:
        emb_vec = None
        # 1) similarity_mod.get_embeddings (if provided)
        if similarity_mod is not None and hasattr(similarity_mod, "get_embeddings"):
            try:
                vecs = similarity_mod.get_embeddings([result.get("full_text", "") or ""])
                if isinstance(vecs, (list, np.ndarray)) and len(vecs) > 0:
                    emb_vec = np.array(vecs[0], dtype="float32")
            except Exception:
                logger.debug("similarity_mod.get_embeddings failed")

        # 2) generic embedder_mod.embed_text
        if emb_vec is None and embedder_mod is not None and hasattr(embedder_mod, "embed_text"):
            try:
                e = embedder_mod.embed_text(result.get("full_text", "") or "")
                if e is not None:
                    emb_vec = np.array(e, dtype="float32")
            except Exception:
                logger.debug("embedder_mod.embed_text failed")

        # 3) compute against stored embeddings if available (faiss or similarity_mod helpers)
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
            # fallback: quick_similarity or text-based heuristics
            if similarity_mod is not None and hasattr(similarity_mod, "quick_similarity"):
                try:
                    similarity_score = float(similarity_mod.quick_similarity(result.get("full_text", "")))
                except Exception:
                    similarity_score = 0.0
            else:
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
    # 6) Enrichment (external lookups / blacklist)
    # -------------------------
    enrichment = {}
    try:
        if external_lookup_mod is not None and hasattr(external_lookup_mod, "check_blacklist"):
            try:
                bl_flag, bl_reasons = external_lookup_mod.check_blacklist(
                    customer_id=result["normalized"].get("customer_id"),
                    phone=result["normalized"].get("phone"),
                    garage_id=result["normalized"].get("garage_id")
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
                if isinstance(feature_row, dict) and feature_row:
                    X_num = model_utils_mod.get_numeric_matrix(pd.DataFrame([feature_row]))
                elif features_df is not None:
                    X_num = model_utils_mod.get_numeric_matrix(features_df)
            except Exception:
                X_num = None
            raw_anom = anomaly_mod.predict_anomaly_score(X_num if X_num is not None else feature_row)
            anomaly_score = float(_first_or_scalar(raw_anom))
        else:
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
    # 8) Fraud classifier
    # -------------------------
    fraud_prob = 0.0
    try:
        if fraud_clf_mod is not None and hasattr(fraud_clf_mod, "predict_proba") and model_utils_mod is not None:
            X_num = None
            try:
                if isinstance(feature_row, dict) and feature_row:
                    X_num = model_utils_mod.get_numeric_matrix(pd.DataFrame([feature_row]))
                elif features_df is not None:
                    X_num = model_utils_mod.get_numeric_matrix(features_df)
            except Exception:
                X_num = None
            raw_prob = fraud_clf_mod.predict_proba(X_num if X_num is not None else feature_row)
            fraud_prob = float(_first_or_scalar(raw_prob))
        else:
            fraud_prob = 0.0
        result["fraud_prob"] = fraud_prob
        result["steps"]["fraud_classifier"] = "ok"
    except Exception as e:
        logger.exception("Fraud classifier error: %s", e)
        result["errors"].append(f"fraud_classifier_error:{e}")
        result["fraud_prob"] = 0.0
        result["steps"]["fraud_classifier"] = "error"

    # -------------------------
    # 9) Rules extraction (decision engine)
    # -------------------------
    rule_flags = {}
    try:
        rule_ctx = {
            "incident_date": result["normalized"].get("incident_date"),
            "policy_id": result["normalized"].get("policy_id") or claim.get("policy_id"),
            "policy_id_record": claim.get("policy_id_record") or result["normalized"].get("policy_id_record"),
            "claim_amount": result["normalized"].get("claim_amount"),
            "median_amount_for_policy": result["normalized"].get("policy_sum_insured") or claim.get("policy_sum_insured")
        }
        if rules_mod is not None and hasattr(rules_mod, "extract_rule_flags"):
            rule_flags = rules_mod.extract_rule_flags(rule_ctx) or {}
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
    # 10) Scoring (decision engine)
    # -------------------------
    try:
        missing_info_flag = int(result["normalized"].get("missing_info_flag", 0))
        scoring_context = {
            "fraud_prob": fraud_prob,
            "anomaly_score": anomaly_score,
            "similarity_score": similarity_score,
            "blacklist_flag": int(result.get("enrichment", {}).get("blacklist_flag", 0)),
            "missing_info_flag": missing_info_flag,
            "normalized": result["normalized"],
            "input": claim,
            "rule_flags": rule_flags
        }

        if scoring_mod is not None and hasattr(scoring_mod, "compute_final_score"):
            final = scoring_mod.compute_final_score(scoring_context)
        else:
            # local fallback (simple weighted aggregation)
            w_cfg = (CONFIG.get("scoring") or {}).get("weights", {}) or {}
            w_model = w_cfg.get("model_score", 0.45)
            w_anom = w_cfg.get("anomaly_score", 0.30)
            w_sim = w_cfg.get("text_similarity_score", w_cfg.get("text_score", 0.15))
            w_black = w_cfg.get("blacklist_score", 0.08)
            w_missing = w_cfg.get("missing_score", 0.02)

            fraud_prob_c = float(np.clip(fraud_prob, 0.0, 1.0))
            similarity_c = float(np.clip(similarity_score, 0.0, 1.0))
            anomaly_c = float(np.clip(anomaly_score, 0.0, 1.0))

            base = (w_model * fraud_prob_c + w_anom * anomaly_c + w_sim * similarity_c + w_black * int(result.get("enrichment", {}).get("blacklist_flag", 0)) + w_missing * missing_info_flag)

            # simple scaling to 0..100
            score_val = float(np.clip(base * 100.0, 0.0, 100.0))

            # rules penalty mapping from CONFIG.rules
            rules_penalty = 0.0
            if isinstance(rule_flags, dict):
                rules_yaml = CONFIG.get("rules", {}) or {}
                for k, v in rule_flags.items():
                    try:
                        if k.startswith("missing") and v:
                            rules_penalty += (rules_yaml.get("missing_fields", {}) or {}).get("penalty", 10)
                        elif k in ("claim_too_high", "policy_mismatch", "policy_expired") and v:
                            rules_penalty += (rules_yaml.get("high_claim_amount", {}) or {}).get("penalty", 20)
                    except Exception:
                        pass

            final_score = float(np.clip(score_val - rules_penalty, 0.0, 100.0))
            thresholds = CONFIG.get("thresholds", {}) or {}
            approve_below = thresholds.get("approve_below", 30)
            reject_above = thresholds.get("reject_above", 70)
            if final_score < approve_below:
                action = "auto_approve"
            elif final_score > reject_above:
                action = "reject"
            else:
                action = "manual_review"

            final = {
                "final_score": round(final_score, 2),
                "action": action,
                "breakdown": {
                    "fraud_prob": fraud_prob_c,
                    "anomaly_score": anomaly_c,
                    "similarity_score": similarity_c,
                    "rule_flags": rule_flags,
                    "weights": {"model": w_model, "anom": w_anom, "sim": w_sim, "black": w_black, "missing": w_missing, "rules_penalty": rules_penalty},
                }
            }

        result["final_score"] = final.get("final_score")
        result["action"] = final.get("action") or "manual_review"
        result["breakdown"] = final.get("breakdown", {})
        result["steps"]["scoring"] = "ok"
    except Exception as e:
        logger.exception("Scoring error: %s", e)
        result["errors"].append(f"scoring_error:{e}")
        result["final_score"] = None
        result["action"] = "manual_review"
        result["breakdown"] = {}
        result["steps"]["scoring"] = "error"

    # -------------------------
    # 11) Explainability (LLM generator preferred, then decision_engine.explainability)
    # -------------------------
    try:
        explain_payload = {
            "final_score": result.get("final_score"),
            "action": result.get("action"),
            "breakdown": result.get("breakdown"),
            "claim_id": claim_id,
            "brief": (result.get("full_text") or "")[:500]
        }
        explanation = ""
        # prefer generative explain generator (LLM)
        if explain_gen_mod is not None and hasattr(explain_gen_mod, "generate_text_explanation"):
            try:
                explanation = explain_gen_mod.generate_text_explanation(explain_payload)
            except Exception:
                logger.debug("explain_gen_mod.generate_text_explanation failed; falling back")
                explanation = ""
        # fallback to decision_engine.explainability.build_explanation (structured)
        if (not explanation) and explainability_mod is not None and hasattr(explainability_mod, "build_explanation"):
            try:
                e_struct = explainability_mod.build_explanation({
                    "final_score": result.get("final_score"),
                    "action": result.get("action"),
                    "breakdown": result.get("breakdown")
                })
                # prefer textual summary + explanation if provided
                if isinstance(e_struct, dict):
                    explanation = e_struct.get("explanation") or e_struct.get("summary") or json.dumps(e_struct)
                else:
                    explanation = str(e_struct)
            except Exception:
                logger.debug("explainability_mod.build_explanation failed")
        # last resort: compact fallback
        if not explanation:
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
    # 12) Persist results (JSON + summary CSV)
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
    # 13) HITL enqueue when manual_review
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
                # fallback: try to persist placeholder via feedback_mod if available
                try:
                    result["steps"]["hitl_enqueue"] = "skipped"
                    logger.info("HITL unavailable, skipped enqueue")
                except Exception:
                    result["steps"]["hitl_enqueue"] = "skipped"
        else:
            result["steps"]["hitl_enqueue"] = "not_required"
    except Exception as e:
        logger.exception("HITL integration error: %s", e)
        result["errors"].append(f"hitl_integration_error:{e}")

    # final safety defaults
    result.setdefault("action", "manual_review")
    result.setdefault("final_score", result.get("final_score", None))
    logger.info("Pipeline finished claim_id=%s action=%s", claim_id, result.get("action"))
    return result


# -----------------------
# Batch runner
# -----------------------
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
