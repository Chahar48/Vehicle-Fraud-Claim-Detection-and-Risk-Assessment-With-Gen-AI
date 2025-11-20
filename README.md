🚗💥 AI-Driven Vehicle Insurance Fraud Detection & Risk Assessment

End-to-end Production-Grade ML + GenAI + HITL System

This repository contains a complete AI/ML pipeline for automated vehicle-insurance claim scoring, fraud detection, anomaly detection, OCR extraction, text analysis, business rules, explainability, and human-in-the-loop (HITL) review.

It is designed to demonstrate real production architecture using:

Machine Learning

Anomaly Detection

OCR and PDF extraction

NLP + Embeddings

Business Rule Engine

Human-in-the-loop (HITL)

FastAPI serving

Streamlit dashboard

Modular architecture

Logging, Storage, Packaging

📌 1. Create & Activate Environment
Windows
python -m venv venv
venv\Scripts\activate

Mac/Linux
python3 -m venv venv
source venv/bin/activate

📌 2. Install Requirements
pip install -r requirements.txt

📌 3. Folder Structure
fraud-detection-genai/
│
├── fraud_detection/
│   ├── ingestion/
│   │     ├── email_ingest.py
│   │     ├── file_router.py
│   │     └── uploader.py
│   │
│   ├── preprocessing/
│   │     ├── pdf_to_images.py
│   │     ├── pdf_to_text.py
│   │     ├── image_to_text.py
│   │     └── normalize_fields.py
│   │
│   ├── extraction/
│   │     ├── ocr_extractor.py
│   │     └── pdf_extractor.py
│   │
│   ├── features/
│   │     ├── numeric_features.py
│   │     └── text_features.py
│   │
│   ├── generative_ai/
│   │     ├── embedder.py
│   │     └── explain_generator.py
│   │
│   ├── models/
│   │     ├── anomaly_detector.py
│   │     ├── fraud_classifier.py
│   │     └── model_utils.py
│   │
│   ├── enrichment/
│   │     └── external_lookup.py
│   │
│   ├── decision_engine/
│   │     ├── rules.py
│   │     ├── scoring.py
│   │     └── explainability.py
│   │
│   ├── hitl/
│   │     ├── review_queue.py
│   │     └── feedback_processor.py
│   │
│   ├── storage/
│   │     └── store.py
│   │
│   ├── orchestration/
│   │     └── pipeline_runner.py
│   │
│   ├── serving/
│   │     └── api_server.py
│   │
│   ├── logging_config.py
│   └── config.py
│
├── ui/
│   └── reviewer_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── results/
│   ├── hitl/
│   └── labels/
│
├── run_api.py
├── pyproject.toml
├── requirements.txt
└── setup.py

📌 4. Full System Architecture (Phases 1–18)
🔵 PHASE 1–3: Ingestion

Route PDFs / images

Accept uploads or email ingestion

Save raw files

🔵 PHASE 4–5: Preprocessing

Convert PDFs → images → text

OCR extraction

Normalize fields (dates, amounts, IDs)

🔵 PHASE 6–7: Extraction

OCR & PDF text extraction

Combine with claim description

🔵 PHASE 8: Feature Engineering

Numeric: ratios, claim history

Text: embeddings, similarity

🔵 PHASE 9–10: ML Models

Isolation Forest (anomaly)

XGBoost / Logistic Regression (fraud classifier)

🔵 PHASE 11: Decision Engine

Business rules (expiry, mismatch, high amount)

Final score + auto_approve / manual_review / reject

Explainability

🔵 PHASE 12: HITL

Queue suspicious claims

Reviewer feedback

Store labels for retraining

🔵 PHASE 13: Storage Layer

Save/load text, PDFs, CSVs, models, labels

🔵 PHASE 14: Monitoring

(Optional) drift + model monitoring

🔵 PHASE 15: Pipeline Runner

Full orchestration engine for end-to-end scoring

🔵 PHASE 16: API Layer

FastAPI endpoint /score_claim

HITL endpoints

🔵 PHASE 17: Entry Script

run_api.py: launch the API server

🔵 PHASE 18: Packaging

pip-installable module

requirements & pyproject

📌 5. Running the API

Start backend server:

python run_api.py

API UI will be available at:
http://localhost:8000/docs

Example Request

Use Postman or CURL:

POST /score_claim
{
  "claim": { ... },
  "attachments": [PDF/Images]
}

📌 6. Running the Streamlit UI
streamlit run ui/reviewer_app.py


UI exposes:

Upload & Score Claims

HITL Pending Queue

Saved Results Browser

Labeled Data + Retraining

📌 7. How to Score a Claim (Step-by-Step)
Step 1 — Upload claim details

Enter:

claim_id

claim_amount

policy details

description

attachments (PDF/Image)

Step 2 — Pipeline Runs Automatically

It performs:

OCR

Text extraction

Normalize fields

Feature engineering

Anomaly detection

Fraud classification

Business rule checks

Final scoring

Save JSON result

Add to HITL queue (if manual_review)

Step 3 — View Result

UI shows:

Final Result

Explanation

Features

Rule Flags

Extracted text

📌 8. HITL Workflow
When pipeline returns action = manual_review:

✔ Added into data/hitl/review_queue.csv
✔ Visible on UI (Pending Reviews tab)
✔ Reviewer inspects info
✔ Marks as FRAUD or NOT FRAUD
✔ Label gets saved into data/labels/labels.csv
✔ Queue entry marked reviewed

This is a real production-compliant HITL loop.

📌 9. Retraining the Fraud Classifier

Open Streamlit tab “Labeled Data / Retrain”

Click:

Export merged dataset

Retrain Model

The system:

Loads labels

Merges with previous scoring summary

Trains supervised model

Saves new model

📌 10. Running Pipeline Without API

You can run a single claim directly:

from fraud_detection.orchestration.pipeline_runner import run_single_claim

claim = {
    "claim_id": "C101",
    "claim_amount": 50000,
    "policy_id": "POL-123",
    "incident_date": "2023-09-10",
    "description": "Collision damage",
    ...
}

result = run_single_claim(claim, attachments=["sample.pdf"])
print(result)

📌 11. What to Show Recruiters
Explain the Architecture

ML + GenAI + OCR + business rules

Full production workflow

HITL loop

FastAPI backend

Streamlit dashboard

Modular pipeline architecture

Demo Sequence

Open UI → Upload a real PDF/Image

Score claim

Show final decision & explanation

Open HITL Tab → Show pending reviews

Mark as fraud/not fraud

Show training tab → retrain model

Show saved results JSON

Talking Points

End-to-end ML/GenAI pipeline

OCR + embeddings

Business rule engine

Human review loop

API + UI + storage layer

Modular, scalable, cloud-ready

📌 12. License

MIT License (or choose your own)

📌 13. AI FRAUD DETECTION — END-TO-END ARCHITECTURE

                           ┌───────────────────────────┐
                           │        User Upload        │
                           │  Claim + PDFs / Images    │
                           └─────────────┬─────────────┘
                                         │
                           Phase 1–3: Ingestion Layer
                                         │
      ┌──────────────────────────────────┼──────────────────────────────────┐
      │                                  │                                  │
┌──────────────┐                  ┌───────────────┐                 ┌────────────────┐
│ file_router  │                  │ uploader.py   │                 │ email_ingest   │
└──────────────┘                  └───────────────┘                 └────────────────┘
      │                                  │                                  │
      └──────────────────────────────────┴──────────────────────────────────┘
                                         │
                              Stored in /data/raw
                                         │
                                         ▼
                         Phase 4–5: Preprocessing Layer
                                         │
             ┌───────────────────────────┼────────────────────────────┐
             │                           │                            │
     ┌────────────────┐         ┌──────────────────┐         ┌──────────────────┐
     │ pdf_to_images  │         │ pdf_to_text      │         │ image_to_text    │
     └────────────────┘         └──────────────────┘         └──────────────────┘
             │                           │                            │
             └───────────────────────────┴────────────────────────────┘
                                         │
                                  Extracted Text
                                         │
                                         ▼
                        Phase 6–7: Unified Extraction Layer
                                         │
                                ┌───────────────────┐
                                │ unified_extractor │
                                └───────────────────┘
                                         │
                                  description + OCR
                                         │
                                         ▼
                       Phase 8: Feature Engineering Layer
                                         │
              ┌──────────────────────────┼────────────────────────────┐
              │                          │                             │
   ┌─────────────────────┐      ┌─────────────────────┐     ┌───────────────────────┐
   │ numeric_features.py │      │ text_features.py     │     │ embedder.py (GenAI)   │
   └─────────────────────┘      └─────────────────────┘     └───────────────────────┘
              │                          │                             │
              └──────────────────────────┴────────────────────────────┘
                                         │
                                         ▼
                          Phase 9–10: ML Models Layer
                                         │
            ┌────────────────────────────┼────────────────────────────────┐
            │                            │                                │
    ┌───────────────────────┐   ┌────────────────────────┐     ┌────────────────────┐
    │ anomaly_detector.py   │   │ fraud_classifier.py     │     │ model_utils.py      │
    └───────────────────────┘   └────────────────────────┘     └────────────────────┘
            │                            │                                │
            └────────────────────────────┴────────────────────────────────┘
                                         │
                                         ▼
                      Phase 11: Decision Engine + Explainability
                                         │
       ┌─────────────────────────────────┼─────────────────────────────────┐
       │                                 │                                 │
┌───────────────────┐          ┌─────────────────────┐          ┌─────────────────────────┐
│ rules.py          │          │ scoring.py          │          │ explainability.py / LLM │
└───────────────────┘          └─────────────────────┘          └─────────────────────────┘
                                         │
                                         ▼
                      Final Output: approve / reject / manual_review
                                         │
                                         ▼
                ┌─────────────────────────────────────────────────────────┐
                │ Phase 12: HITL Queue (Human in the Loop)               │
                │ review_queue.py  ←→  feedback_processor.py             │
                │ Stores → /data/hitl & /data/labels                     │
                └─────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                        Phase 13: Storage Layer (store.py)
                                         │
                    Saves → PDFs, text, CSVs, embeddings, models
                                         │
                                         ▼
                   Phase 14: Monitoring (Optional for POC)
                                         │
                                         ▼
                   Phase 15: Orchestration (pipeline_runner.py)
                                         │
                                         ▼
                      Phase 16: API Layer (FastAPI)
                              api_server.py
                                         │
                                         ▼
                   Phase 17: Entry Script (run_api.py)
                                         │
                                         ▼
                Phase 18: Packaging & Deployment (setup + pyproject)
                                         │
                                         ▼
                     ★ Streamlit UI (reviewer_app.py) ★
       Upload → Score → Explain → HITL Review → Retrain → Results Explorer


📌 14. Author

Mukesh Kumar
Generative AI & ML Engineer
Email: chaharmukesh518@gmail.com