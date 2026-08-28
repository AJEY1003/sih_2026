# MPLADS Analytics & Monitoring Platform (SIH 2026)

## Overview
The Members of Parliament Local Area Development Scheme (MPLADS) involves large-scale fund utilization and execution of thousands of works across the country. Given the volume and complexity of financial and project-related data, this AI-powered monitoring and analytics platform leverages machine learning to detect trends, anomalies in expenditure patterns, and potential fraud.

This project was designed to enhance transparency, accountability, and effective monitoring of MPLADS works by catching inefficiencies early.

## Project Structure

The project currently consists of two major components: a robust ETL Pipeline for data normalization, and an advanced Machine Learning backend.

### 1. Data Normalization & ETL (`prepare_dataset.py`)
The raw data for Lok Sabha and Rajya Sabha (Recommended, Sanctioned, Completed, and Expenditure) comes in disparate, highly denormalized CSVs with structural inconsistencies.

The `prepare_dataset.py` script transforms these raw files into a highly structured **5-Table Star Schema**:
- **`works_master.csv`**: The central fact table tracking timelines, budgets, and cost overruns.
- **`mp_dimension.csv`**: Aggregated performance metrics for individual MPs.
- **`vendor_dimension.csv`**: Historical timeline and cost overrun averages per Vendor.
- **`geography_dimension.csv`**: Spatial and geographic cluster mapping.
- **`compliance_and_ml.csv`**: NLP and Risk features including documents submitted, NOC status, and inspection summaries.

### 2. Machine Learning Pipeline (`backend/ml_pipeline/`)
The `ml_pipeline` directory contains the Python scripts and pre-trained `.pkl` models required for the Backend API to perform real-time predictive analytics.

#### Models included:
- **Fraud & Anomaly Detection (`anomaly_model.pkl`)**
  - **Algorithm:** Isolation Forest, trained via `train_task2_anomaly.py`.
  - **Features:** `Cost_per_Unit`, `Vendor_Avg_Cost_Overrun_Ratio`, `Vendor_Avg_Timeline_Days`, `Days_to_Sanction`, plus engineered `Progress_Gap` (disbursement progress vs. expected physical progress from `Work_Status` - flags money released ahead of actual work) and `*_missing` indicator flags for vendors with no cost/timeline history.
  - Missing vendor history is now median-imputed with an indicator flag rather than dropped, so the model trains on 100% of `works_master.csv` (previously ~24% after a strict `dropna`).

- **Risk & Compliance Scoring baseline (`compliance_risk_model.pkl`)**
  - **Algorithm:** Random Forest Regressor & NLP CountVectorizer Pipeline (`train_task4_compliance.py`).
  - Predicts `Overall_Fraud_Risk_Score` from `Compliance_Score`, `NOC_Status`, `Documents_Submitted`.
  - **Kept as a baseline only:** an exhaustive correlation check found `Overall_Fraud_Risk_Score` has |r| < 0.03 with every feature across all 5 tables - it behaves as noise in this dataset, which is why this model tests at R² ≈ 0. Not used as an authoritative risk source; see the Hybrid Risk Engine below.

- **Hybrid Risk Engine (`hybrid_risk_engine.py`)**
  - Single entry point: `predict_project_risk(project_data)` and `generate_investigation_report(work_id)`.
  - Blends five real, auditable signals into an explainable 0-100 score: Isolation Forest anomaly score (25%), compliance rules (20%), contractor/vendor risk (20%), spatial zone risk (15%, from `geography_dimension.csv`), and SBERT semantic duplicate-billing detection (20%, scoped to same vendor + constituency).
  - An XGBoost model (`train_hybrid_xgboost.py`) trained on raw fields approximates that composite for fast pre-screening (test R² = 0.80) and is reported alongside as a cross-check, not blended into the primary score.
  - `generate_investigation_report(work_id)` returns the full GIS "select project → why is it risky → what else to check" payload: color-coded risk factors, nearby similar works, contractor history, historical cost benchmark, and a recommended action (e.g. `AUDIT / FIELD VERIFICATION` for Critical-risk projects).

### 3. FastAPI Backend (`backend/app/`)
Wraps the ETL star schema and the Hybrid Risk Engine for live HTTP access.

Run it from `backend/`:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive docs (Swagger UI): `http://localhost:8000/docs`

Routes:
| Method & Path | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /api/risk/predict` | Score a project 0-100 via the Hybrid Risk Engine - pass `{"Work_ID": ...}` for a known project, or raw fields for a new/what-if one |
| `GET /api/risk/investigate/{work_id}` | Full GIS investigation payload: why it's risky, nearby similar works, contractor history, cost benchmark, recommended action |
| `GET /api/works` | List works - filter by state/constituency/MP/vendor/status, substring `search`, amount range, sort, pagination |
| `GET /api/works/{work_id}` | One work merged with vendor, geography and compliance data |
| `GET /api/mps` | List MPs - search, sort, pagination |
| `GET /api/mps/{mp_name}` | MP performance detail plus their works |
| `GET /api/vendors` | List vendors - search, sort, pagination |
| `GET /api/vendors/{vendor_name}` | Vendor track record plus a sample of recent works |
| `GET /api/geography` | List constituencies with coordinates and zone fraud risk (for map views) |
| `GET /api/geography/{state}/{constituency_id}` | One constituency's geo + zone risk detail |
| `GET /api/stats/overview` | Dashboard totals: works/MPs/vendors counts, amounts, status/state breakdowns |

Note: `Work_ID` values contain literal slashes (e.g. `WS/MP005/2024-2025/145074`) - pass them as-is in the URL path, no percent-encoding needed (`GET /api/works/WS/MP005/2024-2025/145074` works directly).

## Setup & Installation

### Requirements
- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `sentence-transformers`

### Running the ML Tests
To verify that the machine learning models deserialize properly and can process real-time input:
```bash
python backend/ml_pipeline/test_models.py
```

### Regenerating models
Run once, in order, from `backend/ml_pipeline/`:
```bash
python train_task2_anomaly.py        # Isolation Forest (~seconds)
python train_task4_compliance.py     # baseline regressor (~seconds)
python build_duplicate_index.py      # SBERT embedding index (~9 min on CPU)
python train_hybrid_xgboost.py       # hybrid XGBoost cross-check model (~1 min)
```
`models/doc_embeddings.pkl` (the raw SBERT embedding index, ~124MB) is gitignored since it exceeds GitHub's 100MB file limit and is fully reproducible via `build_duplicate_index.py` - regenerate it locally before using the Hybrid Risk Engine on a brand-new (not-yet-in-dataset) project description. `models/duplicate_scores.csv` (the precomputed lookup table for known `Work_ID`s) is committed, so looking up existing projects works without regenerating anything.

## Next Steps
- Integration of a Web Application / Dashboard (Next.js or Vanilla HTML/JS) to visualize anomalies, geospatial trends, and vendor risk profiles, consuming the FastAPI backend above.
