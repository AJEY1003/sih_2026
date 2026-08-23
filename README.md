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
  - **Algorithm:** Isolation Forest
  - **Purpose:** Identifies statistical anomalies and outliers in Cost per Unit, Sanctioning Delays, and Vendor History to flag potentially fraudulent projects without needing explicitly labeled fraud data.
  
- **Overall Risk & Compliance Scoring (`compliance_risk_model.pkl`)**
  - **Algorithm:** Random Forest Regressor & NLP CountVectorizer Pipeline
  - **Purpose:** Decodes raw comma-separated text (e.g., "Site Plan, NOC") and predicts an `Overall_Fraud_Risk_Score` based on historical compliance.

## Setup & Installation

### Requirements
- Python 3.9+
- `pandas`, `scikit-learn`, `numpy`

### Running the ML Tests
To verify that the machine learning models deserialize properly and can process real-time input:
```bash
python backend/ml_pipeline/test_models.py
```

## Next Steps
- Integration of a Web Application / Dashboard (Next.js or Vanilla HTML/JS) to visualize anomalies, geospatial trends, and vendor risk profiles.
- Connecting the pre-trained `.pkl` models to a FastAPI or Flask backend for live inference.
