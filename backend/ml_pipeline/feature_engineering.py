"""
Shared feature-engineering utilities for the SIH fraud/risk pipeline.

Every training script AND the live hybrid_risk_engine.predict_project_risk()
import from here, so a single project record and a full training dataframe
are always transformed the exact same way. Do not duplicate this logic
elsewhere - if a new engineered feature is needed, add it here.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "output_schema")


# ---------------------------------------------------------------------------
# Work_Status -> expected physical-progress fraction (0 = nothing done yet,
# 1 = fully complete). This ordering is a domain assumption about the project
# lifecycle (estimation -> sanction -> vendor assigned -> execution/inspection
# -> partial -> complete), NOT derived from the financial columns - if it were
# derived from Disbursed_Amount itself, Progress_Gap would be circular and
# could never flag "money released before proportional physical progress".
# ---------------------------------------------------------------------------
STATUS_PROGRESS_MAP = {
    "Time Estimation": 0.0,
    "Sanction": 0.15,
    "Vendor Identification": 0.35,
    "Physical Inspection": 0.65,
    "Work partially Completed": 0.85,
    "Work Completed": 1.0,
}

REQUIRED_DOCS = ["Site Plan", "Utilization Cert", "NOC"]
ISSUE_TOKENS = ["Weather delays", "Vendor issues"]


def _split_tokens(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [t.strip() for t in str(cell).split(",") if t.strip()]


def add_progress_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'Progress_Gap' = financial disbursement progress minus expected
    physical progress implied by Work_Status. Positive => money released
    disproportionately ahead of physical progress (classic red flag).
    Requires Disbursed_Amount, Sanction_Amount, Work_Status columns."""
    df = df.copy()
    sanction = df["Sanction_Amount"].replace(0, np.nan)
    financial_progress = (df["Disbursed_Amount"] / sanction).clip(lower=0, upper=1.5)
    financial_progress = financial_progress.fillna(0.0)
    physical_progress = df["Work_Status"].map(STATUS_PROGRESS_MAP).fillna(0.0)
    df["Progress_Gap"] = financial_progress - physical_progress
    return df


def add_document_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-label binarizes Documents_Submitted / Issues_Encountered into
    Doc_<name> / Issue_<name> boolean columns, plus count summaries."""
    df = df.copy()
    docs = df["Documents_Submitted"].apply(_split_tokens)
    for doc in REQUIRED_DOCS:
        df[f"Doc_{doc.replace(' ', '_')}"] = docs.apply(lambda toks, d=doc: d in toks)
    df["Docs_Submitted_Count"] = docs.apply(len)
    df["Docs_Missing_Count"] = len(REQUIRED_DOCS) - df["Docs_Submitted_Count"].clip(upper=len(REQUIRED_DOCS))

    issues = df["Issues_Encountered"].apply(_split_tokens)
    for issue in ISSUE_TOKENS:
        df[f"Issue_{issue.replace(' ', '_')}"] = issues.apply(lambda toks, i=issue: i in toks)
    df["Issues_Count"] = issues.apply(len)
    return df


def impute_vendor_features(df: pd.DataFrame, medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """Instead of dropping ~76% of rows with dropna() (the old Task 2
    approach), keep every row: add a *_missing indicator (a vendor with no
    track record is itself a risk signal) and impute the value with the
    training-set median. `medians` can be passed in at inference time so a
    single new row is imputed with the exact values learned at train time."""
    df = df.copy()
    cols = ["Vendor_Avg_Cost_Overrun_Ratio", "Vendor_Avg_Timeline_Days"]
    computed_medians = {}
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[f"{col}_missing"] = df[col].isna()
        median_val = medians[col] if medians and col in medians else df[col].median()
        computed_medians[col] = median_val
        df[col] = df[col].fillna(median_val)
    return df, computed_medians


XGB_NUMERIC_COLS = [
    "Cost_per_Unit", "Days_to_Sanction", "Progress_Gap",
    "Vendor_Avg_Cost_Overrun_Ratio", "Vendor_Avg_Timeline_Days",
    "Vendor_Avg_Cost_Overrun_Ratio_missing", "Vendor_Avg_Timeline_Days_missing",
    "Vendor_Risk_Score", "Vendor_Monopoly_Score", "Vendor_Quality_Rating",
    "Vendor_Total_Projects_Count", "Compliance_Score", "Zone_Fraud_Risk_Score",
    "Docs_Submitted_Count", "Docs_Missing_Count", "Issues_Count",
]
XGB_BOOL_COLS = ["Doc_Site_Plan", "Doc_Utilization_Cert", "Doc_NOC",
                  "Issue_Weather_delays", "Issue_Vendor_issues", "Calamity_Flag"]
XGB_CAT_COLS = ["NOC_Status", "House"]


def build_xgb_feature_frame(df: pd.DataFrame, medians: dict = None) -> tuple[pd.DataFrame, dict]:
    """Builds the raw-field feature matrix the hybrid XGBoost model trains
    and predicts on. Only uses fields cheap to obtain (no SBERT / no
    derived component scores) so it can pre-screen quickly. Returns
    (X, medians) - pass `medians` back in at inference time so a single new
    row is imputed identically to training."""
    df = df.copy()
    if "Vendor_Avg_Cost_Overrun_Ratio_missing" not in df.columns:
        df, medians = impute_vendor_features(df, medians)
    if "Progress_Gap" not in df.columns:
        df = add_progress_gap(df)
    if "Docs_Submitted_Count" not in df.columns:
        df = add_document_flags(df)

    for col in XGB_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[XGB_NUMERIC_COLS].copy()

    for col in XGB_BOOL_COLS:
        X[col] = df[col].astype("boolean").fillna(False).astype(int) if col in df.columns else 0

    for col in XGB_CAT_COLS:
        series = df[col].fillna("Missing") if col in df.columns else pd.Series("Missing", index=df.index)
        dummies = pd.get_dummies(series, prefix=col)
        X = pd.concat([X, dummies], axis=1)

    return X, (medians or {})


def align_xgb_features(X: pd.DataFrame, trained_columns: list) -> pd.DataFrame:
    """Aligns a (possibly single-row) feature frame to the exact column set
    the XGBoost model was trained on - required because pd.get_dummies on
    one row won't produce every category column the training data had."""
    return X.reindex(columns=trained_columns, fill_value=0)


def build_master_dataset(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Loads and merges all five output_schema tables into one row-per-work
    dataframe with every engineered feature applied. This is the single
    source of truth used by all training scripts and by the hybrid engine
    when it needs to look up a project's full context by Work_ID."""
    works = pd.read_csv(os.path.join(data_dir, "works_master.csv"))
    vendor = pd.read_csv(os.path.join(data_dir, "vendor_dimension.csv"))
    compliance = pd.read_csv(os.path.join(data_dir, "compliance_and_ml.csv"))
    geo = pd.read_csv(os.path.join(data_dir, "geography_dimension.csv"))

    df = works.merge(vendor, on="Vendor_Name", how="left")
    df = df.merge(compliance, on="Work_ID", how="left")
    df = df.merge(geo, on=["State", "Constituency", "Constituency_ID"], how="left")

    df = add_progress_gap(df)
    df = add_document_flags(df)
    return df
