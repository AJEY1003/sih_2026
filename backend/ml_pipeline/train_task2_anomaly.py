"""
Task 2: Fraud & Anomaly Detection (Isolation Forest) - unsupervised.

v2 changes vs the original version:
  - Missing-value handling no longer drops rows. The original dropna()
    approach discarded ~76% of works_master (59k/77k rows) because
    Vendor_Avg_Cost_Overrun_Ratio and Vendor_Avg_Timeline_Days are null for
    ~70%/~42% of vendors (new/thin-history vendors). We now impute with the
    training median and add *_missing indicator flags instead - a vendor
    with no track record is itself a signal worth handing to the model,
    not a reason to throw the row away. See feature_engineering.impute_vendor_features.
  - Added Progress_Gap (feature_engineering.add_progress_gap): financial
    disbursement progress minus expected physical progress implied by
    Work_Status. Large positive values mean money was released far ahead of
    physical progress - a concrete fraud signal the original 4 features
    couldn't see.
  - Saves anomaly_meta.pkl alongside the model/scaler: the feature list and
    the imputation medians learned here, so a single new record can be
    transformed identically at inference time (see hybrid_risk_engine.py).
  - Also saves a per-feature reference distribution (sorted training values
    for Cost_per_Unit, Vendor_Avg_Timeline_Days, Days_to_Sanction,
    Progress_Gap, Vendor_Avg_Cost_Overrun_Ratio), so the investigation
    report can tell a "cost anomaly" apart from a "timeline anomaly"
    instead of only reporting one blended anomaly score.
"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from feature_engineering import DATA_DIR, add_progress_gap, impute_vendor_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Cost_per_Unit",
    "Vendor_Avg_Cost_Overrun_Ratio",
    "Vendor_Avg_Timeline_Days",
    "Days_to_Sanction",
    "Progress_Gap",
    "Vendor_Avg_Cost_Overrun_Ratio_missing",
    "Vendor_Avg_Timeline_Days_missing",
]


def train_anomaly_model():
    print("Loading datasets for Task 2 (Fraud & Anomaly Detection)...")
    try:
        master_df = pd.read_csv(os.path.join(DATA_DIR, "works_master.csv"))
        vendor_df = pd.read_csv(os.path.join(DATA_DIR, "vendor_dimension.csv"))
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return

    print("Merging data...")
    df = pd.merge(master_df, vendor_df, on="Vendor_Name", how="left")

    print("Engineering features (Progress_Gap, vendor missing-value imputation)...")
    df = add_progress_gap(df)
    df, medians = impute_vendor_features(df)

    # Cost_per_Unit and Days_to_Sanction have no missing values in this
    # dataset (verified), but guard anyway rather than silently dropping.
    before = len(df)
    df_clean = df.dropna(subset=["Cost_per_Unit", "Days_to_Sanction", "Progress_Gap"]).copy()
    print(f"Training on {len(df_clean)} rows out of {before} "
          f"({len(df_clean) / before:.1%} retained, vs ~24% with the old dropna-all-features approach).")

    X = df_clean[FEATURES]

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Isolation Forest...")
    # contamination=0.05 means we assume ~5% of the data might be anomalous/fraudulent
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    # Reference distribution of raw score_samples() over the training set,
    # so a single new project at inference time can be converted to a
    # percentile-based 0-100 risk score the same way risk_components.
    # anomaly_scores_from_iforest ranks a whole batch (see hybrid_risk_engine.py).
    reference_scores = np.sort(model.score_samples(X_scaled))

    # Per-feature reference distributions for the investigation report's
    # "cost anomaly" vs "timeline anomaly" breakdown (see risk_components.py).
    per_feature_reference = {
        col: np.sort(df_clean[col].values)
        for col in ["Cost_per_Unit", "Vendor_Avg_Timeline_Days", "Days_to_Sanction",
                    "Progress_Gap", "Vendor_Avg_Cost_Overrun_Ratio"]
    }

    # Save the model, scaler, and preprocessing metadata
    model_path = os.path.join(MODEL_DIR, "anomaly_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "anomaly_scaler.pkl")
    meta_path = os.path.join(MODEL_DIR, "anomaly_meta.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(meta_path, "wb") as f:
        pickle.dump({"features": FEATURES, "vendor_medians": medians,
                      "reference_scores": reference_scores,
                      "per_feature_reference": per_feature_reference}, f)

    print(f"Model successfully saved to {model_path}")
    print(f"Scaler successfully saved to {scaler_path}")
    print(f"Preprocessing metadata saved to {meta_path}")


if __name__ == "__main__":
    train_anomaly_model()
