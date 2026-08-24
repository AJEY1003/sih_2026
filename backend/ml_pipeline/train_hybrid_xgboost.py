"""
Trains the XGBoost component of the Hybrid Risk Engine.

Important: this does NOT train against Overall_Fraud_Risk_Score. That column
was verified (see train_task4_compliance.py docstring) to have |correlation|
< 0.03 with every available feature - it behaves as noise in this dataset,
so fitting any model to it - XGBoost included - would just be an expensive
way to memorize noise, not a real upgrade over the Task 4 baseline.

Instead, XGBoost is trained as a fast/cheap approximator of the rule-based
composite risk score (risk_components.RISK_WEIGHTS) computed from signals
that ARE real: Isolation Forest anomaly score, compliance rules, contractor/
vendor risk, and spatial zone risk. It deliberately does NOT see the SBERT
duplicate-detection score or the component scores themselves as inputs -
only cheap raw fields - so it can serve as a fast pre-screening estimate
that doesn't require running SBERT per prediction, and so its
feature_importances_ point at actionable raw fields (Compliance_Score,
Progress_Gap, Vendor_Risk_Score, ...) instead of trivially reconstructing a
known linear formula. Expect R^2 to land well above Task 4's ~0.00 (most of
the composite's weight - 80%, everything but the 20% duplicate-detection
share - IS derivable from these raw fields) but below 1.0 - the gap is
exactly the duplicate-detection information it's not given.

hybrid_risk_engine.py reports this model's prediction alongside the live
rule-based composite as a cross-check, not as a replacement for it.
"""
import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from feature_engineering import build_master_dataset, build_xgb_feature_frame
from risk_components import (
    RISK_WEIGHTS, compliance_rule_score, contractor_risk_score, spatial_risk_score,
    anomaly_scores_from_iforest,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_anomaly_artifacts():
    with open(os.path.join(MODEL_DIR, "anomaly_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "anomaly_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "anomaly_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return model, scaler, meta


def train_hybrid_model():
    print("Building master dataset (works + vendor + compliance + geography)...")
    df = build_master_dataset()

    print("Loading Task 2 Isolation Forest to compute live anomaly scores...")
    if_model, if_scaler, if_meta = load_anomaly_artifacts()
    from feature_engineering import impute_vendor_features
    df, _ = impute_vendor_features(df, if_meta["vendor_medians"])
    X_if = df[if_meta["features"]].fillna(0)
    df["anomaly_score"] = anomaly_scores_from_iforest(if_model, if_scaler.transform(X_if))

    print("Computing rule-based component scores (compliance / contractor / spatial)...")
    df["compliance_score_signal"] = compliance_rule_score(df)
    df["contractor_score_signal"] = contractor_risk_score(df)
    df["spatial_score_signal"] = spatial_risk_score(df)

    dup_path = os.path.join(MODEL_DIR, "duplicate_scores.csv")
    if os.path.exists(dup_path):
        print("Merging precomputed SBERT duplicate-detection scores...")
        dup = pd.read_csv(dup_path)[["Work_ID", "duplicate_risk_score"]]
        df = df.merge(dup, on="Work_ID", how="left")
        df["duplicate_risk_score"] = df["duplicate_risk_score"].fillna(0.0)
    else:
        print("WARNING: duplicate_scores.csv not found yet (run build_duplicate_index.py first). "
              "Using 0 for the duplicate-detection signal - composite will be trained without it.")
        df["duplicate_risk_score"] = 0.0

    df["composite_risk_score"] = (
        RISK_WEIGHTS["anomaly"] * df["anomaly_score"]
        + RISK_WEIGHTS["compliance"] * df["compliance_score_signal"]
        + RISK_WEIGHTS["contractor"] * df["contractor_score_signal"]
        + RISK_WEIGHTS["spatial"] * df["spatial_score_signal"]
        + RISK_WEIGHTS["duplicate"] * df["duplicate_risk_score"]
    )
    print("Composite risk score distribution:")
    print(df["composite_risk_score"].describe())

    print("Building XGBoost input feature frame (raw fields only)...")
    X, medians = build_xgb_feature_frame(df)
    y = df["composite_risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost regressor...")
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R^2: {r2:.3f}  (vs Task 4 baseline's ~0.00)")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 feature importances:")
    print(importances.head(10))

    model_path = os.path.join(MODEL_DIR, "hybrid_xgboost_model.pkl")
    meta_path = os.path.join(MODEL_DIR, "hybrid_xgboost_meta.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(meta_path, "wb") as f:
        pickle.dump({"columns": list(X.columns), "vendor_medians": medians,
                      "test_r2": r2, "test_mse": mse}, f)
    print(f"\nModel saved to {model_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    train_hybrid_model()
