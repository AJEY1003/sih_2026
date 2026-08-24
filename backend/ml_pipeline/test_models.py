import os
import pickle
import pandas as pd

def split_and_strip(x):
    return [s.strip() for s in x.split(',')]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def test_anomaly_model():
    print("\n--- Testing Task 2: Fraud & Anomaly Detection Model ---")

    # 1. Load the model, scaler, and preprocessing metadata (feature list +
    # vendor medians) - the model now expects 7 features (v2: added
    # Progress_Gap and the two vendor-missing indicator flags), so we build
    # the test frame from the saved feature list instead of hardcoding it.
    model_path = os.path.join(MODEL_DIR, 'anomaly_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'anomaly_scaler.pkl')
    meta_path = os.path.join(MODEL_DIR, 'anomaly_meta.pkl')

    with open(model_path, 'rb') as f:
        anomaly_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    print("Model, Scaler, and metadata loaded successfully!")

    # 2. Create some test data (one normal project, one highly suspicious project)
    test_data = pd.DataFrame({
        'Cost_per_Unit': [500.0, 999999.0],              # Second project is extremely expensive per unit
        'Vendor_Avg_Cost_Overrun_Ratio': [1.05, 5.5],     # Second vendor overruns budget by 5x
        'Vendor_Avg_Timeline_Days': [120, 900],           # Second vendor takes 900 days on average
        'Days_to_Sanction': [30, 2],                      # Suspiciously fast sanction for the expensive project
        'Progress_Gap': [0.0, 0.9],                       # Second project: money released far ahead of physical progress
        'Vendor_Avg_Cost_Overrun_Ratio_missing': [False, False],
        'Vendor_Avg_Timeline_Days_missing': [False, False],
    })[meta['features']]

    print("Test Input Data:")
    print(test_data)

    # 3. Scale and Predict
    X_scaled = scaler.transform(test_data)
    predictions = anomaly_model.predict(X_scaled)
    
    # Output: 1 for normal (inliers), -1 for anomalies (outliers)
    for i, pred in enumerate(predictions):
        status = "NORMAL" if pred == 1 else "ANOMALY (FRAUD RISK)"
        print(f"Project {i+1} Prediction: {status}")

def test_compliance_model():
    print("\n--- Testing Task 4: Risk & Compliance Scoring Model ---")
    
    # 1. Load the Pipeline (preprocessor + model)
    model_path = os.path.join(MODEL_DIR, 'compliance_risk_model.pkl')
    with open(model_path, 'rb') as f:
        compliance_pipeline = pickle.load(f)
        
    print("Compliance Pipeline loaded successfully!")
    
    # 2. Create test data (one good compliance, one bad compliance)
    test_data = pd.DataFrame({
        'Compliance_Score': [95.5, 40.0],
        'NOC_Status': ['Received', 'Pending'],
        'Documents_Submitted': ['Site Plan, NOC, Utilization Cert', 'Site Plan']
    })
    
    print("Test Input Data:")
    print(test_data)
    
    # 3. Predict Overall Risk Score
    predictions = compliance_pipeline.predict(test_data)
    
    for i, score in enumerate(predictions):
        print(f"Project {i+1} Predicted Risk Score: {score:.2f} / 100")

def test_hybrid_engine():
    print("\n--- Testing Hybrid Risk Engine (predict_project_risk) ---")
    import json
    from feature_engineering import build_master_dataset
    from hybrid_risk_engine import predict_project_risk

    df = build_master_dataset()

    # 1. A known project, looked up purely by Work_ID
    sample_id = df["Work_ID"].iloc[0]
    print(f"\nScoring known project by Work_ID: {sample_id}")
    print(json.dumps(predict_project_risk({"Work_ID": sample_id}), indent=2, default=str))

    # 2. A synthetic brand-new project with deliberately suspicious fields:
    #    money almost fully disbursed while still at an early Work_Status,
    #    pending NOC, no documents, and a description copied from an
    #    existing project by the same vendor/constituency to trigger the
    #    SBERT duplicate check.
    template = df.iloc[0]
    new_project = {
        "Work_ID": "TEST/NEW/0001",
        "Vendor_Name": template["Vendor_Name"],
        "Constituency": template["Constituency"],
        "State": template["State"],
        "Constituency_ID": template["Constituency_ID"],
        "Work_Description": template["Work_Description"],  # near-duplicate on purpose
        "Work_Status": "Vendor Identification",
        "Sanction_Amount": 500000,
        "Disbursed_Amount": 480000,  # heavily disbursed despite early status
        "Cost_per_Unit": 999999.0,
        "Days_to_Sanction": 2,
        "Compliance_Score": 20.0,
        "NOC_Status": "Pending",
        "Documents_Submitted": "",
        "Issues_Encountered": "Vendor issues",
    }
    print("\nScoring a synthetic high-risk new project (not in dataset):")
    print(json.dumps(predict_project_risk(new_project), indent=2, default=str))


def test_investigation_report():
    print("\n--- Testing generate_investigation_report (GIS investigation flow) ---")
    import json
    import pandas as pd
    from hybrid_risk_engine import generate_investigation_report

    # Pick a real project with a known near-duplicate description flag, so
    # the WHY panel actually has something to show end-to-end.
    dup = pd.read_csv(os.path.join(MODEL_DIR, "duplicate_scores.csv"))
    work_id = dup.sort_values("duplicate_risk_score", ascending=False).iloc[0]["Work_ID"]

    print(f"Generating investigation report for: {work_id}")
    print(json.dumps(generate_investigation_report(work_id), indent=2, default=str))


if __name__ == "__main__":
    test_anomaly_model()
    test_compliance_model()
    test_hybrid_engine()
    test_investigation_report()
    print("\nTests completed successfully!")
