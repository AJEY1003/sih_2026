import os
import pickle
import pandas as pd

def split_and_strip(x):
    return [s.strip() for s in x.split(',')]

MODEL_DIR = r"e:\sih\backend\ml_pipeline\models"

def test_anomaly_model():
    print("\n--- Testing Task 2: Fraud & Anomaly Detection Model ---")
    
    # 1. Load the model and scaler
    model_path = os.path.join(MODEL_DIR, 'anomaly_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'anomaly_scaler.pkl')
    
    with open(model_path, 'rb') as f:
        anomaly_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    print("Model and Scaler loaded successfully!")
    
    # 2. Create some test data (one normal project, one highly suspicious project)
    test_data = pd.DataFrame({
        'Cost_per_Unit': [500.0, 999999.0], # Second project is extremely expensive per unit
        'Vendor_Avg_Cost_Overrun_Ratio': [1.05, 5.5], # Second vendor overruns budget by 5x
        'Vendor_Avg_Timeline_Days': [120, 900], # Second vendor takes 900 days on average
        'Days_to_Sanction': [30, 2] # Suspiciously fast sanction for the expensive project
    })
    
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

if __name__ == "__main__":
    test_anomaly_model()
    test_compliance_model()
    print("\nTests completed successfully!")
