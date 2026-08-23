import pandas as pd
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_DIR = r"e:\sih\output_schema"
MODEL_DIR = r"e:\sih\backend\ml_pipeline\models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_anomaly_model():
    print("Loading datasets for Task 2 (Fraud & Anomaly Detection)...")
    try:
        master_df = pd.read_csv(os.path.join(DATA_DIR, 'works_master.csv'))
        vendor_df = pd.read_csv(os.path.join(DATA_DIR, 'vendor_dimension.csv'))
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return

    print("Merging data...")
    # We join vendor data to get Vendor_Avg_Cost_Overrun_Ratio & Vendor_Avg_Timeline_Days
    df = pd.merge(master_df, vendor_df, on='Vendor_Name', how='left')

    # Select features that are key indicators of anomalies
    features = [
        'Cost_per_Unit',
        'Vendor_Avg_Cost_Overrun_Ratio',
        'Vendor_Avg_Timeline_Days',
        'Days_to_Sanction'
    ]

    # Drop missing values strictly for these features
    df_clean = df.dropna(subset=features).copy()
    print(f"Training on {len(df_clean)} rows out of {len(df)} after dropping NaNs in feature columns.")

    X = df_clean[features]

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Isolation Forest...")
    # contamination=0.05 means we assume ~5% of the data might be anomalous/fraudulent
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    # Save the model and scaler
    model_path = os.path.join(MODEL_DIR, 'anomaly_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'anomaly_scaler.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model successfully saved to {model_path}")
    print(f"Scaler successfully saved to {scaler_path}")

if __name__ == "__main__":
    train_anomaly_model()
