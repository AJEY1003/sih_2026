import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
DATA_DIR = r"e:\sih\output_schema"
MODEL_DIR = r"e:\sih\backend\ml_pipeline\models"
os.makedirs(MODEL_DIR, exist_ok=True)

def split_and_strip(x):
    return [s.strip() for s in x.split(',')]

def train_compliance_model():
    print("Loading datasets for Task 4 (Risk & Compliance Scoring)...")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'compliance_and_ml.csv'))
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return

    # Target variable
    y = df['Overall_Fraud_Risk_Score']

    # Features
    # Fill NaN for documents just in case
    df['Documents_Submitted'] = df['Documents_Submitted'].fillna("None")
    df['NOC_Status'] = df['NOC_Status'].fillna("Unknown")
    df['Compliance_Score'] = df['Compliance_Score'].fillna(df['Compliance_Score'].mean())

    X = df[['Compliance_Score', 'NOC_Status', 'Documents_Submitted']]

    print("Building Preprocessing Pipeline...")
    # Preprocessing
    # CountVectorizer is perfect for parsing comma-separated strings like "Site Plan, NOC"
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['NOC_Status']),
            ('doc', CountVectorizer(tokenizer=split_and_strip, token_pattern=None), 'Documents_Submitted'),
            ('num', 'passthrough', ['Compliance_Score'])
        ]
    )

    print("Building Random Forest Pipeline...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10, n_jobs=-1)) # n_jobs=-1 for speed
    ])

    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating Model...")
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R^2: {r2:.2f}")

    print("Saving Model...")
    model_path = os.path.join(MODEL_DIR, 'compliance_risk_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Pipeline successfully saved to {model_path}")

if __name__ == "__main__":
    train_compliance_model()
