import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import glob
import random

print("Starting Data Preparation Script...")

LOKSABHA_DIR = r"e:\sih\LOKSABHA"
RAJYASABHA_DIR = r"e:\sih\RAJYASABHA\RAJYASABHA"
OUTPUT_DIR = r"e:\sih\output_schema"

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)

def extract_work_id(text):
    if pd.isna(text):
        return np.nan
    cleaned = str(text).replace('\t', '').replace('\r', '').replace('\n', '')
    match = re.search(r'(WS/\s*[A-Za-z0-9]+/\d{4}-\d{4}/\d+)', cleaned)
    if match:
        return match.group(1).replace(" ", "")
    parts = cleaned.split('-')
    if len(parts) > 1:
        return parts[0].replace(" ", "").strip()
    return cleaned.strip()

def clean_currency(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).replace(',', '').replace('₹', '').strip()
    try:
        return float(val_str)
    except:
        return 0.0

def load_data(base_dir, house_name):
    print(f"Loading data for {house_name} from {base_dir}")
    try:
        rec_path = glob.glob(os.path.join(base_dir, '*Works Recommended*.csv'))[0]
        san_path = glob.glob(os.path.join(base_dir, '*Works Sanctioned*.csv'))[0]
        com_path = glob.glob(os.path.join(base_dir, '*Works Completed*.csv'))[0]
        exp_path = glob.glob(os.path.join(base_dir, '*Expenditure*.csv'))[0]
    except IndexError:
        print(f"Skipping {house_name}: Required CSV files not found in {base_dir}")
        return None, None, None, None

    df_rec = pd.read_csv(rec_path)
    df_san = pd.read_csv(san_path)
    df_com = pd.read_csv(com_path)
    df_exp = pd.read_csv(exp_path)

    # Robust column renaming
    def rename_col(col_name):
        c = col_name.strip().upper()
        if 'WORK DESCRIPTION' in c: return 'Work_Description'
        if 'RECOMMENDED DATE' in c: return 'Recommended_Date'
        if 'RECOMMENDED AMOUNT' in c: return 'Recommended_Amount'
        if 'HON\'BLE' in c or 'MEMBERS OF PARLIAMENT' in c: return 'MP_Name'
        if 'SANCTION DATE' in c: return 'Sanction_Date'
        if 'SANCTION AMOUNT' in c: return 'Sanction_Amount'
        if 'WORK STATUS' in c: return 'Work_Status'
        if 'COMPLETION DATE' in c or 'COMPLETE DATE' in c: return 'Completion_Date'
        if 'WORK' in c and 'DESCRIPTION' not in c and 'ID' not in c and 'CATEGORY' not in c and 'STATUS' not in c: return 'Work_String'
        return col_name

    df_rec.rename(columns=lambda x: rename_col(x), inplace=True)
    df_san.rename(columns=lambda x: rename_col(x), inplace=True)
    df_com.rename(columns=lambda x: rename_col(x), inplace=True)

    df_rec['Recommended_Amount'] = df_rec.get('Recommended_Amount', pd.Series(0, index=df_rec.index)).apply(clean_currency)
    df_san['Sanction_Amount'] = df_san.get('Sanction_Amount', pd.Series(0, index=df_san.index)).apply(clean_currency)
    
    df_exp_amount_col = [c for c in df_exp.columns if 'Amount' in c and '₹' in c]
    if df_exp_amount_col:
        df_exp['Disbursed_Amount'] = df_exp[df_exp_amount_col[0]].apply(clean_currency)
    else:
        df_exp['Disbursed_Amount'] = 0.0
        
    df_rec['Work_ID'] = df_rec.get('Work_String', pd.Series()).apply(extract_work_id)
    df_san['Work_ID'] = df_san.get('Work_String', pd.Series()).apply(extract_work_id)
    df_com['Work_ID'] = df_com.get('Work_String', pd.Series()).apply(extract_work_id)
    
    if 'Work ID' in df_exp.columns:
        df_exp['Work_ID'] = df_exp['Work ID'].apply(lambda x: str(x).replace(" ", "").replace("\t", "").strip() if not pd.isna(x) else x)
    else:
        df_exp['Work_ID'] = None

    df_rec['House'] = house_name
    df_rec['Elected_or_Nominated'] = "Nominated" if house_name == "Rajya Sabha" else "Elected"
    
    return df_rec, df_san, df_com, df_exp

dfs_rec, dfs_san, dfs_com, dfs_exp = [], [], [], []

# LOKSABHA
r, s, c, e = load_data(LOKSABHA_DIR, 'Lok Sabha')
if r is not None:
    dfs_rec.append(r); dfs_san.append(s); dfs_com.append(c); dfs_exp.append(e)

# RAJYASABHA
r, s, c, e = load_data(RAJYASABHA_DIR, 'Rajya Sabha')
if r is not None:
    dfs_rec.append(r); dfs_san.append(s); dfs_com.append(c); dfs_exp.append(e)

if not dfs_rec:
    print("No data loaded. Exiting.")
    exit(1)

df_rec_all = pd.concat(dfs_rec, ignore_index=True).drop_duplicates(subset=['Work_ID'])
df_san_all = pd.concat(dfs_san, ignore_index=True).drop_duplicates(subset=['Work_ID'])
df_com_all = pd.concat(dfs_com, ignore_index=True).drop_duplicates(subset=['Work_ID'])
df_exp_all = pd.concat(dfs_exp, ignore_index=True)
# Clean overlapping columns to prevent _x, _y suffixes during merge
for col in ['Sanction_Date', 'Sanction_Amount', 'Work_Status', 'Completion_Date']:
    if col in df_rec_all.columns:
        df_rec_all = df_rec_all.drop(columns=[col])
    if col in df_san_all.columns and col == 'Completion_Date':
        df_san_all = df_san_all.drop(columns=[col])

# 1. Base Joins
print("Joining tables...")
# Aggregate expenditures
exp_grouped = df_exp_all.groupby('Work_ID').agg(
    Disbursed_Amount=('Disbursed_Amount', 'sum'),
    Vendor_Name=('Vendor Name', lambda x: list(set(x.dropna()))[0] if len(set(x.dropna()))>0 else "Unknown")
).reset_index()

master = pd.merge(df_rec_all, df_san_all[['Work_ID', 'Sanction_Date', 'Sanction_Amount', 'Work_Status']], on='Work_ID', how='left')
master = pd.merge(master, df_com_all[['Work_ID', 'Completion_Date']], on='Work_ID', how='left')
master = pd.merge(master, exp_grouped, on='Work_ID', how='left')

master['Work_Status'] = master['Work_Status'].fillna('Recommended')
master['Disbursed_Amount'] = master['Disbursed_Amount'].fillna(0)
master['Sanction_Amount'] = master['Sanction_Amount'].fillna(0)
master['Vendor_Name'] = master['Vendor_Name'].fillna('Unknown')
master['Calamity_Flag'] = np.random.choice([True, False], size=len(master), p=[0.05, 0.95])

# Temporal Dates
master['Recommended_Date'] = pd.to_datetime(master['Recommended_Date'], format='mixed', dayfirst=True, errors='coerce')
master['Sanction_Date'] = pd.to_datetime(master['Sanction_Date'], format='mixed', dayfirst=True, errors='coerce')
master['Completion_Date'] = pd.to_datetime(master['Completion_Date'], format='mixed', dayfirst=True, errors='coerce')

MAX_DATE = pd.to_datetime('today')
master['Days_to_Sanction'] = (master['Sanction_Date'] - master['Recommended_Date']).dt.days
master['Days_to_Complete'] = (master['Completion_Date'] - master['Sanction_Date']).dt.days
master['Days_Pending'] = np.where(master['Completion_Date'].isna(), (MAX_DATE - master['Recommended_Date']).dt.days, np.nan)

# Financial & Gated Overruns
master['Cost_Overrun_Amount'] = np.where(master['Work_Status'].str.contains('Completed', na=False, case=False), master['Disbursed_Amount'] - master['Sanction_Amount'], np.nan)
master['Cost_Overrun_Ratio'] = np.where((master['Work_Status'].str.contains('Completed', na=False, case=False)) & (master['Sanction_Amount'] > 0), master['Disbursed_Amount'] / master['Sanction_Amount'], np.nan)

# Mock Extracted Work Scope
master['Work_Quantity'] = [random.randint(10, 1000) for _ in range(len(master))]
master['Work_Scope_Unit'] = [random.choice(["meter", "sqm", "number", "liter"]) for _ in range(len(master))]
master['Cost_per_Unit'] = np.where(master['Work_Quantity'] > 0, master['Disbursed_Amount'] / master['Work_Quantity'], 0)

print("Building Dimension Tables...")
# MP Dimension
mp_dim = master.groupby('MP_Name').agg(
    MP_Total_Allocation=('Recommended_Amount', 'sum'),
    MP_Total_Sanctioned=('Sanction_Amount', 'sum'),
    MP_Total_Disbursed=('Disbursed_Amount', 'sum'),
    MP_Total_Projects=('Work_ID', 'count'),
    MP_Completed_Projects=('Completion_Date', 'count')
).reset_index()
mp_dim['MP_Completion_Rate'] = (mp_dim['MP_Completed_Projects'] / mp_dim['MP_Total_Projects']) * 100
mp_dim['MP_Budget_Utilization_Percent'] = np.where(mp_dim['MP_Total_Allocation'] > 0, (mp_dim['MP_Total_Disbursed'] / mp_dim['MP_Total_Allocation']) * 100, 0)
mp_dim['MP_Performance_Rank'] = mp_dim['MP_Completion_Rate'].rank(ascending=False, method='min')

# Vendor Dimension
vendor_dim = master[master['Vendor_Name'] != 'Unknown'].groupby('Vendor_Name').agg(
    Vendor_Total_Projects_Count=('Work_ID', 'count'),
    Vendor_Avg_Timeline_Days=('Days_to_Complete', 'mean'),
    Vendor_Avg_Cost_Overrun_Ratio=('Cost_Overrun_Ratio', 'mean')
).reset_index()
vendor_dim['Vendor_Quality_Rating'] = [round(random.uniform(1.0, 5.0), 1) for _ in range(len(vendor_dim))]
vendor_dim['Vendor_Monopoly_Score'] = [round(random.uniform(5.0, 40.0), 2) for _ in range(len(vendor_dim))]
vendor_dim['Vendor_Risk_Score'] = [round(random.uniform(0, 100), 2) for _ in range(len(vendor_dim))]

# Geography Dimension
geography_dim = master[['State', 'Constituency']].drop_duplicates().reset_index(drop=True)
geography_dim['Constituency_ID'] = [f"CONST_{i}" for i in range(len(geography_dim))]
geography_dim['District_Code'] = [f"DIST_{i}" for i in range(len(geography_dim))]
geography_dim['Latitude'] = [round(random.uniform(8.0, 37.0), 4) for _ in range(len(geography_dim))]
geography_dim['Longitude'] = [round(random.uniform(68.0, 97.0), 4) for _ in range(len(geography_dim))]
geography_dim['Spatial_Cluster_ID'] = [f"CLUSTER_{random.randint(1,50)}" for _ in range(len(geography_dim))]
geography_dim['Zone_Fraud_Risk_Score'] = [round(random.uniform(10, 90), 2) for _ in range(len(geography_dim))]

# Mapping back Constituency_ID to master
master = pd.merge(master, geography_dim[['State', 'Constituency', 'Constituency_ID']], on=['State', 'Constituency'], how='left')

# AI and Compliance Dimension
ml_dim = master[['Work_ID']].copy()
ml_dim['Documents_Submitted'] = [",".join(random.sample(["Site Plan", "NOC", "Utilization Cert"], random.randint(1,3))) for _ in range(len(ml_dim))]
ml_dim['NOC_Status'] = [random.choice(["Received", "Pending", "Not_Required"]) for _ in range(len(ml_dim))]
ml_dim['Compliance_Score'] = [round(random.uniform(40, 100), 2) for _ in range(len(ml_dim))]
ml_dim['Is_Duplicate_Work'] = np.random.choice([True, False], size=len(ml_dim), p=[0.02, 0.98])
ml_dim['Cost_Anomaly_Flag'] = np.random.choice([True, False], size=len(ml_dim), p=[0.05, 0.95])
ml_dim['Overall_Fraud_Risk_Score'] = [round(random.uniform(0, 100), 2) for _ in range(len(ml_dim))]
ml_dim['Work_Description_Full'] = master['Work_Description'].fillna("")
ml_dim['Inspection_Summary'] = ["Standard inspection passed" if random.random() > 0.2 else "Issues found on site" for _ in range(len(ml_dim))]
ml_dim['Issues_Encountered'] = ["None" if random.random() > 0.3 else "Weather delays, Vendor issues" for _ in range(len(ml_dim))]

# Save the 5 files
print("Saving output files to CSV...")
works_cols = ['Work_ID', 'House', 'MP_Name', 'Vendor_Name', 'Constituency_ID', 'State', 'Constituency', 'Work_Category', 'Work_Description', 'Work_Status', 'Implementing_Agency', 'Calamity_Flag', 'Recommended_Date', 'Sanction_Date', 'Completion_Date', 'Days_to_Sanction', 'Days_to_Complete', 'Days_Pending', 'Recommended_Amount', 'Sanction_Amount', 'Disbursed_Amount', 'Cost_Overrun_Amount', 'Cost_Overrun_Ratio', 'Work_Quantity', 'Work_Scope_Unit', 'Cost_per_Unit']

master_out = master[[c for c in works_cols if c in master.columns]]
master_out.to_csv(os.path.join(OUTPUT_DIR, 'works_master.csv'), index=False)
mp_dim.to_csv(os.path.join(OUTPUT_DIR, 'mp_dimension.csv'), index=False)
vendor_dim.to_csv(os.path.join(OUTPUT_DIR, 'vendor_dimension.csv'), index=False)
geography_dim.to_csv(os.path.join(OUTPUT_DIR, 'geography_dimension.csv'), index=False)
ml_dim.to_csv(os.path.join(OUTPUT_DIR, 'compliance_and_ml.csv'), index=False)

print(f"Data generation complete! Saved 5 files to {OUTPUT_DIR}")
print(f"works_master.csv: {len(master_out)} rows")
