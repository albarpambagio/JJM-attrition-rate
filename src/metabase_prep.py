import pandas as pd
import sqlite3
import os

# Ensure we're in the project root directory
project_root = r"C:\Users\USER\Documents\Projects\JJM-attrition-rate"
if os.getcwd() != project_root:
    os.chdir(project_root)
    print("Changed working directory to:", os.getcwd())

# --- CONFIGURATION ---
# Path to SHAP feature importance CSV
shap_csv = 'results/shap_feature_importance.csv'
# Path to your base (feature-engineered) data
base_data_csv = 'data/employee_data_features.csv'
# Output SQLite database path
sqlite_db = 'results/feature_monitor.db'
# Name of the table to create/update
table_name = 'shap_selected_features'
# Name(s) of target or extra columns to include
extra_columns = ['Attrition']  # Add more if needed

# --- LOAD SHAP FEATURE IMPORTANCE ---
shap_importance = pd.read_csv(shap_csv)
top_features = shap_importance['Feature'].tolist()  # Use all, or .head(N) for top N

# --- LOAD BASE DATA ---
df = pd.read_csv(base_data_csv)

# --- SELECT IMPORTANT VARIABLES (+ TARGET) ---
selected_columns = [col for col in top_features if col in df.columns] + [col for col in extra_columns if col in df.columns]
df_selected = df[selected_columns]

# --- SAVE TO SQLITE ---
os.makedirs(os.path.dirname(sqlite_db), exist_ok=True)
conn = sqlite3.connect(sqlite_db)
df_selected.to_sql(table_name, conn, if_exists='replace', index=False)
conn.close()

print(f"Selected features saved to {sqlite_db} (table: {table_name})")
print(f"Columns: {df_selected.columns.tolist()}")
print(f"Rows: {len(df_selected)}") 