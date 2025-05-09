# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
# %%
"""
05 Inference
Use the trained model to predict attrition on new/unseen data.
"""
# %%
from inference import predict_attrition
from feature_engineering import engineer_features
import pandas as pd
from IPython.display import display, Markdown
# %%
display(Markdown("""
# Inference
This notebook uses the trained attrition model to make predictions on new or unseen employee data.
"""))
# %%
display(Markdown("""
## Load New/Unseen Data
We load new or held-out employee data for inference. Replace with actual new data as needed.
"""))
# %%
# Load new/unseen data (simulate with held-out set)
infer_df = pd.read_csv('data/employee_data_cleaned.csv')  # Replace with actual new data if available
# %%
display(Markdown("""
## Prepare Input Data
We remove columns not needed for prediction (e.g., Attrition, EmployeeId).
"""))
# %%
# Drop Attrition and EmployeeId if present
infer_input = infer_df.drop(columns=[col for col in ['Attrition', 'EmployeeId'] if col in infer_df.columns])
# %%
display(Markdown("""
## Engineer Features for Inference
We apply the same feature engineering steps as in training to ensure consistency.
"""))
# %%
# Engineer features for inference
df_infer_fe = engineer_features(infer_input)
# %%
display(Markdown("""
## Preprocess Categorical Columns
We preprocess categorical variables to match the format used during model training.
"""))
# %%
# Preprocess categorical columns as in training
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                   'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'AgeGroup']
for col in categorical_cols:
    if col in df_infer_fe.columns:
        df_infer_fe[col] = df_infer_fe[col].astype(str).str.replace(' ', '_').str.replace('&', '_and_')
# %%
display(Markdown("""
## Predict Attrition
We use the trained model to predict attrition probabilities and classes for each employee.
"""))
# %%
# Predict attrition
predictions = predict_attrition(df_infer_fe)
predictions.head() 