# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
# %%
"""
01 Data Cleaning
Load and clean the employee attrition dataset.
"""
# %%
from data_processing import load_data, clean_data, split_data
import pandas as pd
from IPython.display import display, Markdown
# %%
display(Markdown("""
# Data Cleaning
This notebook loads and cleans the raw employee attrition dataset, preparing it for analysis and modeling.
"""))
# %%
# Load data
raw_df = load_data('data/employee_data.csv')
raw_df.head()
# %%
display(Markdown("""
## Load Raw Data
We begin by loading the raw dataset to inspect its structure and contents.
"""))
# %%
# Clean data
clean_df = clean_data(raw_df)
clean_df.head()
# %%
display(Markdown("""
## Clean Data
We apply cleaning steps to handle missing values, correct data types, and fix inconsistencies.
"""))
# %%
# Optionally, split data for modeling and inference
model_df, infer_df = split_data(clean_df)
print(f"Modeling set: {model_df.shape}, Inference set: {infer_df.shape}")
# %%
display(Markdown("""
## Split Data (Optional)
We optionally split the cleaned data into modeling and inference sets for downstream tasks.
"""))
# %%
# Save cleaned data for next steps
clean_df.to_csv('data/employee_data_cleaned.csv', index=False)
# %%
display(Markdown("""
## Save Cleaned Data
The cleaned dataset is saved for use in EDA and modeling notebooks.
""")) 