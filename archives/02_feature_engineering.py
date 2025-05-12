# %%
"""
03 Feature Engineering
Create new features for modeling employee attrition.
"""
# %%
from src.feature_engineering import engineer_features
import pandas as pd
from IPython.display import display, Markdown
# %%
display(Markdown("""
# Feature Engineering
This notebook creates new features from the cleaned employee attrition dataset to improve model performance and interpretability.
"""))
# %%
# Load cleaned data
display(Markdown("""
## Load Cleaned Data
We start by loading the cleaned dataset prepared in the previous step.
"""))
clean_df = pd.read_csv('data/employee_data_cleaned.csv')
clean_df.head()
# %%
# Engineer features
display(Markdown("""
## Engineer Features
We apply feature engineering techniques to create new variables that may help predict attrition.
"""))
features_df = engineer_features(clean_df)
features_df.head()
# %%
# Save engineered features for modeling
display(Markdown("""
## Save Engineered Features
The engineered features are saved for use in the modeling notebook.
"""))
features_df.to_csv('data/employee_data_features.csv', index=False) 