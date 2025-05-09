# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
# %%
"""
04 Modeling
Train and evaluate models for employee attrition prediction.
"""
# %%
from modeling import setup_modeling, train_and_tune_model, evaluate_trained_model, plot_feature_importance, save_trained_model
import pandas as pd
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import numpy as np
import os
# %%
display(Markdown("""
# Modeling
This notebook trains and evaluates machine learning models to predict employee attrition.
"""))
# %%
display(Markdown("""
## Load Engineered Features
We load the dataset with engineered features for modeling.
"""))
# %%
# Load engineered features
features_df = pd.read_csv('data/employee_data_features.csv')
features_df.head()
# %%
display(Markdown("""
## Preprocess Categorical Columns
We preprocess categorical variables to ensure they are in a suitable format for modeling.
"""))
# %%
# Preprocess categorical columns for modeling (if needed)
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                   'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'AgeGroup']
for col in categorical_cols:
    features_df[col] = features_df[col].astype(str).str.replace(' ', '_').str.replace('&', '_and_')
# %%
display(Markdown("""
## Setup Modeling Environment
We initialize the modeling environment, including data splitting and preprocessing.
"""))
# %%
# Setup modeling environment
setup_modeling(features_df)
# %%
display(Markdown("""
## Train and Tune Model
We train and tune a machine learning model to optimize for recall (catching as many attrition cases as possible).
"""))
# %%
# Train and tune model
model = train_and_tune_model(model_name='lda', optimize='Recall', n_iter=50)
# %%
display(Markdown("""
## Evaluate Model
We evaluate the trained model's performance using relevant metrics.
"""))
# %%
# Evaluate model
evaluate_trained_model(model)

# Export confusion matrix and classification report
if hasattr(model, 'predict'):
    y_true = features_df['Attrition'] if 'Attrition' in features_df.columns else None
    y_pred = model.predict(features_df.drop(columns=['Attrition', 'EmployeeId'], errors='ignore')) if y_true is not None else None
    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cr = classification_report(y_true, y_pred)
        os.makedirs('results', exist_ok=True)
        with open('results/confusion_matrix.md', 'w') as f:
            f.write('# Confusion Matrix\n')
            f.write(str(cm))
        with open('results/classification_report.md', 'w') as f:
            f.write('# Classification Report\n')
            f.write(cr)
        # ROC and PR curve data (if binary)
        if len(np.unique(y_true)) == 2:
            y_score = model.predict_proba(features_df.drop(columns=['Attrition', 'EmployeeId'], errors='ignore'))[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=np.unique(y_true)[1])
            precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=np.unique(y_true)[1])
            np.savetxt('results/roc_curve.csv', np.column_stack([fpr, tpr]), delimiter=',', header='fpr,tpr', comments='')
            np.savetxt('results/pr_curve.csv', np.column_stack([precision, recall]), delimiter=',', header='precision,recall', comments='')
# %%
display(Markdown("""
## Feature Importance
We analyze which features are most influential in predicting attrition.
"""))
# %%
# Feature importance
plot, importance_df = plot_feature_importance(model)
plot.display()
importance_df.head()
# Export feature importance to CSV
if importance_df is not None:
    importance_df.to_csv('results/feature_importance.csv', index=False)
# %%
display(Markdown("""
## Save Model
The final trained model is saved for future inference and deployment.
"""))
# %%
# Save model
save_trained_model(model, 'models/final_lda_model') 