# %%
"""
04 Modeling
Train and evaluate models for employee attrition prediction.
"""
# %%
from src.modeling import setup_modeling, train_and_tune_model, evaluate_trained_model, plot_feature_importance, save_trained_model
import pandas as pd
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import numpy as np
import os
from pycaret.classification import predict_model, plot_model
import shutil
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

# Save model
model_path = save_trained_model(model, 'models/final_lda_model')
# PyCaret appends .pkl if not present
if not os.path.exists('models/final_lda_model.pkl'):
    print("WARNING: Model file models/final_lda_model.pkl not found after saving.")
else:
    print("Model saved as models/final_lda_model.pkl")

# Export confusion matrix and classification report
try:
    if hasattr(model, 'predict'):
        y_true = features_df['Attrition'] if 'Attrition' in features_df.columns else None
        preds_df = predict_model(model, data=features_df)
        y_pred = preds_df['Label'] if 'Label' in preds_df.columns else None
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
                y_score = preds_df['Score'] if 'Score' in preds_df.columns else None
                if y_score is not None:
                    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=np.unique(y_true)[1])
                    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=np.unique(y_true)[1])
                    np.savetxt('results/roc_curve.csv', np.column_stack([fpr, tpr]), delimiter=',', header='fpr,tpr', comments='')
                    np.savetxt('results/pr_curve.csv', np.column_stack([precision, recall]), delimiter=',', header='precision,recall', comments='')
            # Save predictions for consistency
            preds_df.to_csv('results/predictions.csv', index=False)
        else:
            print("WARNING: y_true or y_pred is None. Confusion matrix not exported.")
    else:
        print("WARNING: Model does not have a 'predict' attribute. Confusion matrix not exported.")
except Exception as e:
    print(f"ERROR exporting confusion matrix: {e}")

# %%
display(Markdown("""
## Feature Importance
We analyze which features are most influential in predicting attrition.
"""))
# %%
# Feature importance
plot_model(model, plot='feature', save=True)
shutil.move('Feature Importance.png', 'results/feature_importance_plot.png')
importance_df = plot_feature_importance(model)
display(importance_df.head())
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