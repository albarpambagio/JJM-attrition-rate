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
from pycaret.classification import predict_model, plot_model, pull
import shutil
import shap
import matplotlib.pyplot as plt
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
# --- SHAP summary plot and CSV export ---
from pycaret.classification import get_config

# Get the transformed training data used by PyCaret
X_train_transformed = get_config('X_train_transformed')

# Create a SHAP explainer for your model
explainer = shap.Explainer(model, X_train_transformed)
shap_values = explainer(X_train_transformed)

# Generate and save SHAP summary plot
shap.summary_plot(shap_values.values, X_train_transformed, show=False)
plt.tight_layout()
plt.savefig('results/shap_summary_plot.png', bbox_inches='tight')
plt.close()

# SHAP feature importance (mean absolute SHAP value per feature)
shap_importance = pd.DataFrame({
    'Feature': X_train_transformed.columns,
    'MeanAbsSHAP': np.abs(shap_values.values).mean(axis=0)
}).sort_values(by='MeanAbsSHAP', ascending=False)
shap_importance.to_csv('results/shap_feature_importance.csv', index=False)

# %%
display(Markdown("""
## Save Model
The final trained model is saved for future inference and deployment.
"""))
# %%
# Save model
save_trained_model(model, 'models/final_lda_model') 

# %%
display(Markdown("""
## Model Performance Insights

### Feature Importance & Model Performance
- **Model Used:** Linear Discriminant Analysis (LDA)
- **Performance:** Accuracy: 70%, AUC: 0.78, Recall: 0.75 (good for catching attrition cases), Precision: 0.34 (many false positives)
- **Key Features (by SHAP importance):**
    1. **OverTime** — Most influential; employees who work overtime are much more likely to leave.
    2. **EnvironmentSatisfaction** — Lower satisfaction increases attrition risk.
    3. **Age** — Younger employees tend to have higher attrition risk.
    4. **MonthlyIncome** — Lower income is associated with higher attrition.
    5. **DailyRate, DistanceFromHome, RoleStability, MonthlyRate** — These also contribute, but to a lesser extent.

#### SHAP Feature Importance Analysis
The table below (from `shap_feature_importance.csv`) shows the mean absolute SHAP value for each feature, which quantifies the average impact of each feature on the model's prediction for employee attrition. A higher value means the feature has a greater influence on the model's output.

| Rank | Feature                 | MeanAbsSHAP |
|------|-------------------------|-------------|
| 1    | OverTime                | 0.75        |
| 2    | EnvironmentSatisfaction | 0.56        |
| 3    | Age                     | 0.37        |
| 4    | MonthlyIncome           | 0.26        |
| 5    | DailyRate               | 0.20        |
| 6    | DistanceFromHome        | 0.19        |
| 7    | RoleStability           | 0.16        |
| 8    | MonthlyRate             | 0.15        |

**Interpretation:**
- **OverTime** is by far the most important feature, with a mean absolute SHAP value of 0.75. This means that whether or not an employee works overtime has the largest average effect on the model's prediction of attrition.
- **EnvironmentSatisfaction** is the second most important, indicating that employees' satisfaction with their work environment is a key driver of attrition risk.
- **Age** is also significant, suggesting that attrition risk varies notably with employee age (often, younger employees are more likely to leave).
- **MonthlyIncome** and **DailyRate** both have moderate influence, showing that compensation factors play a role, but are less critical than overtime or satisfaction.
- **DistanceFromHome**, **RoleStability**, and **MonthlyRate** have smaller but still meaningful impacts.

**Actionable Insights:**
- **Monitor and manage overtime:** Since overtime is the top driver, reducing excessive overtime or compensating for it may help reduce attrition.
- **Improve environment satisfaction:** Initiatives to boost workplace satisfaction could have a strong effect on retention.
- **Targeted retention for younger employees:** Since age is a key factor, consider tailored retention programs for younger staff.
- **Review compensation and stability:** While not the top factors, fair pay and stable roles still contribute to retention and should not be neglected.

## Modeling Recommendations
2. **Modeling Improvements:**
   - While recall is high, precision is low. Consider:
     - Collecting more data to balance class distribution
     - Engineering interaction features between key variables
     - Experimenting with alternative models (XGBoost, Random Forest)
     - Adjusting classification decision thresholds
     - Implementing feature selection to reduce noise
"""))