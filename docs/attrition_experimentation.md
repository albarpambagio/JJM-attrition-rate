# Attrition Model Experimentation Log

## 1. Data Preparation & Setup
- **Data split:** The original dataset is split into two parts:
  - **Modeling set (80%)**: Used for all cleaning, feature engineering, EDA, and model development.
  - **Inference set (20%)**: Held out and only minimally processed (dropping `Attrition` and `EmployeeId`) to simulate new/unseen employee data for deployment or inference testing.
- **Categorical columns** were preprocessed to replace spaces and ampersands for consistent encoding.
- **PyCaret setup** included:
  - Numeric, categorical, and ordinal feature specification
  - One-hot encoding (max 10 categories, rare grouping)
  - Normalization (z-score)
  - Feature selection (LightGBM estimator, 20% features kept)
  - Imbalance handling (SMOTE)
  - 70% train split, 10-fold cross-validation (on modeling set only)

## 2. Baseline Model Comparison
- **compare_models()** was run to benchmark a variety of classifiers on the modeling set.
- **Key results:**
  - Tree-based models (Random Forest, Extra Trees, LightGBM, GBC) had high accuracy but very low recall (missed most attrition cases).
  - Linear models (LDA, Ridge, Logistic Regression) had much higher recall (~67%) but lower accuracy (~70%).
  - Naive Bayes had the highest recall but very low accuracy.

| Model         | Accuracy | AUC   | Recall | Precision | F1   |
|--------------|----------|-------|--------|-----------|------|
| LDA          | 0.70     | 0.71  | 0.67   | 0.24      | 0.35 |
| Ridge        | 0.70     | 0.71  | 0.67   | 0.24      | 0.35 |
| Logistic Reg | 0.69     | 0.71  | 0.68   | 0.24      | 0.35 |
| RF           | 0.85     | 0.66  | 0.23   | 0.32      | 0.26 |
| Dummy        | 0.88     | 0.50  | 0.00   | 0.00      | 0.00 |

## 3. Tuning Linear Models for Recall
- **LDA** was tuned with `tune_model(..., optimize='Recall')` on the modeling set.
- **Result:**
  - Recall remained high (~67%), precision low (~24%), F1 moderate (~0.35).
  - No significant improvement from bagging/ensembling.

## 4. Ensembling & Advanced Techniques
### a. Bagging (Ensemble)
- Bagging the tuned LDA did not improve recall or F1.

### b. Blending
- Blending LDA with tuned Random Forest and GBC increased accuracy (~0.85) but **dropped recall to ~23%** (not desirable for attrition detection).

### c. Stacking
- Stacking (LDA + RF + GBC) returned to LDA-like performance (recall ~67%, F1 ~0.35).

## 5. Inference Set for Deployment Testing
- The inference set is used to simulate new employee data for deployment or API testing.
- Predictions are made using the final model and the held-out inference set, ensuring a realistic evaluation of deployment performance.

## 6. Latest Tuned LDA Results (After Improved Data Prep)
- **Cross-validation metrics (mean across folds):**

| Metric    | Value  | Std   |
|-----------|--------|-------|
| Accuracy  | 0.718  | 0.039 |
| AUC       | 0.719  | 0.122 |
| Recall    | 0.630  | 0.179 |
| Precision | 0.323  | 0.063 |
| F1        | 0.425  | 0.093 |
| Kappa     | 0.262  | 0.115 |
| MCC       | 0.291  | 0.135 |

- **Interpretation:**
  - The model now catches 63% of true attrition cases (recall), with moderate precision (32%).
  - AUC remains strong, indicating good separability.
  - Fold variance is present but overall performance is stable.
  - The new workflow is more realistic and less prone to overfitting, providing a robust estimate of generalization.

## 7. Key Findings
- **Linear models (LDA, Ridge, LR) optimized for recall are best for catching attrition cases.**
- **Ensembling and stacking did not improve recall or F1 in this scenario.**
- **Blending with tree models increased accuracy but drastically reduced recall.**
- **If recall is the business priority, stick with tuned LDA or similar linear models.**
- **The improved workflow with a held-out inference set provides a more realistic and production-ready evaluation.**

## 8. Recommendations & Next Steps
- Consider adjusting the probability threshold to further optimize recall/precision trade-off.
- Explore additional feature engineering or alternative imbalance techniques if higher recall is needed.
- Use confusion matrix and business cost analysis to select the final model.

## SHAP Feature Selection

To identify the most influential features for predicting employee attrition, we used SHAP (SHapley Additive exPlanations) values. SHAP provides a unified measure of feature importance by quantifying the contribution of each feature to the model's predictions. After training our model, we calculated SHAP values for all features and selected those with the highest mean absolute SHAP values. These features were then used both in the final model and as the basis for the visualizations in the Metabase dashboard. This approach ensures that the dashboard highlights the factors that truly drive attrition predictions, making the insights more actionable and interpretable.

---

*This log will be updated as further experiments are conducted.* 