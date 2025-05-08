# Attrition Model Experimentation Log

## 1. Data Preparation & Setup
- **Categorical columns** were preprocessed to replace spaces and ampersands for consistent encoding.
- **PyCaret setup** included:
  - Numeric, categorical, and ordinal feature specification
  - One-hot encoding (max 10 categories, rare grouping)
  - Normalization (z-score)
  - Feature selection (LightGBM estimator, 20% features kept)
  - Imbalance handling (SMOTE)
  - 70% train split, 10-fold cross-validation

## 2. Baseline Model Comparison
- **compare_models()** was run to benchmark a variety of classifiers.
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
- **LDA** was tuned with `tune_model(..., optimize='Recall')`.
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

## 5. Key Findings
- **Linear models (LDA, Ridge, LR) optimized for recall are best for catching attrition cases.**
- **Ensembling and stacking did not improve recall or F1 in this scenario.**
- **Blending with tree models increased accuracy but drastically reduced recall.**
- **If recall is the business priority, stick with tuned LDA or similar linear models.**

## 6. Recommendations & Next Steps
- Consider adjusting the probability threshold to further optimize recall/precision trade-off.
- Explore additional feature engineering or alternative imbalance techniques if higher recall is needed.
- Use confusion matrix and business cost analysis to select the final model.

---

*This log will be updated as further experiments are conducted.* 