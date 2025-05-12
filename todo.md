# TODO List

<!--
  This file uses TODO: and FIXME: tags for compatibility with the Todo Tree extension.
  Place this file in your project root for best results.
-->

---
## High Priority
- [x] Summarize key findings and actionable recommendations for stakeholders (before conversion)
- [x] Add markdown explanations and interpretations to all notebooks
- [x] Add confusion matrix and classification report to model evaluation
- [x] Plot ROC and Precision-Recall curves for best models
- [x] Prepare visualizations and tables for a final report or presentation
- [x] Automate the workflow (e.g., with a pipeline or script)
- [x] Use SHAP for global feature importance and actionable insights
- [ ] Estimate business impact: Calculate cost/benefit of false positives/negatives
- [ ] Write a prediction script or notebook for batch or real-time scoring
- [ ] Automate categorical preprocessing for both training and prediction
- [ ] Validate rare category grouping and adjust encoding if needed
- [ ] Finalize and document the best model for deployment
- [ ] Prepare a summary report/dashboard for stakeholders

## Medium Priority
- [ ] Try probability threshold adjustment for the best model
- [ ] Experiment with other imbalance techniques (e.g., SMOTE variants, undersampling)
- [ ] Try additional models (e.g., CatBoost, XGBoost, MLP, Random Forest) and compare with LDA
- [ ] Use SHAP for interpretability of all candidate models
- [ ] Tune hyperparameters with higher n_iter for more thorough search
- [ ] Compare model performance on a true holdout set (if available)
- [ ] Explore additional features (e.g., interaction terms, tenure buckets, department-level aggregates)
- [ ] Use SHAP or LIME for local interpretability

## Optional/Long-term
- [ ] Integrate with BI tools (Power BI, Tableau) for dashboarding
- [ ] Set up experiment tracking (e.g., MLflow, Weights & Biases)
- [ ] Ensure all code is well-documented and commented
- [ ] Keep requirements.txt/pyproject.toml up to date
- [ ] Input validation (checking for missing columns, correct dtypes, etc.)
- [ ] Error handling (for bad input, model load errors, etc.)
- [ ] Logging and monitoring
- [ ] Security (if exposing as an API)
- [ ] Documentation for API endpoints

---



