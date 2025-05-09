# Lessons Learned & Best Practices

This document summarizes the key best practices and lessons learned from the Employee Attrition Prediction project. It is intended as a learning resource.

---

## 1. Data Handling & Preparation
- **Early Data Split:**
  - The original dataset is split into a modeling set (80%) and an inference set (20%) *before* any cleaning or feature engineering. This prevents data leakage and ensures realistic model evaluation.
- **Consistent Preprocessing:**
  - All cleaning, type casting, and feature engineering are performed only on the modeling set. The inference set is only minimally processed until deployment.
- **Feature Engineering:**
  - New features (e.g., AgeGroup, TenureRatio, OverallSatisfaction) are created to improve interpretability and predictive power.
- **Data Validation:**
  - Systematic checks for missing values, duplicates, and correct data types are performed and documented.

## 2. Modeling & Experimentation
- **Model Selection Criteria:**
  - Model choice is driven by business needs (e.g., recall is prioritized for attrition detection).
  - Multiple models are compared using cross-validation; linear models (LDA, Ridge, Logistic Regression) are favored for recall.
- **Imbalance Handling:**
  - Techniques like SMOTE are used to address class imbalance.
- **Holdout & Cross-Validation:**
  - 10-fold cross-validation is used on the modeling set; the inference set is held out for final evaluation.
- **Feature Importance & Interpretability:**
  - Feature importance plots are generated for the final model to aid business understanding.

## 3. Deployment & Inference
- **Pipeline Saving:**
  - The final model and preprocessing pipeline are saved using `save_model` for reproducibility and deployment.
- **API Design:**
  - A FastAPI app is implemented for real-time or batch predictions, with input validation and tracking via EmployeeId.
- **Consistent Inference Preprocessing:**
  - The same feature engineering and categorical preprocessing are applied to inference data as in training.

## 4. Experiment Tracking & Evidence
- **MLflow Integration:**
  - PyCaret's MLflow integration is recommended for logging parameters, metrics, and artifacts for each experiment run.
  - This provides a UI and audit trail for all modeling decisions and results.
- **Saving Artifacts:**
  - All important outputs (models, plots, predictions) are saved for reproducibility and evidence.

## 4a. Pipeline Automation & Execution
- **Automated Pipeline Script:**
  - A robust `run_all.py` script orchestrates the execution of all notebooks/scripts in the correct order, ensuring dependencies are respected.
  - The script checks for required outputs after each step and skips steps if outputs are up-to-date, supporting fast iterative development.
- **Error Handling & Logging:**
  - Each script is run in a subprocess with detailed error handling. Errors are logged to timestamped log files for easy debugging.
  - The pipeline can be configured to stop on error or continue, and missing outputs are clearly reported.
- **Checkpoint/Resume & Force Options:**
  - The pipeline supports resuming from any step (`--from-step`) and forcing re-runs (`--force`), making it easy to debug or rerun only parts of the workflow.
- **Output Validation:**
  - After each step, the script validates that all expected outputs are present and up-to-date, warning if anything is missing or stale.
- **Reproducibility & Collaboration:**
  - This approach ensures that the entire workflow can be rerun end-to-end or in parts, supporting reproducibility, collaboration, and robust experimentation.

## 5. Documentation & Collaboration
- **Comprehensive README:**
  - The README includes project summary, API usage, and links to detailed docs.
- **Modular Documentation:**
  - Separate markdown files document data preparation, experimentation, future works, and lessons learned.
- **Clear TODOs:**
  - A prioritized TODO list tracks outstanding tasks and future improvements.

## 6. Generalizable Practices
- **Data Leakage Prevention:**
  - Always split data before any transformation or feature engineering.
- **Business-Driven Metrics:**
  - Choose model metrics and thresholds based on business impact, not just technical performance.
- **Reproducibility:**
  - Save all code, data splits, and model artifacts; use experiment tracking tools.
- **Continuous Improvement:**
  - Document future work and lessons learned for ongoing project evolution.

---

*This document is a living resource. Update as new best practices and lessons are discovered.* 