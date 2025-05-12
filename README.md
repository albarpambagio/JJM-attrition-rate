# Employee Attrition Prediction API

## 🎯 Business Problem

**Jaya Jaya Maju** is a multinational company with over 1,000 employees across Indonesia. Despite its growth, the company faces a major challenge in managing and retaining its workforce. Currently, the employee attrition rate has exceeded 10% and may continue to rise if not addressed promptly.

To tackle this issue, the HR team aims to understand the key drivers behind employee attrition and leverage technology to monitor workforce trends on an ongoing basis. The business needs a data-driven solution to identify the main factors contributing to attrition, predict high-risk employees, and present insights through an intuitive dashboard, enabling faster and more targeted HR interventions.

## 📋 Project Scope

### Data Preparation & Analysis
- Data cleaning and validation
- Feature engineering (age groups, tenure ratios, satisfaction indices)
- Exploratory Data Analysis (EDA) with visualizations
- Data splitting for modeling and inference

### Model Development
- Automated preprocessing using PyCaret
- Multiple model comparison and selection
- Hyperparameter tuning and optimization
- Model performance evaluation and interpretation

### Deployment & Monitoring
- FastAPI service for predictions
- Metabase dashboard integration
- Key metrics monitoring:
  - Overall attrition rates and trends
  - Department and role-wise attrition
  - Work environment metrics
  - Career development indicators
  - Compensation and benefits analysis
  - Predictive insights and early warnings

## 📁 Project Structure

```
JJM-attrition-rate/
├── api/                         # FastAPI app for predictions (main.py, etc.)
├── archives/                    # (Older scripts, legacy, or archived analysis)
├── data/                        # Raw and processed data files
├── docs/                        # Documentation (markdown, pdf, etc.)
│   ├── data_preparation_and_eda.md
│   ├── attrition_experimentation.md
│   ├── metabase_setup.md
│   ├── lessons_learned.md
│   ├── experiment_result.txt
│   ├── albarpambagio-dashboard_page_1.png
│   └── albarpambagio-dashboard_page_2.png    
├── docker/                      # Docker and deployment files
├── eda_outputs/                 # EDA result files (plots, tables, etc.)
├── logs/                        # Log files
├── models/                      # Saved models and model artifacts
├── notebooks/                   # All notebook-style scripts (for Jupytext or .ipynb)
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_modeling.py
│   └── 05_inference.py
├── results/                     # Output files (mv, csv, db, etc.)
├── src/                         # All Python modules (reusable code)
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── eda_tools.py
│   ├── modeling.py
│   ├── inference.py
│   └── metabase_prep.py
├── run_all.py                   # Orchestration script
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── todo.md
└── .python-version
```

- **All reusable code** (functions, classes) lives in `src/`.
- **All notebook-style scripts** (for step-by-step analysis, EDA, modeling, etc.) live in `notebooks/`.
- **Data** and **models** are in their respective folders.
- **Documentation** (markdown, pdf, etc.) is in `docs/`.
- **Deployment files** (Docker, Kubernetes, etc.) are in `docker/`.
- **EDA outputs** (figures, charts) are in `eda_outputs/`.
- **Log files** are in `logs/`.
- **Output files** (csv, db, etc.) are in `results/`.
- **`archives/`** contains legacy or old scripts for reference only.

## 📊 Project Summary

### Data Source
- The employee attrition dataset used in this project is sourced from [Dicoding Academy Employee Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee).

### Data Preparation & EDA
- The dataset is split into a **modeling set (80%)** for all cleaning, feature engineering, EDA, and model development, and an **inference set (20%)** held out to simulate new/unseen employee data for deployment.
- Extensive data validation, cleaning, and type casting are performed on the modeling set only, with new features engineered for interpretability and predictive power (e.g., AgeGroup, TenureRatio, OverallSatisfaction, PromotionRate).
- Exploratory Data Analysis (EDA) includes interactive visualizations of attrition by department, job role, age group, and satisfaction distributions.
- The inference set is only minimally processed to avoid data leakage.
- [Full details in docs/data_preparation_and_eda.md](docs/data_preparation_and_eda.md)

### Model Experimentation & Results
- Multiple models were benchmarked; **linear models (LDA, Ridge, Logistic Regression)** achieved the best recall (~67%) for attrition detection, while tree models had higher accuracy but much lower recall.
- The final tuned LDA model (after improved data prep) achieved:
  - **Accuracy:** 0.72
  - **AUC:** 0.72
  - **Recall:** 0.63
  - **Precision:** 0.32
  - **F1:** 0.43
- Ensembling and stacking did not improve recall; blending increased accuracy but reduced recall.
- The workflow with a held-out inference set provides a realistic, production-ready evaluation.
- [Full details in docs/attrition_experimentation.md](docs/attrition_experimentation.md)

### Feature Importance Analysis
- **SHAP (SHapley Additive exPlanations)** values are used to understand feature importance and model predictions:
  - Global feature importance shows which factors most influence attrition predictions
  - Feature importance scores are saved in `results/shap_feature_importance.csv`
  - Top features are used to create the Metabase dashboard for monitoring
- Key findings from SHAP analysis:
  - Work-life balance and job satisfaction are strong predictors of attrition
  - Age and tenure ratios provide important context for attrition risk
  - Department and job role interactions reveal patterns in employee retention
- The analysis helps identify actionable insights for HR interventions
- [Full details in docs/attrition_experimentation.md](docs/attrition_experimentation.md)

## 🚀 Implementation

### API Deployment & Usage

### 1. Install Requirements
Make sure you have all dependencies installed (ideally in a virtual environment):
```bash
pip install -r requirements.txt
```

### 2. Run the API Server
Start the FastAPI server using Uvicorn:
```bash
uvicorn api:app --reload
```
- The API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 3. Example: Predict Attrition
#### Endpoint
`POST /predict`

#### Request Body (JSON)
A list of employee records (all fields except `Attrition`; `EmployeeId` is optional for tracking):
```json
[
  {
    "EmployeeId": 1,
    "Age": 35,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 1100,
    "Department": "Research & Development",
    "DistanceFromHome": 5,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 60,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 4,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "MonthlyRate": 20000,
    "NumCompaniesWorked": 1,
    "Over18": "Y",
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StandardHours": 80,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2
  }
]
```

#### Example Response
```json
[
  {
    "EmployeeId": 1,
    "Label": 0,
    "Score": 0.85
  }
]
```
- `Label`: 1 = Attrition predicted, 0 = No attrition
- `Score`: Probability/confidence of attrition

### Business Dashboard

The project includes a Metabase dashboard that focuses on monitoring the most important factors influencing employee attrition, as identified by SHAP analysis.

### Dashboard Implementation

1. **Data Source**
   - SQLite database (`results/feature_monitor.db`)
   - Table: `shap_selected_features`
   - Contains only the most impactful features identified by SHAP analysis

2. **Key Features Monitored**
   - Top SHAP-identified features from the model
   - Target variable (Attrition)
   - Features are automatically selected based on SHAP importance scores

3. **Setup & Access**
   - Dashboard is accessible at `http://localhost:3000`
   - Uses Metabase OSS (Open Source)
   - Data is stored in a local SQLite database
   - Features are automatically updated when the model is retrained

For detailed setup instructions, see [Metabase Setup Guide](docs/metabase_setup.md).

## 🎯 Conclusion

This project successfully developed a comprehensive solution for predicting and managing employee attrition at Jaya Jaya Maju. The implementation combines machine learning with business intelligence to provide actionable insights for HR decision-making.

### Key Findings

1. **Demographics & Attrition Patterns**
   - 17% overall attrition rate
   - Higher attrition in younger age groups (26-35 years)
   - 29% of employees work overtime, a significant risk factor
   - Most employees in R&D (66%) and Sales (30%)

2. **Critical Factors (SHAP Analysis)**
   - **OverTime** (0.75): Most influential factor in attrition
   - **EnvironmentSatisfaction** (0.56): Second most important driver
   - **Age** (0.37): Younger employees more likely to leave
   - **MonthlyIncome** (0.26): Compensation impacts retention
   - **DailyRate & DistanceFromHome**: Moderate influence

### Recommendations

1. **Immediate Actions**
   - Implement overtime management policies
   - Enhance work environment satisfaction programs
   - Develop targeted retention strategies for younger employees
   - Review compensation structures

2. **Long-term Strategies**
   - Regular monitoring of key metrics through the dashboard
   - Continuous model retraining with new data
   - Department-specific retention programs
   - Career development initiatives

3. **Technical Improvements**
   - Collect more data to improve model precision
   - Engineer interaction features between key variables
   - Experiment with alternative models
   - Implement automated monitoring of model performance

### Project Impact
- **Data-Driven Insights**: Identified key factors driving attrition through SHAP analysis
- **Predictive Capability**: Achieved 75% recall in identifying at-risk employees
- **Monitoring System**: Established automated dashboard for continuous tracking
- **Actionable Intelligence**: Provided clear recommendations for retention strategies

### Business Value
- **Cost Reduction**: Early identification of attrition risks can reduce recruitment costs
- **Improved Retention**: Targeted interventions based on data-driven insights
- **Resource Optimization**: Better allocation of HR resources to high-risk areas
- **Strategic Planning**: Enhanced ability to forecast and plan for workforce changes

### Future Outlook
The project establishes a foundation for ongoing workforce analytics and predictive modeling. With regular updates and refinements, this system will continue to provide valuable insights for employee retention and organizational development.

For detailed analysis and methodology, see:
- [Data Preparation & EDA](docs/data_preparation_and_eda.md)
- [Model Experimentation](docs/attrition_experimentation.md)

## 📄 More Information
- The API applies all feature engineering and preprocessing steps as in model training.
- For more details, see the code in `api.py` and the modules in `src/`.

## 🔄 Updated Workflow for Attrition Analysis

1. **Run Analysis Scripts**: Use `run_all.py` to execute the data cleaning, feature engineering, modeling, and inference scripts in sequence. This will generate all necessary intermediate and final output files.

2. **SHAP Feature Importance**: After modeling, SHAP values are calculated to determine the most important features influencing attrition predictions. The results are saved in `results/shap_feature_importance.csv`.

3. **Prepare Data for Dashboarding**: The `src/metabase_prep.py` script selects the top SHAP features and creates a SQLite database (`results/feature_monitor.db`) containing only these features and the target variable.

4. **Metabase Dashboard**: Connect Metabase to the SQLite database and use the `shap_selected_features` table as the basis for your dashboard visualizations. This ensures the dashboard focuses on the most impactful factors for attrition. See [Metabase Setup Guide](docs/metabase_setup.md) for detailed instructions.
