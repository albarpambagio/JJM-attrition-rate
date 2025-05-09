# Employee Attrition Prediction API

## 📁 Project Structure

```
JJM-attrition-rate/
├── data/                        # Raw and processed data files
├── models/                      # Saved models and model artifacts
├── eda_outputs/                 # EDA result files (plots, tables, etc.)
├── notebook/                    # All notebook-style scripts (for Jupytext or .ipynb)
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_modeling.py
│   └── 05_inference.py
├── src/                         # All Python modules (reusable code)
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── eda_tools.py
│   ├── modeling.py
│   └── inference.py
├── api.py                       # FastAPI app for predictions
├── attrition_analysis.py        # (ARCHIVED: legacy monolithic script) # TODO: update project structure
├── README.md
├── pyproject.toml
├── .gitignore
├── todo.md
└── logs.log
```

- **All reusable code** (functions, classes) lives in `src/`.
- **All notebook-style scripts** (for step-by-step analysis, EDA, modeling, etc.) live in `notebook/`.
- **Data** and **models** are in their respective folders.
- **`attrition_analysis.py`** is archived for reference only.

---

## 🛠️ How to Import Modules in Notebooks/Scripts

At the top of each notebook in `notebook/`, add:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
```
Then import modules as:
```python
from data_processing import load_data, clean_data, split_data
from feature_engineering import engineer_features
# ...etc.
```

---

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

### Future Works
- For a robust, production-ready deployment, the following are recommended:
  - Input validation (check for missing columns, correct dtypes, etc.)
  - Error handling (bad input, model load errors, etc.)
  - Logging and monitoring
  - Security (authentication, rate limiting, etc.)
  - API endpoint documentation
- [See docs/future_works.md for more](docs/future_works.md)

---

## 🚀 API Deployment & Usage

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

### 3. API Documentation
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 4. Example: Predict Attrition
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

---

## 📄 More Information
- The API applies all feature engineering and preprocessing steps as in model training.
- For more details, see the code in `api.py` and the modules in `src/`.

---
