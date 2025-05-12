import os
import warnings
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
warnings.filterwarnings("ignore")


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from src.feature_engineering import engineer_features
from src.inference import predict_attrition

app = FastAPI(title="Attrition Prediction API", description="Predict employee attrition using a trained LDA model.")

class EmployeeRecord(BaseModel):
    # Define all input fields except Attrition (and EmployeeId is optional for tracking)
    EmployeeId: Optional[int]
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EmployeeCount: int
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    Over18: str
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StandardHours: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

@app.get("/")
def root():
    return {"message": "Attrition Prediction API. Use /predict to get attrition predictions."}

@app.post("/predict")
def predict(records: List[EmployeeRecord]):
    """
    Predict attrition for a list of employees.
    Example request:
    [
      {"EmployeeId": 1, "Age": 35, ...},
      {"EmployeeId": 2, "Age": 42, ...}
    ]
    """
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([r.dict() for r in records])
        # Save EmployeeId for tracking if present
        employee_ids = df['EmployeeId'].reset_index(drop=True) if 'EmployeeId' in df.columns else None
        # Drop EmployeeId before feature engineering and prediction
        if 'EmployeeId' in df.columns:
            df = df.drop(columns=['EmployeeId'])
        # Apply feature engineering
        df_fe = engineer_features(df)
        # Preprocess categorical columns
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                           'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'AgeGroup']
        for col in categorical_cols:
            df_fe[col] = df_fe[col].astype(str).str.replace(' ', '_').str.replace('&', '_and_')
        # Predict
        preds = predict_attrition(df_fe)
        # Add EmployeeId back if present
        if employee_ids is not None:
            preds = pd.concat([employee_ids, preds.reset_index(drop=True)], axis=1)
        # Return predictions as list of dicts
        return preds.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 