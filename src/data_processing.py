import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load employee data from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the dataset by handling missing values and adjusting data types."""
    df_clean = df.copy()
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    # Convert Attrition to numeric if needed
    if df_clean['Attrition'].dtype == 'object':
        df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
    df_clean = df_clean[df_clean['Attrition'].notna()]
    # Convert categorical columns
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                       'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype('category')
    # Convert ordinal columns
    ordinal_cols = {
        'Education': 'int8',
        'EnvironmentSatisfaction': 'int8',
        'JobInvolvement': 'int8',
        'JobLevel': 'int8',
        'JobSatisfaction': 'int8',
        'PerformanceRating': 'int8',
        'RelationshipSatisfaction': 'int8',
        'StockOptionLevel': 'int8',
        'WorkLifeBalance': 'int8'
    }
    for col, dtype in ordinal_cols.items():
        df_clean[col] = df_clean[col].astype(dtype)
    # Convert numeric columns
    df_clean['Age'] = df_clean['Age'].astype('int8')
    df_clean['DailyRate'] = df_clean['DailyRate'].astype('int16')
    df_clean['HourlyRate'] = df_clean['HourlyRate'].astype('int16')
    df_clean['MonthlyIncome'] = df_clean['MonthlyIncome'].astype('int32')
    df_clean['MonthlyRate'] = df_clean['MonthlyRate'].astype('int32')
    df_clean['PercentSalaryHike'] = df_clean['PercentSalaryHike'].astype('int8')
    df_clean['StandardHours'] = df_clean['StandardHours'].astype('int8')
    df_clean['TrainingTimesLastYear'] = df_clean['TrainingTimesLastYear'].astype('int8')
    years_cols = ['NumCompaniesWorked', 'TotalWorkingYears', 'YearsAtCompany',
                 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    for col in years_cols:
        df_clean[col] = df_clean[col].astype('int8')
    return df_clean

def split_data(df, test_size=0.2, random_state=42):
    """Split data into modeling and inference sets."""
    df_model, df_infer = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Attrition'])
    return df_model, df_infer 