import pandas as pd

def engineer_features(df):
    """Create new features from existing data."""
    df_fe = df.copy()
    # Age groups
    df_fe['AgeGroup'] = pd.cut(df_fe['Age'], 
                              bins=[0, 25, 35, 45, 55, 100],
                              labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    # Tenure ratio
    df_fe['TenureRatio'] = df_fe.apply(
        lambda x: x['YearsAtCompany'] / x['TotalWorkingYears'] if x['TotalWorkingYears'] > 0 else 0,
        axis=1
    )
    # Satisfaction index
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                        'RelationshipSatisfaction', 'WorkLifeBalance']
    df_fe['OverallSatisfaction'] = df_fe[satisfaction_cols].mean(axis=1)
    # Salary-related features
    df_fe['SalaryToAgeRatio'] = df_fe['MonthlyIncome'] / df_fe['Age']
    df_fe['SalaryToTenureRatio'] = df_fe['MonthlyIncome'] / df_fe['YearsAtCompany'].replace(0, 1)
    # Career progression features
    df_fe['PromotionRate'] = df_fe['YearsAtCompany'] / df_fe['YearsSinceLastPromotion'].replace(0, 1)
    df_fe['RoleStability'] = df_fe['YearsInCurrentRole'] / df_fe['YearsAtCompany'].replace(0, 1)
    # Travel impact
    df_fe['TravelImpact'] = df_fe['BusinessTravel'].map({
        'Non-Travel': 0,
        'Travel_Rarely': 1,
        'Travel_Frequently': 2
    })
    return df_fe 