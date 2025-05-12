from pycaret.classification import setup, create_model, tune_model, evaluate_model, plot_model, pull, save_model, load_model, predict_model
import pandas as pd

def setup_modeling(df, target='Attrition', session_id=123):
    """Setup PyCaret classification environment."""
    # Drop EmployeeId if it exists
    if 'EmployeeId' in df.columns:
        df = df.drop('EmployeeId', axis=1)
        
    clf = setup(
        data=df,
        target=target,
        numeric_features=['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
                        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
                        'PercentSalaryHike', 'StandardHours', 'TotalWorkingYears',
                        'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
                        'YearsSinceLastPromotion', 'YearsWithCurrManager',
                        'TenureRatio', 'OverallSatisfaction', 'SalaryToAgeRatio',
                        'SalaryToTenureRatio', 'PromotionRate', 'RoleStability',
                        'TravelImpact'],
        categorical_features=['BusinessTravel', 'Department', 'EducationField',
                            'Gender', 'JobRole', 'MaritalStatus', 'Over18',
                            'OverTime', 'AgeGroup'],
        ordinal_features={
            'Education': [1, 2, 3, 4, 5],
            'EnvironmentSatisfaction': [1, 2, 3, 4],
            'JobInvolvement': [1, 2, 3, 4],
            'JobLevel': [1, 2, 3, 4, 5],
            'JobSatisfaction': [1, 2, 3, 4],
            'PerformanceRating': [1, 2, 3, 4],
            'RelationshipSatisfaction': [1, 2, 3, 4],
            'StockOptionLevel': [0, 1, 2, 3],
            'WorkLifeBalance': [1, 2, 3, 4]
        },
        encoding_method='onehot',
        max_encoding_ohe=10,
        rare_to_value=0.05,
        rare_value='_rare_',
        normalize=True,
        feature_selection=True,
        feature_selection_estimator='lightgbm',
        session_id=session_id,
        train_size=0.7,
        fix_imbalance=True,
        fix_imbalance_method='smote',
        preprocess=True,
        verbose=False,
        memory=False
    )
    return clf

def train_and_tune_model(model_name='lda', optimize='Recall', n_iter=50):
    """Create and tune a model."""
    best_model = create_model(model_name, verbose=False)
    tuned_model = tune_model(best_model, optimize=optimize, n_iter=n_iter, verbose=False)
    return tuned_model

def evaluate_trained_model(model):
    """Evaluate a trained model using PyCaret's evaluate_model."""
    evaluate_model(model)

def plot_feature_importance(model):
    """Plot feature importance and return the importance DataFrame."""
    plot_model(model, plot='feature')
    importance_df = pull()
    return importance_df

def save_trained_model(model, path):
    """Save a trained model to disk."""
    save_model(model, path)

def load_trained_model(path):
    """Load a trained model from disk."""
    return load_model(path)

def predict_with_model(model, input_data):
    """Make predictions with a trained model."""
    return predict_model(model, data=input_data)