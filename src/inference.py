from pycaret.classification import load_model, predict_model
import pandas as pd

def predict_attrition(input_data, model_path='models/final_lda_model'):
    """
    Load the saved LDA model and make predictions on new data.
    Args:
        input_data (pd.DataFrame): DataFrame with the same features as used in training (no Attrition or EmployeeId)
        model_path (str): Path to the saved model
    Returns:
        pd.DataFrame: DataFrame with predictions and probabilities
    """
    model = load_model(model_path)
    predictions = predict_model(model, data=input_data)
    return predictions 