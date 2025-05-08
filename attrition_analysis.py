#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: jupytext,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
"""
Employee Attrition Analysis
This script implements the execution plan for analyzing employee attrition data and building a predictive model.
"""

# Import required libraries
import pandas as pd
import numpy as np
import altair as alt
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Configure Altair
alt.data_transformers.enable('default')
alt.theme.enable('default')


# %% [markdown]
# ## 2.1 Split Data for Modeling and Inference
# We split the original data into a modeling set (for training/validation) and an inference set (to simulate new/unseen employee data).

# %%
# Load data
df = pd.read_csv('data/employee_data.csv')

# Remove rows with missing Attrition (as in cleaning)
df = df[df['Attrition'].notna()]

# Split: 80% for modeling, 20% for inference (simulate new data)
df_model, df_infer = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Attrition'])

# Use df_model for all modeling steps below

# %% [markdown]
# ## 2.2 Data Cleaning (on Modeling Set Only)

# %%
def clean_data(df):
    """Clean the dataset by handling missing values and adjusting data types."""
    print("\nCleaning data...")
    df_clean = df.copy()
    
    # Handle duplicates
    print("\nChecking for duplicates...")
    # Check for duplicate rows
    duplicate_rows = df_clean.duplicated().sum()
    if duplicate_rows > 0:
        print(f"Found {duplicate_rows} duplicate rows. Removing them...")
        df_clean = df_clean.drop_duplicates()
    
    # Check for duplicate values in each column
    duplicate_by_column = pd.Series({
        column: df_clean[column].duplicated().sum() 
        for column in df_clean.columns 
        if df_clean[column].duplicated().sum() > 0
    })
    if not duplicate_by_column.empty:
        print("\nDuplicate values by column:")
        print(duplicate_by_column)
    
    # Handle Attrition column
    print("\nAttrition values before cleaning:")
    print(df_clean['Attrition'].value_counts(dropna=False))
    
    # If Attrition is not already numeric, convert it
    if df_clean['Attrition'].dtype == 'object':
        df_clean['Attrition'] = df_clean['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Remove rows where Attrition is missing
    df_clean = df_clean[df_clean['Attrition'].notna()]
    
    print("\nAttrition values after cleaning:")
    print(df_clean['Attrition'].value_counts(dropna=False))
    
    # Convert categorical columns to appropriate type
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                       'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype('category')
    
    # Convert ordinal columns to appropriate type (1-4 or 1-5 scale)
    ordinal_cols = {
        'Education': 'int8',  # 1-Below College to 5-Doctor
        'EnvironmentSatisfaction': 'int8',  # 1-Low to 4-Very High
        'JobInvolvement': 'int8',  # 1-Low to 4-Very High
        'JobLevel': 'int8',  # 1 to 5
        'JobSatisfaction': 'int8',  # 1-Low to 4-Very High
        'PerformanceRating': 'int8',  # 1-Low to 4-Outstanding
        'RelationshipSatisfaction': 'int8',  # 1-Low to 4-Very High
        'StockOptionLevel': 'int8',  # 0 to 3
        'WorkLifeBalance': 'int8'  # 1-Low to 4-Outstanding
    }
    for col, dtype in ordinal_cols.items():
        df_clean[col] = df_clean[col].astype(dtype)
    
    # Convert numeric columns to appropriate types
    df_clean['Age'] = df_clean['Age'].astype('int8')
    df_clean['DailyRate'] = df_clean['DailyRate'].astype('int16')
    df_clean['HourlyRate'] = df_clean['HourlyRate'].astype('int16')
    df_clean['MonthlyIncome'] = df_clean['MonthlyIncome'].astype('int32')
    df_clean['MonthlyRate'] = df_clean['MonthlyRate'].astype('int32')
    df_clean['PercentSalaryHike'] = df_clean['PercentSalaryHike'].astype('int8')
    df_clean['StandardHours'] = df_clean['StandardHours'].astype('int8')
    df_clean['TrainingTimesLastYear'] = df_clean['TrainingTimesLastYear'].astype('int8')
    
    # Convert years-related columns to appropriate types
    years_cols = ['NumCompaniesWorked', 'TotalWorkingYears', 'YearsAtCompany',
                 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    for col in years_cols:
        df_clean[col] = df_clean[col].astype('int8')
    
    return df_clean

# Clean data (use only df_model)
df_clean = clean_data(df_model)

# %%
df_clean.head(10)

# %% [markdown]
# ## 2.3 Data Validation and Exploration (on Modeling Set Only)

# %%
display(Markdown('### Data Validation: Modeling Set'))

display(Markdown(f'**Shape:** {df_model.shape}'))

display(Markdown('**Column Data Types:**'))
display(df_model.dtypes.to_frame('Dtype').T)

display(Markdown('**Missing Values per Column:**'))
display(df_model.isna().sum().to_frame('Missing Values').T)

display(Markdown(f'**Duplicate Rows:** {df_model.duplicated().sum()}'))

display(Markdown('**Quick Summary Statistics:**'))
display(df_model.describe(include='all'))

# %% [markdown]
# ## Prepare Inference Data (for Deployment/Testing)
# This is a held-out set, simulating new employee data. Drop Attrition and EmployeeId.

# %%
df_infer_input = df_infer.drop(columns=['Attrition', 'EmployeeId'])
# df_infer_input can be used for local deployment/inference testing

# %% [markdown]
# ## 4. Feature Engineering

# %%
def engineer_features(df):
    """Create new features from existing data."""
    print("\nEngineering features...")
    df_fe = df.copy()
    
    # Create age groups
    df_fe['AgeGroup'] = pd.cut(df_fe['Age'], 
                              bins=[0, 25, 35, 45, 55, 100],
                              labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Calculate tenure ratios (handle division by zero)
    df_fe['TenureRatio'] = df_fe.apply(
        lambda x: x['YearsAtCompany'] / x['TotalWorkingYears'] if x['TotalWorkingYears'] > 0 else 0,
        axis=1
    )
    
    # Create satisfaction index
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 
                        'RelationshipSatisfaction', 'WorkLifeBalance']
    df_fe['OverallSatisfaction'] = df_fe[satisfaction_cols].mean(axis=1)
    
    # Create salary-related features
    df_fe['SalaryToAgeRatio'] = df_fe['MonthlyIncome'] / df_fe['Age']
    df_fe['SalaryToTenureRatio'] = df_fe['MonthlyIncome'] / df_fe['YearsAtCompany'].replace(0, 1)
    
    # Create career progression features
    df_fe['PromotionRate'] = df_fe['YearsAtCompany'] / df_fe['YearsSinceLastPromotion'].replace(0, 1)
    df_fe['RoleStability'] = df_fe['YearsInCurrentRole'] / df_fe['YearsAtCompany'].replace(0, 1)
    
    # Create travel impact feature
    df_fe['TravelImpact'] = df_fe['BusinessTravel'].map({
        'Non-Travel': 0,
        'Travel_Rarely': 1,
        'Travel_Frequently': 2
    })
    
    print("\nNew features created:")
    print(df_fe[['AgeGroup', 'TenureRatio', 'OverallSatisfaction', 
                'SalaryToAgeRatio', 'PromotionRate', 'RoleStability']].head())
    
    return df_fe

# Engineer features
df_fe = engineer_features(df_clean)

# %%
df_fe.head(10)


# %% [markdown]
# ## 5. Exploratory Data Analysis

# %%
def plot_attrition_by_category(df, column):
    """Plot attrition by category using Altair.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to plot
    
    Returns:
        alt.Chart: Altair chart object
    """
    try:
        # Validate input
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        
        # Create the chart
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{column}:N', title=column, sort='-y'),
            y=alt.Y('count()', title='Count'),
            color='Attrition:N',
            tooltip=['Attrition', 'count()']
        ).properties(
            title=f'Attrition by {column}',
            width=600,
            height=400
        ).interactive()
        
        return chart
        
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        return None

def create_correlation_heatmap(df):
    """Create correlation heatmap using Altair.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        alt.Chart: Altair chart object
    """
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in dataframe")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().reset_index().melt('index')
        
        # Create heatmap
        chart = alt.Chart(corr_matrix).mark_rect().encode(
            x='index:N',
            y='variable:N',
            color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue')),
            tooltip=['index', 'variable', 'value']
        ).properties(
            title='Correlation Matrix',
            width=800,
            height=800
        ).interactive()
        
        return chart
        
    except Exception as e:
        print(f"Error creating correlation heatmap: {str(e)}")
        return None

def plot_satisfaction_distribution(df):
    """Create satisfaction distribution plot using Altair.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        alt.Chart: Altair chart object
    """
    try:
        # Validate required columns
        required_cols = ['OverallSatisfaction', 'Attrition']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Create satisfaction distribution
        chart = alt.Chart(df).mark_boxplot().encode(
            y=alt.Y('OverallSatisfaction:Q', title='Overall Satisfaction'),
            x=alt.X('Attrition:N', title='Attrition Status'),
            color='Attrition:N',
            tooltip=['OverallSatisfaction', 'Attrition']
        ).properties(
            title='Satisfaction Distribution by Attrition Status',
            width=400,
            height=300
        ).interactive()
        
        return chart
        
    except Exception as e:
        print(f"Error creating satisfaction distribution plot: {str(e)}")
        return None

def perform_eda(df):
    """Perform exploratory data analysis using Altair.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        dict: Dictionary containing all created charts
    """
    print("\nPerforming exploratory data analysis...")
    
    # Dictionary to store all charts
    charts = {}
    
    # Plot attrition by various categories
    categories = ['Department', 'JobRole', 'AgeGroup']
    for category in categories:
        chart = plot_attrition_by_category(df, category)
        if chart:
            charts[f'attrition_by_{category.lower()}'] = chart
            display(chart)  # Display chart in notebook
    
    # Create correlation heatmap
    corr_chart = create_correlation_heatmap(df)
    if corr_chart:
        charts['correlation_matrix'] = corr_chart
        display(corr_chart)  # Display chart in notebook
    
    # Create satisfaction distribution
    sat_chart = plot_satisfaction_distribution(df)
    if sat_chart:
        charts['satisfaction_distribution'] = sat_chart
        display(sat_chart)  # Display chart in notebook
    
    return charts

# %%
df_fe['Department'].value_counts()

# %%
display(plot_attrition_by_category(df_fe, 'Department'))

# %%
display(create_correlation_heatmap(df_fe))

# %%
display(plot_satisfaction_distribution(df_fe))

# %% [markdown]
# ## 6. Model Development

# %%
# Preprocess categorical columns to ensure consistent naming
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                   'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'AgeGroup']

for col in categorical_cols:
    df_fe[col] = df_fe[col].astype(str).str.replace(' ', '_').str.replace('&', '_and_')
    
if 'EmployeeId' in df_fe.columns:
    df_fe = df_fe.drop(columns=['EmployeeId'])
    
clf = setup(data=df_fe, 
            target='Attrition',
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
            # Encoding parameters to ensure consistency
            encoding_method='onehot',  # Explicitly set encoding method
            max_encoding_ohe=10,       # Limit one-hot encoding categories
            rare_to_value=0.05,        # Group rare categories
            rare_value='_rare_',       # Name for rare category group
            
            # Other parameters
            normalize=True,
            feature_selection=True,
            session_id=123,
            train_size=0.7,
            fix_imbalance=True,
            fix_imbalance_method='smote',
            
            # Additional troubleshooting parameters
            preprocess=True,           # Ensure preprocessing is enabled
            verbose=True,             # Show more details
            memory=False)             # Disable memory caching for debugging


# %%
# best_model = compare_models()

# %% [markdown]
# ### 6.2 Tune the Best Linear Model for Recall

# %%
best_linear = create_model('lda')  # or 'ridge', 'lr'
tuned_linear = tune_model(best_linear, optimize='Recall', n_iter=50)
evaluate_model(tuned_linear)

# %% [markdown]
# ### 6.3 Ensemble the Tuned Linear Model

# %%
# ensemble_lr = ensemble_model(tuned_linear, method='Bagging')
# evaluate_model(ensemble_lr)

# %% [markdown]
# ### 6.4 Blend with Other Strong Models

# %%
# tuned_rf = tune_model(create_model('rf'), optimize='Recall', n_iter=50)
# tuned_gbc = tune_model(create_model('gbc'), optimize='Recall', n_iter=50)
# blended = blend_models([tuned_linear, tuned_rf, tuned_gbc], optimize='Recall')
# evaluate_model(blended)

# %% [markdown]
# ### 6.5 Stack Models

# %%
# stacked = stack_models([tuned_linear, tuned_rf, tuned_gbc], optimize='Recall')
# evaluate_model(stacked) 

# %% [markdown]
# ### 6.6 Feature Importance Analysis (LDA Chosen for Interpretability)
# Although multiple models were compared and tuned, LDA is chosen for feature importance analysis due to its interpretability and recall performance.

# %%
# Use the tuned LDA model for feature importance, regardless of previous best/ensemble/blend/stack results
tuned_lda = tuned_linear  # tuned_linear is the tuned LDA model from earlier
feature_importance_plot = plot_model(tuned_lda, plot='feature')
display(feature_importance_plot)

# %% [markdown]
# ## 7. Deployment Preparation
# Save the final preprocessing pipeline and tuned LDA model for use in an API or other deployment scenario.

# %%
from pycaret.classification import save_model, load_model

# Save the tuned LDA model to disk
save_model(tuned_lda, 'models/final_lda_model')

# %%
def predict_attrition(input_data):
    """
    Load the saved LDA model and make predictions on new data.
    Args:
        input_data (pd.DataFrame): DataFrame with the same features as used in training (no Attrition or EmployeeId)
    Returns:
        pd.DataFrame: DataFrame with predictions and probabilities
    """
    model = load_model('models/final_lda_model')
    predictions = predict_model(model, data=input_data)
    return predictions

# Save EmployeeId for tracking
employee_ids = df_infer_input['EmployeeId'].reset_index(drop=True)

# Drop EmployeeId before feature engineering and prediction
infer_features = df_infer_input.drop(columns=['EmployeeId'])

# Apply feature engineering to inference data
df_infer_fe = engineer_features(infer_features)

# Preprocess categorical columns to ensure consistent naming (as in training)
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 
                   'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime', 'AgeGroup']
for col in categorical_cols:
    df_infer_fe[col] = df_infer_fe[col].astype(str).str.replace(' ', '_').str.replace('&', '_and_')

# Predict
preds_infer = predict_attrition(df_infer_fe)

# Add EmployeeId back to the predictions for tracking
preds_infer = pd.concat([employee_ids, preds_infer.reset_index(drop=True)], axis=1)
display(preds_infer.head())


# %%
