# Data Preparation and Exploratory Data Analysis (EDA)

## 0. Data Split: Modeling vs. Inference Set
- The original dataset is split into two parts:
  - **Modeling set (80%)**: Used for all cleaning, validation, feature engineering, EDA, and model development.
  - **Inference set (20%)**: Held out and only minimally processed (dropping `Attrition` and `EmployeeId`) to simulate new/unseen employee data for deployment or inference testing.
- This split ensures that model evaluation and deployment are realistic and unbiased.

## 1. Data Loading
- The dataset is loaded from `data/employee_data.csv` using pandas.
- Initial checks include displaying the shape, info, and missing values of the DataFrame.
- Duplicate rows and duplicate values in columns are checked and summarized.

## 2. Data Validation (on Modeling Set Only)
- The script checks for:
  - The number of rows and columns
  - Data types of each column
  - Missing values per column
  - Duplicate rows and duplicate values in each column
- Results are displayed for review.

## 3. Data Cleaning (on Modeling Set Only)
- Duplicate rows are removed if found.
- The `Attrition` column is converted from 'Yes'/'No' to 1/0, and rows with missing `Attrition` are removed.
- Categorical columns are explicitly cast to the 'category' dtype.
- Ordinal columns (e.g., Education, JobLevel) are cast to integer types.
- Numeric columns are cast to appropriate integer types for memory efficiency.
- Years-related columns are also cast to integer types.

## 4. Feature Engineering (on Modeling Set Only)
- New features are created, including:
  - `AgeGroup`: Binned age categories
  - `TenureRatio`: Years at company divided by total working years
  - `OverallSatisfaction`: Mean of satisfaction-related columns
  - `SalaryToAgeRatio` and `SalaryToTenureRatio`: Salary normalized by age and tenure
  - `PromotionRate` and `RoleStability`: Career progression metrics
  - `TravelImpact`: Encoded business travel frequency
- The first few rows of the engineered DataFrame are displayed for inspection.

## 5. Exploratory Data Analysis (EDA, on Modeling Set Only)
- Several Altair charts are generated to visualize:
  - Attrition by Department, JobRole, and AgeGroup
  - Correlation heatmap for numeric features
  - Satisfaction distribution by attrition status
- Value counts and distributions are displayed for key categorical variables.
- EDA is performed interactively, with charts displayed in the notebook or saved as HTML.

## 6. Inference Set Preparation
- The inference set is only minimally processed:
  - `Attrition` and `EmployeeId` columns are dropped.
  - The resulting DataFrame simulates new employee data for deployment or API testing.
- No cleaning, feature engineering, or EDA is performed on the inference set to avoid data leakage.

## 7. Key Checks and Findings
- The data is checked for class imbalance in the `Attrition` column (on the modeling set).
- Categorical and ordinal features are validated for correct type and value ranges.
- Feature engineering is used to create interpretable and potentially predictive features.
- EDA highlights relationships between attrition and key variables (e.g., department, satisfaction).

## Data Pipeline Updates

The data pipeline now includes an additional step where SHAP feature importance is calculated after model training. The most important features, as determined by SHAP values, are selected and used to create a new SQLite database (`results/feature_monitor.db`). This database is used as the data source for the Metabase dashboard, ensuring that the visualizations reflect the most relevant factors influencing employee attrition.

Key output files:
- `results/shap_feature_importance.csv`: Contains SHAP values for all features
- `results/feature_monitor.db`: SQLite database with selected features for dashboarding

---

*This document summarizes the data preparation and EDA steps as implemented in the attrition analysis pipeline. Update as new steps or findings are added.* 