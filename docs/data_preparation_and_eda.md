# Data Preparation and Exploratory Data Analysis (EDA)

## 1. Data Loading
- The dataset is loaded from `data/employee_data.csv` using pandas.
- Initial checks include displaying the shape, info, and missing values of the DataFrame.
- Duplicate rows and duplicate values in columns are checked and summarized.

## 2. Data Validation
- The script checks for:
  - The number of rows and columns
  - Data types of each column
  - Missing values per column
  - Duplicate rows and duplicate values in each column
- Results are displayed for review.

## 3. Data Cleaning
- Duplicate rows are removed if found.
- The `Attrition` column is converted from 'Yes'/'No' to 1/0, and missing values are filled with 0 (assuming 'No').
- Categorical columns are explicitly cast to the 'category' dtype.
- Ordinal columns (e.g., Education, JobLevel) are cast to integer types.
- Numeric columns are cast to appropriate integer types for memory efficiency.
- Years-related columns are also cast to integer types.

## 4. Feature Engineering
- New features are created, including:
  - `AgeGroup`: Binned age categories
  - `TenureRatio`: Years at company divided by total working years
  - `OverallSatisfaction`: Mean of satisfaction-related columns
  - `SalaryToAgeRatio` and `SalaryToTenureRatio`: Salary normalized by age and tenure
  - `PromotionRate` and `RoleStability`: Career progression metrics
  - `TravelImpact`: Encoded business travel frequency
- The first few rows of the engineered DataFrame are displayed for inspection.

## 5. Exploratory Data Analysis (EDA)
- Several Altair charts are generated to visualize:
  - Attrition by Department, JobRole, and AgeGroup
  - Correlation heatmap for numeric features
  - Satisfaction distribution by attrition status
- Value counts and distributions are displayed for key categorical variables.
- EDA is performed interactively, with charts displayed in the notebook or saved as HTML.

## 6. Key Checks and Findings
- The data is checked for class imbalance in the `Attrition` column.
- Categorical and ordinal features are validated for correct type and value ranges.
- Feature engineering is used to create interpretable and potentially predictive features.
- EDA highlights relationships between attrition and key variables (e.g., department, satisfaction).

---

*This document summarizes the data preparation and EDA steps as implemented in the attrition analysis pipeline. Update as new steps or findings are added.* 