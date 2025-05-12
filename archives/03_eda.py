# %%
"""
02 Exploratory Data Analysis (EDA)
Visualize and summarize the cleaned employee attrition data.
"""
# %%
from src.eda_tools import plot_attrition_by_category, create_correlation_heatmap, plot_satisfaction_distribution, perform_eda, \
    get_numeric_summary, get_categorical_summary, get_value_counts, get_missing_values, t_test_by_attrition, chi_square_test, correlation_test, \
    plot_histograms, plot_boxplots_by_attrition, plot_barplots_for_categorical, plot_stacked_bar_chart, plot_pairplot, plot_violin_by_attrition, plot_categorical_heatmap
import pandas as pd
from IPython.display import display
from IPython.display import Markdown
# %%
display(Markdown("""
# Exploratory Data Analysis (EDA)
This notebook explores and visualizes the cleaned employee attrition dataset. We will summarize the data, visualize key patterns, and perform basic statistical tests to understand attrition drivers.
"""))
# %%
display(Markdown("""
## Data Preview
Let's load and preview the cleaned dataset to understand its structure and contents.
"""))
# %%
# Load feature-engineered data for EDA
clean_df = pd.read_csv('data/employee_data_features.csv')
clean_df.head()
# %%
display(Markdown("""
## Attrition by Department
This chart shows the distribution of attrition across different departments, helping us identify which departments have higher attrition rates.
"""))
# Plot attrition by Department
chart_dept = plot_attrition_by_category(clean_df, 'Department')
chart_dept.display()
chart_dept.save('eda_outputs/attrition_by_department.png')
# %%
display(Markdown("""
## Correlation Heatmap
The correlation heatmap visualizes relationships between numeric features, highlighting which variables are strongly correlated.
"""))
# Plot correlation heatmap
chart_corr = create_correlation_heatmap(clean_df)
chart_corr.display()
chart_corr.save('eda_outputs/correlation_heatmap.png')
# %%
display(Markdown("""
## Satisfaction Distribution by Attrition
This boxplot compares overall satisfaction scores between employees who left and those who stayed.
"""))
# Plot satisfaction distribution
chart_sat = plot_satisfaction_distribution(clean_df)
chart_sat.display()
chart_sat.save('eda_outputs/satisfaction_distribution.png')
# %%
display(Markdown("""
## All EDA Charts
Below are additional EDA charts for other key categories and features.
"""))
# Perform full EDA (all charts)
charts = perform_eda(clean_df)
for name, chart in charts.items():
    print(name)
    chart.display()
    chart.save(f'eda_outputs/{name}.png')
# %%
display(Markdown("""
## Descriptive Statistics
We summarize the dataset with numeric and categorical statistics, value counts, and missing value analysis.
"""))
# %%
display(Markdown("""
### Numeric Summary
Describes the central tendency and spread of numeric features.
"""))
# Descriptive statistics: Numeric summary
print('Numeric Summary:')
numeric_summary = get_numeric_summary(clean_df)
display(numeric_summary)
numeric_summary.to_csv('results/numeric_summary.csv')
# %%
display(Markdown("""
### Categorical Summary
Shows the frequency and diversity of values in categorical features.
"""))
# Descriptive statistics: Categorical summary
print('Categorical Summary:')
categorical_summary = get_categorical_summary(clean_df)
display(categorical_summary)
categorical_summary.to_csv('results/categorical_summary.csv')
# %%
display(Markdown("""
### Value Counts
Frequency of each category in categorical columns.
"""))
# Value counts for categorical columns
print('Value Counts:')
value_counts = get_value_counts(clean_df)
with open('results/value_counts.md', 'w') as f:
    f.write('# Value Counts\n')
    for col, counts in value_counts.items():
        f.write(f'\n## {col}\n')
        f.write(counts.to_markdown())
        f.write('\n')
for col, counts in value_counts.items():
    print(f'\n{col}:')
    print(counts)
# %%
display(Markdown("""
### Missing Values
Count and percentage of missing values per column.
"""))
# Missing values
print('Missing Values:')
missing_values = get_missing_values(clean_df)
display(missing_values)
missing_values.to_csv('results/missing_values.csv')
# %%
display(Markdown("""
## Inferential Statistics
We use statistical tests to compare groups and assess relationships between features.
"""))
# %%
display(Markdown("""
### T-test: MonthlyIncome by Attrition
Tests if the average monthly income differs between employees who left and those who stayed.
"""))
# Inferential statistics: T-test for MonthlyIncome by Attrition
if 'MonthlyIncome' in clean_df.columns and 'Attrition' in clean_df.columns:
    ttest_result = t_test_by_attrition(clean_df, 'MonthlyIncome')
    print('T-test for MonthlyIncome by Attrition:', ttest_result)
    with open('results/ttest_monthlyincome_by_attrition.md', 'w') as f:
        f.write('# T-test: MonthlyIncome by Attrition\n')
        f.write(str(ttest_result))
# %%
display(Markdown("""
### Chi-square: Department vs. Attrition
Tests if attrition rates differ significantly across departments.
"""))
# Inferential statistics: Chi-square test for Department vs. Attrition
if 'Department' in clean_df.columns and 'Attrition' in clean_df.columns:
    chi2_result = chi_square_test(clean_df, 'Department', 'Attrition')
    print('Chi-square test for Department vs. Attrition:', chi2_result)
    with open('results/chi2_department_vs_attrition.md', 'w') as f:
        f.write('# Chi-square: Department vs. Attrition\n')
        f.write(str(chi2_result))
# %%
display(Markdown("""
### Correlation: Age and MonthlyIncome
Assesses the linear relationship between age and monthly income.
"""))
# Inferential statistics: Correlation test for Age and MonthlyIncome
if 'Age' in clean_df.columns and 'MonthlyIncome' in clean_df.columns:
    corr_result = correlation_test(clean_df, 'Age', 'MonthlyIncome')
    print('Correlation test for Age and MonthlyIncome:', corr_result)
    with open('results/correlation_age_monthlyincome.md', 'w') as f:
        f.write('# Correlation: Age and MonthlyIncome\n')
        f.write(str(corr_result))
# %%
display(Markdown("""
## Visualizations of Key Features
We visualize distributions and relationships for important features to uncover patterns related to attrition.
"""))
# %%
display(Markdown("""
### Histograms
Show the distribution of key numeric features.
"""))
# Histograms for key numeric features
hist_charts = plot_histograms(clean_df, numeric_cols=['Age', 'MonthlyIncome', 'YearsAtCompany'] if all(col in clean_df.columns for col in ['Age', 'MonthlyIncome', 'YearsAtCompany']) else None)
for name, chart in hist_charts.items():
    print(f'Histogram: {name}')
    chart.display()
    chart.save(f'eda_outputs/histogram_{name}.png')
# %%
display(Markdown("""
### Boxplots by Attrition
Compare distributions of numeric features between attrition groups.
"""))
# Boxplots by attrition for key numeric features
box_charts = plot_boxplots_by_attrition(clean_df, numeric_cols=['Age', 'MonthlyIncome', 'YearsAtCompany'] if all(col in clean_df.columns for col in ['Age', 'MonthlyIncome', 'YearsAtCompany']) else None)
for name, chart in box_charts.items():
    print(f'Boxplot: {name}')
    chart.display()
    chart.save(f'eda_outputs/boxplot_{name}_by_attrition.png')
# %%
display(Markdown("""
### Bar Plots for Categorical Features
Show attrition counts for key categorical features.
"""))
# Bar plots for key categorical features
bar_charts = plot_barplots_for_categorical(clean_df, categorical_cols=['Department', 'JobRole', 'MaritalStatus'] if all(col in clean_df.columns for col in ['Department', 'JobRole', 'MaritalStatus']) else None)
for name, chart in bar_charts.items():
    print(f'Barplot: {name}')
    chart.display()
    chart.save(f'eda_outputs/barplot_{name}_by_attrition.png')
# %%
display(Markdown("""
### Stacked Bar Chart: Department by Attrition
Visualizes the proportion of attrition within each department.
"""))
# Stacked bar chart for Department by Attrition
if 'Department' in clean_df.columns:
    stacked_chart = plot_stacked_bar_chart(clean_df, 'Department')
    stacked_chart.display()
    stacked_chart.save('eda_outputs/stacked_bar_department_by_attrition.png')
# %%
display(Markdown("""
### Pairplot (Scatterplot Matrix)
Visualizes pairwise relationships between numeric features, colored by attrition.
"""))
# Pairplot (scatterplot matrix) for numeric features
pair_charts = plot_pairplot(clean_df, numeric_cols=['Age', 'MonthlyIncome', 'YearsAtCompany'] if all(col in clean_df.columns for col in ['Age', 'MonthlyIncome', 'YearsAtCompany']) else None)
for i, chart in enumerate(pair_charts):
    chart.display()
    chart.save(f'eda_outputs/pairplot_{i}.png')
# %%
display(Markdown("""
### Violin Plot: MonthlyIncome by Attrition
Shows the distribution and density of monthly income for each attrition group.
"""))
# Violin plot for MonthlyIncome by Attrition
if 'MonthlyIncome' in clean_df.columns:
    violin_chart = plot_violin_by_attrition(clean_df, 'MonthlyIncome')
    violin_chart.display()
    violin_chart.save('eda_outputs/violin_monthlyincome_by_attrition.png')
# %%
display(Markdown("""
### Heatmap: Department vs. JobRole
Shows the frequency of employees in each Department-JobRole combination.
"""))
# Heatmap for Department vs. JobRole
if 'Department' in clean_df.columns and 'JobRole' in clean_df.columns:
    heatmap_chart = plot_categorical_heatmap(clean_df, 'Department', 'JobRole')
    heatmap_chart.display()
    heatmap_chart.save('eda_outputs/heatmap_department_jobrole.png')
# %%
display(Markdown("""
# Findings, Insights, and Recommendations

## Findings & Insights

### 1. Demographics & Categorical Distributions
- **Department:** Most employees are in Research & Development (66%), followed by Sales (30%), with Human Resources being a small minority (4%).
- **Job Roles:** The largest groups are Sales Executive, Research Scientist, and Laboratory Technician.
- **Marital Status:** The majority are Married (44%), with Single (33%) and Divorced (23%) also represented.
- **Gender:** There are more males (59%) than females (41%).
- **Age Groups:** The largest age group is 26-35 (41%), followed by 36-45 (31%).
- **Related cell:** "### Categorical Summary" and "### Value Counts"
- **Output files:** `results/categorical_summary.csv`, `results/value_counts.md`

### 2. Numeric Summary
- **Age:** Mean: 37 years (range: 18–60).
- **Monthly Income:** Mean: $6,626 (range: $1,009–$19,999).
- **Years at Company:** Mean: 7.1 years (range: 0–40).
- **Related cell:** "### Numeric Summary"
- **Output file:** `results/numeric_summary.csv`

### 3. Attrition Patterns
- **Attrition Rate:** About 17% of employees have left (mean of Attrition column).
- **OverTime:** 29% of employees work overtime, which may be a risk factor for attrition.
- **Related cell:** "## Attrition by Department", "## All EDA Charts"
- **Output/plots:** `eda_outputs/attrition_by_jobrole.png`, `eda_outputs/attrition_by_agegroup.png`, and other plots in `eda_outputs/`

### 4. Statistical Tests
- **T-test (MonthlyIncome by Attrition):** Result: `nan` (likely due to missing or constant data in one group; check data integrity).
  - **Cell:** "### T-test: MonthlyIncome by Attrition"
  - **Output:** `results/ttest_monthlyincome_by_attrition.md`
- **Chi-square (Department vs. Attrition):** p-value ≈ 0.099 (not statistically significant at 0.05), suggesting department is not a strong predictor of attrition.
  - **Cell:** "### Chi-square: Department vs. Attrition"
  - **Output:** `results/chi2_department_vs_attrition.md`
- **Correlation (Age and MonthlyIncome):** Correlation: 0.50 (p < 0.001), indicating a moderate positive relationship—older employees tend to earn more.
  - **Cell:** "### Correlation: Age and MonthlyIncome"
  - **Output:** `results/correlation_age_monthlyincome.md`

### 5. Visual Insights (from eda_outputs/)
- Attrition is higher in certain job roles and age groups (see `eda_outputs/attrition_by_jobrole.png`, `eda_outputs/attrition_by_agegroup.png`).
- OverTime and Marital Status show visible differences in attrition rates.
- Correlation heatmap shows strong relationships between some numeric features (see `eda_outputs/correlation_heatmap.png`).
- Boxplots and violin plots reveal that employees who left often have lower satisfaction and different income distributions.
- **Related cells:** "## Visualizations of Key Features" and all subsequent visualization sections
- **Output/plots:** All relevant files in `eda_outputs/`

## Recommendations

1. **Focus Retention Efforts on At-Risk Groups:**
   - Target job roles and age groups with higher attrition.
   - Monitor employees working overtime and those with lower satisfaction scores.
   - **Supported by:** Categorical summary, value counts, attrition plots (see above)
2. **Further Investigate Data Issues:**
   - The t-test for MonthlyIncome by Attrition returned `nan`. Check for missing or constant values in the relevant groups.
   - **Supported by:** T-test cell and output
3. **Monitor and Support Employees:**
   - Implement programs to improve job satisfaction, especially for at-risk roles.
   - Review compensation and promotion policies for fairness and competitiveness.
   - **Supported by:** Numeric/categorical summaries, boxplots, violin plots
4. **Continue Data-Driven Monitoring:**
   - Regularly update the analysis as new data comes in.
   - Use the exported results and visualizations for ongoing reporting and management decisions.
   - **Supported by:** All summary and visualization outputs
""")) 