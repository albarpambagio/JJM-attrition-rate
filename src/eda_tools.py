import altair as alt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, pearsonr, spearmanr

def plot_attrition_by_category(df, column):
    """Plot attrition by category using Altair."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
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

def create_correlation_heatmap(df):
    """Create correlation heatmap using Altair."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataframe")
    corr_matrix = df[numeric_cols].corr().reset_index().melt('index')
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

def plot_satisfaction_distribution(df):
    """Create satisfaction distribution plot using Altair."""
    required_cols = ['OverallSatisfaction', 'Attrition']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
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

def perform_eda(df):
    """Perform exploratory data analysis using Altair. Returns a dict of charts."""
    charts = {}
    categories = ['Department', 'JobRole', 'AgeGroup']
    for category in categories:
        chart = plot_attrition_by_category(df, category)
        charts[f'attrition_by_{category.lower()}'] = chart
    charts['satisfaction_distribution'] = plot_satisfaction_distribution(df)
    return charts

def get_numeric_summary(df):
    """Return summary statistics for numeric columns."""
    return df.describe()

def get_categorical_summary(df):
    """Return summary statistics for categorical columns."""
    return df.describe(include='object')

def get_value_counts(df):
    """Return value counts for each categorical column as a dict."""
    return {col: df[col].value_counts() for col in df.select_dtypes(include='object').columns}

def get_missing_values(df):
    """Return missing value counts and percentages per column as a DataFrame."""
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    return pd.DataFrame({'missing_count': total, 'missing_percent': percent})

def t_test_by_attrition(df, column, target='Attrition', positive_class='Yes'):
    """Perform t-test for a numeric column between attrition groups."""
    group1 = df[df[target] == positive_class][column].dropna()
    group2 = df[df[target] != positive_class][column].dropna()
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    return {'t_stat': t_stat, 'p_value': p_val}

def chi_square_test(df, col1, col2):
    """Perform chi-square test between two categorical columns."""
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, ex = chi2_contingency(contingency)
    return {'chi2': chi2, 'p_value': p, 'dof': dof}

def correlation_test(df, col1, col2, method='pearson'):
    """Compute correlation and p-value between two numeric columns."""
    x, y = df[col1].dropna(), df[col2].dropna()
    if method == 'pearson':
        corr, p_val = pearsonr(x, y)
    else:
        corr, p_val = spearmanr(x, y)
    return {'correlation': corr, 'p_value': p_val}

def plot_histograms(df, numeric_cols=None):
    """Plot histograms for specified or all numeric columns using Altair."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    charts = {}
    for col in numeric_cols:
        chart = alt.Chart(df).mark_bar().encode(
            alt.X(f'{col}:Q', bin=alt.Bin(maxbins=30), title=col),
            y='count()',
            tooltip=[col, 'count()']
        ).properties(title=f'Histogram of {col}', width=300, height=200)
        charts[col] = chart
    return charts

def plot_boxplots_by_attrition(df, numeric_cols=None, target='Attrition'):
    """Plot boxplots for numeric columns by attrition status using Altair."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    charts = {}
    for col in numeric_cols:
        chart = alt.Chart(df).mark_boxplot().encode(
            y=alt.Y(f'{col}:Q', title=col),
            x=alt.X(f'{target}:N', title=target),
            color=f'{target}:N',
            tooltip=[col, target]
        ).properties(title=f'{col} by {target}', width=300, height=200)
        charts[col] = chart
    return charts

def plot_barplots_for_categorical(df, categorical_cols=None, target='Attrition'):
    """Plot bar plots for categorical features showing attrition counts using Altair."""
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include='object').columns
    charts = {}
    for col in categorical_cols:
        if col == target:
            continue
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{col}:N', title=col, sort='-y'),
            y='count()',
            color=f'{target}:N',
            tooltip=[col, target, 'count()']
        ).properties(title=f'{col} by {target}', width=300, height=200)
        charts[col] = chart
    return charts

def plot_stacked_bar_chart(df, categorical_col, target='Attrition'):
    """Plot a stacked bar chart for a categorical column by attrition status using Altair."""
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f'{categorical_col}:N', title=categorical_col),
        y=alt.Y('count()', stack='normalize', title='Proportion'),
        color=f'{target}:N',
        tooltip=[categorical_col, target, 'count()']
    ).properties(title=f'Stacked Bar: {categorical_col} by {target}', width=300, height=200)
    return chart

def plot_pairplot(df, numeric_cols=None, target='Attrition'):
    """Create a scatterplot matrix (pairplot) for numeric columns colored by attrition using Altair."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    charts = []
    for i, col_x in enumerate(numeric_cols):
        for j, col_y in enumerate(numeric_cols):
            if i < j:
                chart = alt.Chart(df).mark_circle(size=30, opacity=0.5).encode(
                    x=alt.X(f'{col_x}:Q', title=col_x),
                    y=alt.Y(f'{col_y}:Q', title=col_y),
                    color=f'{target}:N',
                    tooltip=[col_x, col_y, target]
                ).properties(width=200, height=200, title=f'{col_x} vs {col_y}')
                charts.append(chart)
    return charts

def plot_violin_by_attrition(df, column, target='Attrition'):
    """Plot a violin plot for a numeric column by attrition status using Altair (approximate using density + boxplot)."""
    base = alt.Chart(df).transform_density(
        column,
        as_=[column, 'density'],
        groupby=[target]
    )
    violin = base.mark_area(orient='horizontal').encode(
        y=alt.Y(f'{column}:Q', title=column),
        x=alt.X('density:Q', stack='center', impute=None, title=None),
        color=f'{target}:N',
        tooltip=[column, target]
    ).properties(width=100, height=300)
    box = alt.Chart(df).mark_boxplot(size=30).encode(
        y=alt.Y(f'{column}:Q', title=column),
        x=alt.X(f'{target}:N', title=target),
        color=f'{target}:N'
    )
    return violin | box

def plot_categorical_heatmap(df, col1, col2, target='Attrition'):
    """Plot a heatmap of counts for two categorical columns using Altair."""
    heatmap_data = df.groupby([col1, col2]).size().reset_index(name='count')
    chart = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X(f'{col1}:N', title=col1),
        y=alt.Y(f'{col2}:N', title=col2),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='blues')),
        tooltip=[col1, col2, 'count']
    ).properties(title=f'Heatmap of {col1} vs {col2}', width=300, height=300)
    return chart 