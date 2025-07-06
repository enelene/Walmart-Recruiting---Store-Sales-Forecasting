# Databricks notebook source

print("--- SECTION 1: SETUP AND DATA LOADING ---")

# Ensure necessary libraries are installed
%pip install pandas matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

# Load the final processed data from your DBFS path
PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
TRAIN_PATH = os.path.join(PROCESSED_DIR, 'train_processed_final.csv')

try:
    df = pd.read_csv(TRAIN_PATH, parse_dates=['Date'])
    print("Successfully loaded processed training data.")
except FileNotFoundError:
    print(f"ERROR: Processed data not found at '{TRAIN_PATH}'.")
    dbutils.notebook.exit("Please run the data preparation notebook first.")

# ==============================================================================
# SECTION 2: VISUALIZATION - SALES HEATMAP BY MONTH AND YEAR
# ==============================================================================
print("\n--- SECTION 2: VISUALIZATION - SALES HEATMAP BY MONTH AND YEAR ---")

# Create a pivot table to aggregate average sales by month and year
monthly_sales = df.pivot_table(index='Month', columns='Year', values='Weekly_Sales', aggfunc='mean')

# Plot the heatmap
sns.heatmap(monthly_sales, cmap='viridis', annot=True, fmt='.0f')
plt.title('Average Weekly Sales by Month and Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Month', fontsize=12)
plt.show()

print("""
### INSIGHTS FROM THE SALES HEATMAP:

1.  **Strong Yearly Seasonality:** The pattern is very consistent across all three years. Sales are always highest in December (Month 12) and November (Month 11), which directly corresponds to the holiday shopping season (Thanksgiving, Black Friday, Christmas).
2.  **Peak Sales Month:** December is clearly the "hottest" month, with the highest average sales.
3.  **Summer Lull:** There appears to be a slight dip in sales during the summer months (June-August).
4.  **Justification for Features:** This plot visually confirms why our `Month`, `WeekOfYear`, `IsBlackFridayWeek`, and `IsChristmasWeek` features are so important. They directly capture this dominant seasonal pattern.
""")

# ==============================================================================
# SECTION 3: VISUALIZATION - CORRELATION OF NUMERICAL FEATURES
# ==============================================================================
print("\n--- SECTION 3: VISUALIZATION - CORRELATION OF NUMERICAL FEATURES ---")

# Select key numerical features to analyze
numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
correlation_matrix = df[numeric_cols].corr()

# Plot the correlation heatmap
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()

print("""
### INSIGHTS FROM THE CORRELATION MATRIX:

1.  **Store `Size` is Key:** `Size` has the strongest positive correlation with `Weekly_Sales` (0.24). This makes perfect sense: larger stores have more products and customers, leading to higher sales. This is a very important predictive feature.
2.  **Economic Factors are Weakly Correlated:** `CPI` and `Unemployment` have very weak linear correlations with sales (-0.07 and -0.11). This doesn't mean they are useless, but it suggests their effect is not a simple linear one and might be more complex or vary by region.
3.  **`Temperature` and `Fuel_Price`:** These also have very weak correlations. Their impact is likely non-linear and might only affect certain departments (e.g., temperature affecting clothing sales).
4.  **Justification for Models:** The weak linear correlations show that a simple linear regression model would perform poorly. This justifies the need for more complex, non-linear models like LightGBM that can capture the intricate relationships between features.
""")

# ==============================================================================
# SECTION 4: VISUALIZATION - SALES DISTRIBUTION BY DEPARTMENT
# ==============================================================================
print("\n--- SECTION 4: VISUALIZATION - SALES DISTRIBUTION BY DEPARTMENT ---")

# To make the plot readable, let's analyze the departments for a single, large store (e.g., Store 4)
store_4_df = df[df['Store'] == 4]

# Order departments by their median sales for a cleaner plot
ordered_depts = store_4_df.groupby('Dept')['Weekly_Sales'].median().sort_values(ascending=False).index

plt.figure(figsize=(18, 8))
sns.boxplot(x='Dept', y='Weekly_Sales', data=store_4_df, order=ordered_depts)
plt.title('Weekly Sales Distribution by Department for Store 4', fontsize=16)
plt.xlabel('Department', fontsize=12)
plt.ylabel('Weekly Sales', fontsize=12)
plt.xticks(rotation=90)
plt.show()

print("""
### INSIGHTS FROM THE DEPARTMENT BOXPLOT:

1.  **Huge Variation:** There is an enormous difference in sales volume between departments. Some departments (like 92 and 95) are massive revenue drivers, while many others have very low sales.
2.  **Different Volatility:** The size of the boxes varies greatly. Some departments have very consistent, predictable sales (short boxes), while others are highly volatile (tall boxes).
3.  **Justification for `groupby`:** This plot is the ultimate justification for why we **must** `groupby(['Store', 'Dept'])` when creating our lag and rolling features. Treating all departments the same would be a huge mistake. The sales history of Dept 92 has no bearing on the sales history of Dept 28.
4.  **Justification for `Store_Dept` Feature:** This also shows why creating a combined `Store_Dept` categorical feature is a powerful idea. It allows the model to learn a unique behavior for every single box on this chart.
""")

