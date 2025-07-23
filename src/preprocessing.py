# src/preprocessing.py (Advanced V2)

import pandas as pd
import numpy as np

def advanced_feature_engineering(df):
    df_copy = df.copy()

    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    if 'Weekly_Sales' in df_copy.columns:
        df_copy['Weekly_Sales'] = df_copy['Weekly_Sales'].clip(lower=0)


    thanksgiving_dates = pd.to_datetime(["2010-11-26", "2011-11-25"])
    super_bowl_dates = pd.to_datetime(["2010-02-12", "2011-02-11", "2012-02-10"])
    labor_day_dates = pd.to_datetime(["2010-09-10", "2011-09-09", "2012-09-07"])
    christmas_dates = pd.to_datetime(["2010-12-31", "2011-12-30"])
    
    df_copy['IsBlackFridayWeek'] = df_copy.Date.isin(thanksgiving_dates).astype(int)
    df_copy['IsSuperBowlWeek'] = df_copy.Date.isin(super_bowl_dates).astype(int)
    df_copy['IsLaborDayWeek'] = df_copy.Date.isin(labor_day_dates).astype(int)
    df_copy['IsChristmasWeek'] = df_copy.Date.isin(christmas_dates).astype(int)
    
    # --- Feature 2: Basic Time-Based ---
    df_copy['Year'] = df_copy['Date'].dt.year
    df_copy['Month'] = df_copy['Date'].dt.month
    df_copy['WeekOfYear'] = df_copy['Date'].dt.isocalendar().week.astype(int)
    df_copy['Day'] = df_copy['Date'].dt.day

    # --- Feature 3: Advanced Lag and Rolling Features ---
    df_copy.sort_values(by=['Store', 'Dept', 'Date'], inplace=True)
    if 'Weekly_Sales' in df_copy.columns:
        # More Lag Features
        lags = [1, 2, 4, 13, 26, 52] # 1-2 weeks, 1 month, 1 quarter, half-year, 1 year
        for lag in lags:
            df_copy[f'Sales_Lag_{lag}'] = df_copy.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
            
        # More Rolling Features (over a 4-week window)
        rolling_window = 4
        df_copy['Sales_Roll_Mean_4'] = df_copy.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).rolling(rolling_window).mean()
        )
        df_copy['Sales_Roll_Std_4'] = df_copy.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).rolling(rolling_window).std()
        )
        df_copy['Sales_Roll_Min_4'] = df_copy.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).rolling(rolling_window).min()
        )
        df_copy['Sales_Roll_Max_4'] = df_copy.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(
            lambda x: x.shift(1).rolling(rolling_window).max()
        )

    # --- Feature 4: Markdown Features ---
    markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
    df_copy[markdown_cols] = df_copy[markdown_cols].fillna(0)
    df_copy['HasMarkdown'] = (df_copy[markdown_cols].sum(axis=1) > 0).astype(int)

    # -- Feature Group 5: Weekly Returns  --
    if 'Weekly_Sales' in df.columns:
        df['Weekly_Returns'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].pct_change()

    # --- Feature 6: Interaction Features ---
    # Create a combined Store and Department feature.
    # We will treat this as a category in LightGBM.
    df_copy['Store_Dept'] = df_copy['Store'].astype(str) + '_' + df_copy['Dept'].astype(str)
    df_copy['Store_Dept'] = df_copy['Store_Dept'].astype('category')

    # --- Encoding and Final Cleanup ---
    # One-hot encode 'Type'
    df_copy = pd.get_dummies(df_copy, columns=['Type'], prefix='Type')
    
    # --- Final Data Cleaning ---
    # Fill any remaining NaNs that were created by lag/rolling features
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(0, inplace=True)

    return df_copy
