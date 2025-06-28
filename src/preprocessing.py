# src/preprocessing.py

import pandas as pd
import numpy as np

def advanced_feature_engineering(df):
    """
    Performs our complete, optimized preprocessing and feature engineering pipeline.
    This function is intended to be imported and used by all notebooks.
    """
    df_copy = df.copy()
    
    # Convert 'Date' and handle potential negative sales by clipping at 0
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    if 'Weekly_Sales' in df_copy.columns:
        df_copy['Weekly_Sales'] = df_copy['Weekly_Sales'].clip(lower=0)

    # --- Feature: Specific Holidays (including Black Friday) ---
    thanksgiving_dates = pd.to_datetime(["2010-11-26", "2011-11-25"])
    super_bowl_dates = pd.to_datetime(["2010-02-12", "2011-02-11", "2012-02-10"])
    
    df_copy['IsBlackFridayWeek'] = df_copy.Date.isin(thanksgiving_dates).astype(int)
    df_copy['IsSuperBowlWeek'] = df_copy.Date.isin(super_bowl_dates).astype(int)
    
    # --- Feature: Basic Time-Based ---
    df_copy['Year'] = df_copy['Date'].dt.year
    df_copy['Month'] = df_copy['Date'].dt.month
    df_copy['WeekOfYear'] = df_copy['Date'].dt.isocalendar().week.astype(int)

    # --- Feature: Lag and Rolling (Time-Series Specific) ---
    df_copy.sort_values(by=['Store', 'Dept', 'Date'], inplace=True)
    if 'Weekly_Sales' in df_copy.columns:
        df_copy['Sales_Lag_1'] = df_copy.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1)
        df_copy['Sales_Lag_52'] = df_copy.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52)

    # --- Handling Missing Values ---
    df_copy[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']] = df_copy[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(0, inplace=True)

    # --- Encoding and Final Cleanup ---
    df_copy = pd.get_dummies(df_copy, columns=['Type'], prefix='Type')
    
    return df_copy
