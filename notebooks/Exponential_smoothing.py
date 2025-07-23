# Databricks notebook source
!pip install -r requirements.txt
!pip install dagshub
import pandas as pd
import numpy as np
import mlflow
import os
import zipfile
from src.preprocessing import advanced_feature_engineering

# COMMAND ----------

# MAGIC %md
# MAGIC The Core Idea: Weighted Averages
# MAGIC At its heart, all exponential smoothing is about forecasting the future by looking at a weighted average of the past. The key innovation is that it gives more weight to recent observations and exponentially less weight to older observations. This is intuitive—what happened last week is probably more relevant to this week's sales than what happened last year.
# MAGIC
# MAGIC Double Exponential Smoothing (Holt's Linear Trend Method)
# MAGIC This is the second level of exponential smoothing. It's designed for data that has a trend, but no seasonality.
# MAGIC
# MAGIC What it learns: It builds on Single Exponential Smoothing by adding a second component: the trend. So, it learns and forecasts two things:
# MAGIC
# MAGIC Level: The baseline average value of the series.
# MAGIC
# MAGIC Trend: The slope of the series (is it generally increasing or decreasing?).
# MAGIC
# MAGIC How it works: It maintains two smoothed values—one for the level and one for the trend—and uses both to produce a forecast. The forecast is a straight line that follows the estimated trend.
# MAGIC
# MAGIC For your Walmart Project: This method is better than Single Smoothing because it can capture the general upward or downward movement in sales over time. However, it will completely miss the seasonal holiday spikes. Your forecast would be a straight line, unable to predict the repeating yearly patterns.
# MAGIC
# MAGIC Triple Exponential Smoothing (Holt-Winters Method)
# MAGIC This is the most advanced of the three and the most relevant for your project. It's designed for data that has both a trend and seasonality.
# MAGIC
# MAGIC What it learns: It adds a third component to the mix: seasonality. It learns and forecasts three things:
# MAGIC
# MAGIC Level: The baseline average.
# MAGIC
# MAGIC Trend: The slope of the data.
# MAGIC
# MAGIC Seasonality: The repeating pattern within a fixed period (e.g., every 52 weeks for your data).
# MAGIC
# MAGIC How it works: It maintains three smoothed values for these components. When it forecasts, it combines the level, the trend, and the value from the last seasonal cycle to produce a much more nuanced prediction.
# MAGIC
# MAGIC Additive vs. Multiplicative Seasonality: You have to tell the model how the seasonality combines with the trend.
# MAGIC
# MAGIC Additive: Assumes the seasonal fluctuations are roughly constant regardless of the sales level (e.g., sales always go up by $5,000 for Christmas).
# MAGIC
# MAGIC Multiplicative: Assumes the seasonal fluctuations are proportional to the sales level (e.g., sales always go up by 20% for Christmas). For retail sales, multiplicative seasonality is often a better fit.
# MAGIC
# MAGIC For your Walmart Project: This is an excellent model to use. It can capture the overall sales trend and the critical holiday spikes, making it a strong contender and a great baseline to compare against your LightGBM model.

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt, ExponentialSmoothing

# --- 1. Define WMAE Metric ---
# The custom evaluation metric for the competition
def wmae(y_true, y_pred, is_holiday):
    # Ensure is_holiday is a boolean array for np.where
    is_holiday_bool = np.array(is_holiday, dtype=bool)
    weights = np.where(is_holiday_bool, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

# --- 2. Load and Prepare Data ---
# This script assumes you have already run your preprocessing script and have
# the 'train_final.csv' and 'validation_final.csv' files.

PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
try:
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_final.csv'), parse_dates=['Date'])
    validation_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'validation_final.csv'), parse_dates=['Date'])
    print("Train and validation data loaded successfully.")
except FileNotFoundError:
    print("ERROR: Processed data not found. Please run the preprocessing script first.")
    # Create dummy data if files are not found for demonstration
    train_dates = pd.date_range(start='2010-02-05', end='2012-01-27', freq='W-FRI')
    validation_dates = pd.date_range(start='2012-02-03', end='2012-10-26', freq='W-FRI')
    train_df = pd.DataFrame({'Date': train_dates, 'Store': 4, 'Dept': 1, 'Weekly_Sales': 15000, 'IsHoliday': False})
    validation_df = pd.DataFrame({'Date': validation_dates, 'Store': 4, 'Dept': 1, 'Weekly_Sales': 16000, 'IsHoliday': False})


# --- 3. Isolate a Single Time Series for Demonstration ---
# We'll test on the same Store/Dept to see how the models perform.
store_id = 4
dept_id = 1

# Prepare training data
train_ts_df = train_df[(train_df['Store'] == store_id) & (train_df['Dept'] == dept_id)].set_index('Date')
train_ts = train_ts_df['Weekly_Sales'].asfreq('W-FRI').fillna(method='ffill')

# Prepare validation data
validation_ts_df = validation_df[(validation_df['Store'] == store_id) & (validation_df['Dept'] == dept_id)].set_index('Date')
validation_ts = validation_ts_df['Weekly_Sales'].asfreq('W-FRI').fillna(method='ffill')
validation_holidays = validation_ts_df['IsHoliday'].asfreq('W-FRI').fillna(False)

# Check if data is available
if train_ts.empty or validation_ts.empty:
    print(f"No data available for Store {store_id}, Dept {dept_id}. Please choose another.")
else:
    # --- 4. Train and Forecast Models ---

    # -- Model 1: Double Exponential Smoothing (Holt's Method) --
    print("\nTraining Double Exponential Smoothing model...")
    des_model = Holt(train_ts, initialization_method="estimated").fit()
    des_forecast = des_model.forecast(len(validation_ts))
    print("Forecasting complete.")

    # -- Model 2: Triple Exponential Smoothing (Holt-Winters) --
    print("\nTraining Triple Exponential Smoothing model...")
    # For sales data, 'multiplicative' is often a better fit for trend and seasonality.
    # seasonal_periods=52 tells the model the seasonal pattern repeats every 52 weeks.
    tes_model = ExponentialSmoothing(
        train_ts,
        trend='mul',
        seasonal='mul',
        seasonal_periods=52,
        initialization_method="estimated"
    ).fit()
    tes_forecast = tes_model.forecast(len(validation_ts))
    print("Forecasting complete.")


    # --- 5. Evaluate and Plot Results ---

    # Calculate WMAE scores
    wmae_des = wmae(validation_ts, des_forecast, validation_holidays)
    wmae_tes = wmae(validation_ts, tes_forecast, validation_holidays)

    print(f"\n--- Model Performance for Store {store_id}, Dept {dept_id} ---")
    print(f"Double Exponential Smoothing WMAE: {wmae_des:.2f}")
    print(f"Triple Exponential Smoothing WMAE: {wmae_tes:.2f}")

    # Plot the results for visual comparison
    plt.figure(figsize=(16, 8))
    plt.plot(train_ts.index, train_ts, label='Training Data', color='gray')
    plt.plot(validation_ts.index, validation_ts, label='Actual Sales (Validation)', color='blue', linewidth=2)
    plt.plot(des_forecast.index, des_forecast, label=f'Double ES Forecast (WMAE: {wmae_des:.2f})', color='orange', linestyle='--')
    plt.plot(tes_forecast.index, tes_forecast, label=f'Triple ES Forecast (WMAE: {wmae_tes:.2f})', color='green', linestyle='--')

    plt.title(f'Exponential Smoothing Forecast vs Actuals (Store {store_id}, Dept {dept_id})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Weekly Sales', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()


# COMMAND ----------

import pandas as pd
import numpy as np

# --- 1. Define the Evaluation Metrics ---

def wmae(y_true, y_pred, is_holiday):
    """Calculates the Weighted Mean Absolute Error."""
    is_holiday_bool = np.array(is_holiday, dtype=bool)
    weights = np.where(is_holiday_bool, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def mae(y_true, y_pred):
    """Calculates the Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape(y_true, y_pred):
    """Calculates the Mean Absolute Percentage Error."""
    # Add a small epsilon to avoid division by zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.inf # Or 0, depending on how you want to handle all-zero actuals
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


# --- 2. Example Usage ---

# Assume you have your validation data and predictions
# Let's create some dummy data for demonstration
validation_data = {
    'Weekly_Sales': [22000, 48000, 40000, 18000, 85000],
    'IsHoliday':    [False, True,  False, False, True]
}
predictions =     [23500, 45000, 41000, 18500, 78000]

validation_df = pd.DataFrame(validation_data)
y_true = validation_df['Weekly_Sales']
is_holiday = validation_df['IsHoliday']
y_pred = pd.Series(predictions)


# --- 3. Calculate and Print All Metrics ---

wmae_score = wmae(y_true, y_pred, is_holiday)
mae_score = mae(y_true, y_pred)
rmse_score = rmse(y_true, y_pred)
mape_score = mape(y_true, y_pred)

print("--- Model Evaluation Results ---")
print(f"WMAE (Competition Metric): ${wmae_score:,.2f}")
print(f"MAE (Average $ Error):     ${mae_score:,.2f}")
print(f"RMSE (Penalizes Large Errors): ${rmse_score:,.2f}")
print(f"MAPE (Average % Error):      {mape_score:.2f}%")

# --- 4. Interpretation Example ---
if rmse_score > mae_score * 1.2: # Check if RMSE is significantly larger than MAE
    print("\nInterpretation: The RMSE is notably higher than the MAE.")
    print("This suggests the model is making a few large errors that are heavily penalized by RMSE.")
else:
    print("\nInterpretation: The RMSE and MAE are relatively close.")
    print("This suggests the model's errors are fairly consistent in magnitude.")


