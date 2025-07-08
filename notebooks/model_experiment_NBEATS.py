# Databricks notebook source
pip list

# COMMAND ----------

# MAGIC %pip install "pandas>=1.5.3" "darts[torch, wandb]==0.27.1"
# MAGIC

# COMMAND ----------

# MAGIC %pip install wandb

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import os
import warnings
import wandb
import torch
from tqdm.notebook import tqdm

# Import from the darts library
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.loggers import WandbLogger

# COMMAND ----------

wandb.login(key = "720b5644412076fa3e35eb1ffccab9895b8369db")
PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
TRAIN_PATH = os.path.join(PROCESSED_DIR, 'train_processed_final.csv')
try:
    df = pd.read_csv(TRAIN_PATH, parse_dates=['Date'])
    print("Successfully loaded processed training data.")
except FileNotFoundError:
    dbutils.notebook.exit("Data preparation notebook must be run first.")

# COMMAND ----------

# MAGIC %md
# MAGIC SECTION 3: DATA PREPARATION FOR PYTORCH FORECASTING

# COMMAND ----------

print("\n--- SECTION 4: DATA PREPARATION FOR DARTS ---")

# --- 4.1: Select a single, robust time series to model ---
series_stats = df.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'count']).dropna()
series_stats['cov'] = series_stats['std'] / series_stats['mean']
series_stats = series_stats[series_stats['count'] >= 143]
best_candidate = series_stats.sort_values(by='cov', ascending=True).index[0]
STORE_ID, DEPT_ID = best_candidate
print(f"Selected candidate series: Store {STORE_ID}, Department {DEPT_ID}")

ts_df = df[(df['Store'] == STORE_ID) & (df['Dept'] == DEPT_ID)].copy()

# --- 4.2: Create Darts TimeSeries objects ---
y_series = TimeSeries.from_dataframe(ts_df, time_col='Date', value_cols='Weekly_Sales', freq='W-FRI', fill_missing_dates=True, fillna_value=0)
covariate_cols = ['IsBlackFridayWeek', 'IsSuperBowlWeek', 'IsLaborDayWeek', 'IsChristmasWeek', 'Month', 'WeekOfYear']
covariates = TimeSeries.from_dataframe(ts_df, time_col='Date', value_cols=[col for col in covariate_cols if col in ts_df.columns], freq='W-FRI', fill_missing_dates=True, fillna_value=0)

# --- 4.3: Normalize the data ---
scaler_y, scaler_cov = Scaler(), Scaler()
y_scaled = scaler_y.fit_transform(y_series)
covariates_scaled = scaler_cov.fit_transform(covariates)

# --- 4.4: Create training and validation sets ---
val_split_point = -29
y_train, y_val = y_scaled[:val_split_point], y_scaled[val_split_point:]
cov_train, cov_val = covariates_scaled[:val_split_point], covariates_scaled[val_split_point:]


# COMMAND ----------

from pytorch_lightning.loggers import WandbLogger

# COMMAND ----------

print("\n--- SECTION 5: N-BEATS MODEL TRAINING WITH DARTS ---")

# --- 5.1: Setup Logging ---
wandb_logger = WandbLogger(project="Walmart-Sales-Forecasting-DL", name=f"Darts-NBEATS-S{STORE_ID}-D{DEPT_ID}")

# --- 5.2: Define the N-BEATS Model ---
model_nbeats = NBEATSModel(
    input_chunk_length=36,
    output_chunk_length=8,
    n_epochs=50,
    random_state=42,
    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu", 
        "devices": 1,
        "logger": wandb_logger
    }
)

# --- 5.3: Start Training ---
print("Starting N-BEATS model training with Darts...")
# THE FIX IS HERE: We use 'past_covariates' instead of 'future_covariates'.
model_nbeats.fit(
    series=y_train,
    past_covariates=cov_train
)
print("Training complete.")

print("\n--- SECTION 6: EVALUATION ---")

# --- 6.1: Make Predictions ---
# We provide the historical series and the past covariates to make a forecast.
predictions_scaled = model_nbeats.predict(
    n=len(y_val),
    series=y_train,
    past_covariates=covariates_scaled
)

# --- 6.2: Inverse Transform the Predictions ---
predictions_unscaled = scaler_y.inverse_transform(predictions_scaled)

# --- 6.3: Calculate WMAE ---
is_holiday_test = ts_df[ts_df['Date'].isin(y_val.time_index)]['IsHoliday'].values

def wmae_manual(y_true, y_pred, is_holiday):
    true_vals = y_true.pd_series().values
    pred_vals = y_pred.pd_series().values
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(true_vals - pred_vals)) / np.sum(weights)

final_wmae = wmae_manual(y_val, predictions_unscaled, is_holiday_test)
print(f"Final WMAE for Darts N-BEATS model: {final_wmae:.2f}")

# Log the final score to the same wandb run
wandb.log({"final_wmae": final_wmae})

# COMMAND ----------

import matplotlib.pyplot as plt 
# --- 6.4: Plot the results ---
plt.figure(figsize=(12, 6))
y_series.plot(label='Actual')
predictions_unscaled.plot(label='Forecast')
plt.title(f'N-BEATS Forecast vs Actuals | WMAE: {final_wmae:.2f}')
plt.legend()
plt.show()

# COMMAND ----------

wandb.finish()

# COMMAND ----------


# --- 4.1: Define Model Parameters ---
INPUT_CHUNK_LENGTH = 36
OUTPUT_CHUNK_LENGTH = 8
MIN_SERIES_LENGTH = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH

# --- 4.2: Filter for series that are long enough ---
series_lengths = df.groupby(['Store', 'Dept']).size()
long_enough_series = series_lengths[series_lengths >= MIN_SERIES_LENGTH].index
df_filtered = df.set_index(['Store', 'Dept']).loc[long_enough_series].reset_index()
print(f"Filtered data to {len(long_enough_series)} series that are long enough for the model.")

# --- 4.3: Convert DataFrame to a List of TimeSeries objects ---
print("Converting DataFrame to list of Darts TimeSeries objects...")
# Target Series (y)
all_y_series = TimeSeries.from_group_dataframe(
    df_filtered, group_cols=['Store', 'Dept'], time_col='Date',
    value_cols='Weekly_Sales', freq='W-FRI', fill_missing_dates=True, fillna_value=0
)
# Past Covariates (features we know from the past)
past_cov_cols = ['IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'IsBlackFridayWeek', 'IsSuperBowlWeek', 'IsLaborDayWeek', 'IsChristmasWeek', 'Year', 'Month', 'WeekOfYear', 'Day', 'HasMarkdown', 'Type_A', 'Type_B', 'Type_C']
all_past_covs = TimeSeries.from_group_dataframe(
    df_filtered, group_cols=['Store', 'Dept'], time_col='Date',
    value_cols=[col for col in past_cov_cols if col in df.columns], freq='W-FRI', fill_missing_dates=True, fillna_value=0
)

# --- 4.4: Normalize the data ---
print("Scaling the data...")
scaler_y, scaler_past = Scaler(), Scaler()
all_y_scaled = scaler_y.fit_transform(all_y_series)
all_past_scaled = scaler_past.fit_transform(all_past_covs)

print("Data preparation for Darts is complete.")

# COMMAND ----------


#  Manually create Training and Validation Sets
print("Manually splitting data into training and validation sets...")
val_split_point = -29
train_y, val_y = [], []
train_past, val_past = [], []

for i in range(len(all_y_scaled)):
    series = all_y_scaled[i]
    covariates = all_past_scaled[i]
    
    # Use the robust .split_before() method which handles lengths correctly
    try:
        # The split point must be a timestamp or an integer index
        split_time = series.time_index[val_split_point]
        
        train_series, val_series = series.split_before(split_time)
        train_cov, val_cov = covariates.split_before(split_time)
        
        train_y.append(train_series)
        val_y.append(val_series)
        train_past.append(train_cov)
        val_past.append(val_cov)
    except (ValueError, IndexError):
        # This will skip any series that is still too short after filtering, making it robust
        pass

# COMMAND ----------


# --- THE FIX IS HERE ---
# 4.5: Manually create Training and Validation Sets
print("Manually splitting data into training and validation sets...")
train_y, val_y = [], []
train_past, val_past = [], []
# We will use the last 29 weeks as our validation set
val_duration = 29

for series, covariates in zip(all_y_scaled, all_past_scaled):
    # Use the robust .split_before() method with a float to get a percentage split
    # This ensures the validation set has enough history for the model.
    try:
        train_series, val_series = series.split_before(0.8) # 80% for training
        train_cov, val_cov = covariates.split_before(0.8)
        
        train_y.append(train_series)
        val_y.append(val_series)
        train_past.append(train_cov)
        val_past.append(val_cov)
    except (ValueError, IndexError):
        # This will skip any series that is still too short, making it robust
        pass

print(f"Created {len(train_y)} training and validation series.")

# ==============================================================================
# SECTION 5: N-BEATS GLOBAL MODEL TRAINING
# ==============================================================================
print("\n--- SECTION 5: N-BEATS GLOBAL MODEL TRAINING ---")

# --- 5.1: Setup Logging ---
wandb_logger = WandbLogger(project="Walmart-Sales-Forecasting-DL", name="Darts-NBEATS-Global-Final")

# --- 5.2: Define the N-BEATS Model ---
model_nbeats = NBEATSModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,
    n_epochs=20,
    random_state=42,
    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu", 
        "devices": 1,
        "logger": wandb_logger
    }
)

# --- 5.3: Start Training ---
print("Starting GLOBAL N-BEATS model training with Darts...")
# We pass the manually created lists of series to the model.
model_nbeats.fit(
    series=train_y,
    past_covariates=train_past,
    val_series=val_y,
    val_past_covariates=val_past,
    verbose=True
)
print("Training complete.")

# ==============================================================================
# SECTION 6: EVALUATION
# ==============================================================================
print("\n--- SECTION 6: EVALUATION ---")

# --- 6.1: Make Predictions on the Validation Set ---
print("Generating predictions on the validation set...")
predictions_scaled = model_nbeats.predict(
    n=val_duration,
    series=train_y,
    past_covariates=all_past_scaled
)

# --- 6.2: Inverse Transform the Predictions ---
predictions_unscaled = scaler_y.inverse_transform(predictions_scaled)

# --- 6.3: Calculate Average WMAE ---
total_wmae = 0
num_series_evaluated = 0

print("Calculating WMAE across all validation series...")
for i in tqdm(range(len(val_y))):
    original_val_series = all_y_series[i][-val_duration:]
    
    static_covs = original_val_series.static_covariates
    store_id = int(static_covs['Store'].iloc[0])
    dept_id = int(static_covs['Dept'].iloc[0])
    
    is_holiday_test = df[
        (df['Store'] == store_id) &
        (df['Dept'] == dept_id) &
        (df['Date'].isin(original_val_series.time_index))
    ]['IsHoliday'].values
    
    def wmae_manual(y_true, y_pred, is_holiday):
        true_vals = y_true.pd_series().values
        pred_vals = y_pred.pd_series().values
        weights = np.where(is_holiday[:len(true_vals)], 5, 1)
        return np.sum(weights * np.abs(true_vals - pred_vals)) / np.sum(weights)

    if len(is_holiday_test) == len(original_val_series):
        series_wmae = wmae_manual(original_val_series, predictions_unscaled[i], is_holiday_test)
        total_wmae += series_wmae
        num_series_evaluated += 1

average_wmae = total_wmae / num_series_evaluated if num_series_evaluated > 0 else 0
print(f"\n-------------------------------------------------")
print(f"Final Average WMAE for Darts N-BEATS model: {average_wmae:.2f}")
print(f"-------------------------------------------------")
wandb.log({"final_avg_wmae": average_wmae})

wandb.finish()


# COMMAND ----------


# --- 5.1: Setup Logging ---
wandb_logger = WandbLogger(project="Walmart-Sales-Forecasting-DL", name="Darts-NBEATS-Global-Final")

# --- 5.2: Define the N-BEATS Model ---
model_nbeats = NBEATSModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,
    n_epochs=20,
    random_state=42,
    batch_size=512,
    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu", 
        "devices": 1,
        "logger": wandb_logger
    }
)

# --- 5.3: Start Training ---
print("Starting GLOBAL N-BEATS model training with Darts...")
# THE FIX IS HERE: We pass the full, unsplit lists to the model and let the
# library handle the validation split by providing a fraction.
model_nbeats.fit(
    series=all_y_scaled,
    past_covariates=all_past_scaled,
    val_series=0.2, # Use the last 20% of each series for validation
    val_past_covariates=0.2,
    verbose=True
)
print("Training complete.")

wandb.finish()


# COMMAND ----------

print("\n--- SECTION 6: EVALUATION ---")

# --- 6.1: Make Predictions on the Validation Set ---
print("Generating predictions on the validation set...")
predictions_scaled = model_nbeats.predict(
    n=29,
    series=train_y,
    past_covariates=all_past_scaled
)

# --- 6.2: Inverse Transform the Predictions ---
predictions_unscaled = scaler_y.inverse_transform(predictions_scaled)

# --- 6.3: Calculate Average WMAE ---
total_wmae = 0
num_series_evaluated = 0

print("Calculating WMAE across all validation series...")
for i in tqdm(range(len(val_y))):
    original_val_series = all_y_series[i][-29:]
    
    store_id = int(original_val_series.static_covariates['Store'].iloc[0])
    dept_id = int(original_val_series.static_covariates['Dept'].iloc[0])
    
    is_holiday_test = df[
        (df['Store'] == store_id) &
        (df['Dept'] == dept_id) &
        (df['Date'].isin(original_val_series.time_index))
    ]['IsHoliday'].values
    
    def wmae_manual(y_true, y_pred, is_holiday):
        true_vals = y_true.pd_series().values
        pred_vals = y_pred.pd_series().values
        weights = np.where(is_holiday[:len(true_vals)], 5, 1)
        return np.sum(weights * np.abs(true_vals - pred_vals)) / np.sum(weights)

    if len(is_holiday_test) == len(original_val_series):
        series_wmae = wmae_manual(original_val_series, predictions_unscaled[i], is_holiday_test)
        total_wmae += series_wmae
        num_series_evaluated += 1

average_wmae = total_wmae / num_series_evaluated if num_series_evaluated > 0 else 0
print(f"\n-------------------------------------------------")
print(f"Final Average WMAE for Darts N-BEATS model: {average_wmae:.2f}")
print(f"-------------------------------------------------")
wandb.log({"final_avg_wmae": average_wmae})

# --- 6.4: Plot the results for one example series ---
plt.figure(figsize=(12, 6))
all_y_series[0].plot(label='Actual')
predictions_unscaled[0].plot(label='Forecast')
plt.title(f'N-BEATS Forecast vs Actuals (Example Series)')
plt.legend()
plt.show()

wandb.finish()

