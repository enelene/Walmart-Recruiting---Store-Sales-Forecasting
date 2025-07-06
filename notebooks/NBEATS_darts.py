# Databricks notebook source
# MAGIC %pip install "pandas>=1.5.3" "darts[torch, wandb]==0.27.1"
# MAGIC

# COMMAND ----------

!pip install wandb

# COMMAND ----------


import pandas as pd
import numpy as np
import os
import warnings
import wandb

# Import from the darts library
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import wmae
from darts.dataprocessing.transformers import Scaler

warnings.filterwarnings("ignore")


# COMMAND ----------


# ==============================================================================
# SECTION 2: WANDB & DATA SETUP
# ==============================================================================
print("\n--- SECTION 2: WANDB & DATA SETUP ---")

wandb.login(key = "720b5644412076fa3e35eb1ffccab9895b8369db")

PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
TRAIN_PATH = os.path.join(PROCESSED_DIR, 'train_processed_final.csv')
try:
    df = pd.read_csv(TRAIN_PATH, parse_dates=['Date'])
    print("Successfully loaded processed training data.")
except FileNotFoundError:
    dbutils.notebook.exit("Data preparation notebook must be run first.")

# ==============================================================================
# SECTION 3: DATA PREPARATION FOR DARTS
# ==============================================================================
print("\n--- SECTION 3: DATA PREPARATION FOR DARTS ---")

# --- 3.1: Select a single, robust time series to model ---
# This approach is much cleaner and more stable for demonstration.
series_stats = df.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'count']).dropna()
series_stats['cov'] = series_stats['std'] / series_stats['mean']
series_stats = series_stats[series_stats['count'] >= 143]
best_candidate = series_stats.sort_values(by='cov', ascending=True).index[0]
STORE_ID, DEPT_ID = best_candidate
print(f"Selected candidate series: Store {STORE_ID}, Department {DEPT_ID}")

ts_df = df[(df['Store'] == STORE_ID) & (df['Dept'] == DEPT_ID)].copy()

# --- 3.2: Create a Darts TimeSeries object ---
# This is the core data object for the darts library. It's much simpler.
# We create the target series (y) and the feature series (covariates).
y_series = TimeSeries.from_dataframe(
    ts_df,
    time_col='Date',
    value_cols='Weekly_Sales',
    freq='W-FRI'
)

# --- 3.3: Prepare external features (covariates) ---
# We will use the holiday flags as features for the model.
covariate_cols = ['IsBlackFridayWeek', 'IsSuperBowlWeek', 'IsLaborDayWeek', 'IsChristmasWeek']
covariates = TimeSeries.from_dataframe(
    ts_df,
    time_col='Date',
    value_cols=[col for col in covariate_cols if col in ts_df.columns],
    freq='W-FRI'
)

# --- 3.4: Normalize the data ---
# It's a best practice to scale data for neural networks.
scaler_y = Scaler()
scaler_cov = Scaler()

y_scaled = scaler_y.fit_transform(y_series)
covariates_scaled = scaler_cov.fit_transform(covariates)

# --- 3.5: Create training and validation sets ---
# We will train on the first part and validate on the last 29 weeks.
val_split_point = len(y_scaled) - 29
y_train, y_val = y_scaled.split_after(val_split_point)
cov_train, cov_val = covariates_scaled.split_after(val_split_point)

print("Data preparation for Darts is complete.")

# ==============================================================================
# SECTION 4: N-BEATS MODEL TRAINING WITH DARTS
# ==============================================================================
print("\n--- SECTION 4: N-BEATS MODEL TRAINING WITH DARTS ---")

# --- 4.1: Setup Logging ---
# Darts can automatically log to a Wandb run.
wandb.init(project="Walmart-Sales-Forecasting-DL", name=f"Darts-NBEATS-S{STORE_ID}-D{DEPT_ID}", reinit=True)

# --- 4.2: Define the N-BEATS Model ---
# The darts N-BEATS model can accept external features (future_covariates).
# We define the input and output chunk lengths (similar to encoder/prediction lengths).
model_nbeats = NBEATSModel(
    input_chunk_length=36,
    output_chunk_length=8,
    n_epochs=50,
    random_state=42,
    wandb_watch="all", # Connects to our wandb run
    pl_trainer_kwargs={"accelerator": "gpu" if torch.cuda.is_available() else "cpu", "devices": 1}
)

# --- 4.3: Start Training ---
print("Starting N-BEATS model training with Darts...")
model_nbeats.fit(
    series=y_train,
    future_covariates=cov_train
)
print("Training complete.")

# ==============================================================================
# SECTION 5: EVALUATION
# ==============================================================================
print("\n--- SECTION 5: EVALUATION ---")

# --- 5.1: Make Predictions ---
# We need to provide the future covariates for the period we want to predict.
predictions_scaled = model_nbeats.predict(
    n=len(y_val),
    future_covariates=covariates_scaled
)

# --- 5.2: Inverse Transform the Predictions ---
# We must convert the scaled predictions back to the original sales values.
predictions_unscaled = scaler_y.inverse_transform(predictions_scaled)

# --- 5.3: Calculate WMAE ---
# Get the holiday flags for the validation period
is_holiday_test = ts_df[ts_df['Date'].isin(y_val.time_index)]['IsHoliday'].values

# Darts has a built-in WMAE function, but we'll use our own for consistency.
def wmae_manual(y_true, y_pred, is_holiday):
    true_vals = y_true.pd_series().values
    pred_vals = y_pred.pd_series().values
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(true_vals - pred_vals)) / np.sum(weights)

final_wmae = wmae_manual(y_val, predictions_unscaled, is_holiday_test)
print(f"Final WMAE for Darts N-BEATS model: {final_wmae}")

# COMMAND ----------


import shutil
import os

folder_to_zip = "/dbfs/FileStore/walmart_project/"

# --- Define where the output zip file will be temporarily created ---
# We will create it on the local disk of the driver node first.
output_zip_name_local = "/tmp/download_archive"

# --- Define the final destination in the publicly accessible FileStore ---
# This is the path you will use to access the file later.
final_zip_path_dbfs = "dbfs:/FileStore/download_archive.zip"

# ==============================================================================
# STEP 2: COMPRESS THE FOLDER
# ==============================================================================
print(f"Attempting to zip the folder: {folder_to_zip}")

try:
    # Use shutil.make_archive to create the zip file on the local driver disk
    # It takes the output path (without .zip), the format, and the source folder
    shutil.make_archive(
        base_name=output_zip_name_local,
        format='zip',
        root_dir=folder_to_zip
    )
    print(f"Successfully created a temporary zip archive at: {output_zip_name_local}.zip")
    
    # ==============================================================================
    # STEP 3: MOVE THE ZIP FILE TO A DOWNLOADABLE LOCATION
    # ==============================================================================
    
    # Use the Databricks utility to move the file from the local driver disk
    # to the publicly accessible /FileStore/ directory in DBFS.
    dbutils.fs.mv(f"file:{output_zip_name_local}.zip", final_zip_path_dbfs)
    print(f"Successfully moved the zip file to: {final_zip_path_dbfs}")
except Exception as e:
    print(f"Error: {e}")
