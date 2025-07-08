# Databricks notebook source
pip list

# COMMAND ----------

# MAGIC %pip install "pandas>=1.5.3" "darts[torch, wandb]==0.27.1"
# MAGIC

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %pip install wandb

# COMMAND ----------

print("\n--- SECTION 2: IMPORTS & SETUP ---")

import pandas as pd
import numpy as np
import os
import warnings
import wandb
import torch
from tqdm.notebook import tqdm

# Import from the darts library
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore")


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


# --- 4.1: Filter for series that are long enough for training and validation ---
# A series needs to be at least input_chunk_length + output_chunk_length long.
INPUT_CHUNK_LENGTH = 36
OUTPUT_CHUNK_LENGTH = 8
MIN_SERIES_LENGTH = INPUT_CHUNK_LENGTH + OUTPUT_CHUNK_LENGTH

series_lengths = df.groupby(['Store', 'Dept']).size()
long_enough_series = series_lengths[series_lengths >= MIN_SERIES_LENGTH].index

df_filtered = df.set_index(['Store', 'Dept']).loc[long_enough_series].reset_index()
print(f"Filtered data to {len(long_enough_series)} series that are long enough for the model.")

# --- 4.2: Convert DataFrame to a List of TimeSeries objects ---
print("Converting DataFrame to list of Darts TimeSeries objects...")
# This is the standard way to prepare data for global models in Darts.
all_series = TimeSeries.from_group_dataframe(
    df_filtered,
    group_cols=['Store', 'Dept'],
    time_col='Date',
    value_cols='Weekly_Sales',
    freq='W-FRI',
    fill_missing_dates=True,
    fillna_value=0
)

# --- 4.3: Create Covariates (Features) ---
# Create calendar features that are known in the future.
future_covariates_list = []
for series in tqdm(all_series, desc="Generating future covariates"):
    covariates = datetime_attribute_timeseries(
        series, attribute="month", one_hot=True
    )
    covariates = covariates.stack(
        datetime_attribute_timeseries(series, attribute="week", one_hot=True)
    )
    future_covariates_list.append(covariates)

# --- 4.4: Normalize the data ---
print("Scaling the data...")
scaler = Scaler()
all_series_scaled = scaler.fit_transform(all_series)
future_covariates_scaled = Scaler().fit_transform(future_covariates_list)

# --- 4.5: Create Training and Validation Sets ---
# Darts handles the train/validation split internally when you pass a validation_series to .fit()
print("Data preparation for Darts is complete.")

# COMMAND ----------


# --- 5.1: Setup Logging ---
wandb_logger = WandbLogger(project="Walmart-Sales-Forecasting-DL", name="Darts-TFT-Global-Model")

# --- 5.2: Define the TFT Model ---
# This model is designed for this exact type of global, multivariate data.
model_tft = TFTModel(
    input_chunk_length=INPUT_CHUNK_LENGTH,
    output_chunk_length=OUTPUT_CHUNK_LENGTH,
    n_epochs=15, # TFTs train slower, so we use fewer epochs for the first run
    random_state=42,
    pl_trainer_kwargs={
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu", 
        "devices": 1,
        "logger": wandb_logger
    }
)

# --- 5.3: Start Training ---
# We pass the list of all series to the model.
# We also provide a validation set for early stopping and metric tracking.
print("Starting GLOBAL TFT model training with Darts...")
model_tft.fit(
    series=all_series_scaled,
    future_covariates=future_covariates_scaled,
    val_series=all_series_scaled, # Use the same set for validation for simplicity
    val_future_covariates=future_covariates_scaled
)
print("Training complete.")

wandb.finish()

