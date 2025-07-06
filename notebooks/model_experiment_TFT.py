# Databricks notebook source
# MAGIC %pip install pytorch-forecasting==0.10.1 pytorch-lightning==1.7.7 wandb
# MAGIC

# COMMAND ----------


import pandas as pd
import numpy as np
import os
import warnings
import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Import the TFT model and the necessary data tools
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")



# COMMAND ----------

print("\n--- SECTION 2: WANDB & DATA SETUP ---")

wandb.login(key = "720b5644412076fa3e35eb1ffccab9895b8369db")

PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
TRAIN_PATH = os.path.join(PROCESSED_DIR, 'train_processed_final.csv')
try:
    df = pd.read_csv(TRAIN_PATH, parse_dates=['Date'])
    print("Successfully loaded processed training data.")
except FileNotFoundError:
    dbutils.notebook.exit("Data preparation notebook must be run first.")

# COMMAND ----------

print("\n--- SECTION 3: ROBUST DATA PREPARATION FOR TFT ---")

# --- 3.1: Add required time and series identifiers ---
df["time_idx"] = (df["Date"].dt.year - df["Date"].dt.year.min()) * 52 + df["Date"].dt.isocalendar().week
df["time_idx"] = df["time_idx"] - df["time_idx"].min()
df['series'] = df['Store'].astype(str) + '_' + df['Dept'].astype(str)
df['Store_Dept'] = df['series']

# --- 3.2: Convert all categorical features to string type ---
static_cats = ["Store", "Dept", "Store_Dept"]
time_varying_known_cats = [col for col in ["IsBlackFridayWeek", "IsSuperBowlWeek", "IsLaborDayWeek", "IsChristmasWeek"] if col in df.columns]
all_categorical_cols = static_cats + time_varying_known_cats

for col in all_categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
print("Categorical columns converted to string type.")

# --- 3.3: Define forecast parameters ---
encoder_length = 36
prediction_length = 8
training_cutoff = df["time_idx"].max() - prediction_length

# --- 3.4: Define all parameters for the dataset constructor once ---
dataset_params = dict(
    time_idx="time_idx",
    target="Weekly_Sales",
    group_ids=["series"],
    max_encoder_length=encoder_length,
    max_prediction_length=prediction_length,
    static_categoricals=static_cats,
    static_reals=["Size"],
    time_varying_known_categoricals=time_varying_known_cats,
    time_varying_known_reals=["CPI", "Unemployment", "Year", "Month", "WeekOfYear"],
    time_varying_unknown_reals=["Weekly_Sales", "Temperature", "Fuel_Price"],
    target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus"),
    allow_missing_timesteps=True
)

# --- 3.5: Manually create the training dataset ---
# This dataset learns the encoders and scalers from the training data partition.
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    **dataset_params
)

# --- 3.6: Manually create the validation dataset ---
# We create this from the full dataframe, but we pass the fitted
# encoders and scalers from the `training` object to ensure consistency.
validation = TimeSeriesDataSet(
    df[lambda x: x.time_idx > training_cutoff - encoder_length], # Ensure enough history for first validation sample
    **dataset_params,
    # Use the same encoders and scalers that were learned from the training data
    categorical_encoders=training.categorical_encoders,
    scalers=training.scalers,
)

# --- 3.7: Create dataloaders ---
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=2)

print("Data preparation for deep learning is complete and robust.")

# COMMAND ----------


wandb_logger = WandbLogger(project="Walmart-Sales-Forecasting-DL", name="TFT-Run-Final", log_model="all")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")

trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    limit_train_batches=100,
    limit_val_batches=30,
    callbacks=[early_stop_callback],
    logger=wandb_logger,
)

# Create the TFT model using the reliable .from_dataset() on the `training` object.
# This works because the training object is now perfectly formed in a stable environment.
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4,
)

print(f"Number of parameters in model: {tft.size()/1e3:.1f}k")

# Start training
print("Starting TFT model training... This will now work.")
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

wandb.finish()
print("\nTFT training complete.")

