# Databricks notebook source
!pip install -r requirements.txt

# COMMAND ----------

# MAGIC %pip install pytorch-forecasting wandb
# MAGIC
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC import os
# MAGIC import warnings
# MAGIC import wandb
# MAGIC
# MAGIC import torch
# MAGIC import pytorch_lightning as pl
# MAGIC from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
# MAGIC from pytorch_lightning.loggers import WandbLogger
# MAGIC
# MAGIC from pytorch_forecasting import TimeSeriesDataSet, NBeats
# MAGIC from pytorch_forecasting.data import GroupNormalizer
# MAGIC from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss
# MAGIC

# COMMAND ----------

warnings.filterwarnings("ignore")
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

# MAGIC %md
# MAGIC SECTION 3: DATA PREPARATION FOR PYTORCH FORECASTING

# COMMAND ----------


# --- 3.1: Add required time and series identifiers ---
df["time_idx"] = (df["Date"].dt.year - df["Date"].dt.year.min()) * 52 + df["Date"].dt.isocalendar().week
df["time_idx"] = df["time_idx"] - df["time_idx"].min()
df['series'] = df['Store'].astype(str) + '_' + df['Dept'].astype(str)

# --- 3.2: Create a clean, univariate DataFrame ---
# This contains ONLY the three columns required by a univariate N-BEATS model.
univariate_df = df[['time_idx', 'Weekly_Sales', 'series']].copy()
print("Created a clean, univariate dataframe for the model.")

# --- 3.3: Define forecast parameters ---
encoder_length = 24
prediction_length = 8
training_cutoff = univariate_df["time_idx"].max() - prediction_length

# --- 3.4: Manually create the training dataset ---
# This dataset learns the encoders and scalers from the training data partition.
# --- 3.4: Manually create the training dataset ---
training = TimeSeriesDataSet(
    univariate_df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Weekly_Sales",
    group_ids=["series"],
    max_encoder_length=encoder_length,
    max_prediction_length=prediction_length,
    time_varying_unknown_reals=["Weekly_Sales"],  # 🔥 REQUIRED FIX
    target_normalizer=GroupNormalizer(groups=["series"], transformation="softplus"),
    allow_missing_timesteps=True
)

# --- 3.5: Validation dataset ---
validation = TimeSeriesDataSet(
    univariate_df,
    time_idx="time_idx",
    target="Weekly_Sales",
    group_ids=["series"],
    max_encoder_length=encoder_length,
    max_prediction_length=prediction_length,
    time_varying_unknown_reals=["Weekly_Sales"],  # 🔥 REQUIRED FIX
    categorical_encoders=training.categorical_encoders,
    scalers=training.scalers,
    allow_missing_timesteps=True
)


# --- 3.6: Create dataloaders ---
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=2)


# COMMAND ----------

pip install --upgrade pytorch-forecasting


# COMMAND ----------

from pytorch_forecasting.models import NBeats
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data import GroupNormalizer, TimeSeriesDataSet
import pytorch_lightning as pl


# COMMAND ----------

!pip uninstall pytorch-forecasting pytorch-lightning -y
!pip install pytorch-lightning==2.2.2 pytorch-forecasting==1.4.0


# COMMAND ----------

import pytorch_forecasting
import pytorch_lightning

print("forecasting:", pytorch_forecasting.__version__)
print("lightning:", pytorch_lightning.__version__)


# COMMAND ----------

# Databricks cell
%pip uninstall -y pytorch-forecasting pytorch-lightning
%pip install pytorch-lightning==2.2.2 pytorch-forecasting==1.4.0 --force-reinstall


# COMMAND ----------

from pytorch_forecasting.models.nbeats import NBeats
from pytorch_forecasting.metrics import MAE

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

nbeats = NBeats.from_dataset(
    training,
    learning_rate=3e-2,
    weight_decay=1e-2,
    widths=[32, 512],
    backcast_loss_ratio=0.1,
    loss=MAE()
)

trainer.fit(
    nbeats,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)


# COMMAND ----------


from pytorch_forecasting import NBeats
# from pytorch_forecasting.models.base_model import ModelWrapper
from pytorch_forecasting.models.nbeats import NBeatsModel


wandb_logger = WandbLogger(project="Walmart-Sales-Forecasting-DL", name="NBEATS-Univariate-Final", log_model="all")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")




# Create the core N-BEATS model
base_model = NBeats.from_dataset(
    training,
    learning_rate=3e-2,
    weight_decay=1e-2,
    widths=[32, 512],
    backcast_loss_ratio=0.1,
    loss=MAE()
)

# Manually wrap it into a LightningModule
nbeats = ModelWrapper(base_model)


trainer.fit(
    nbeats,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)


wandb.finish()
print("\nN-BEATS training complete.")


# COMMAND ----------


early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")

trainer = pl.Trainer(
    max_epochs=50, # Train for more epochs on a single series
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    logger=wandb_logger,
)

# Use the .from_dataset() helper function, which will now work.
# This function correctly wraps the model in a LightningModule.
nbeats = NBeats.from_dataset(
    training,
    learning_rate=3e-2,
    weight_decay=1e-2,
    widths=[32, 512],
    backcast_loss_ratio=0.1,
    loss=MAE()
)

nbeats = nbeats.to_lightning_module()  # 🔥 This wraps it correctl

print("Starting N-BEATS model training... This will now work.")
trainer.fit(
    nbeats,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

wandb.finish()
print("\nN-BEATS training complete.")

