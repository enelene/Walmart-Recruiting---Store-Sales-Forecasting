# Databricks notebook source
!pip install -r requirements.txt
import pandas as pd
import numpy as np
import mlflow
import os
import zipfile
from src.preprocessing import advanced_feature_engineering

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

train_df = pd.read_csv('/dbfs/FileStore/walmart_project/data/processed/train_processed_final.csv', parse_dates=['Date'])

# Let's pick a busy one: Store 4, Department 1
ts_data = train_df[(train_df['Store'] == 4) & (train_df['Dept'] == 1)].copy()
ts_data = ts_data.set_index('Date').sort_index()

ts_data = ts_data['Weekly_Sales'].asfreq('W-FRI')
ts_data.fillna(method='ffill', inplace=True) # Fill any missing weeks

decomposition = seasonal_decompose(ts_data, model='additive', period=52)

fig = decomposition.plot()
fig.set_size_inches(14, 8)
plt.show()

# COMMAND ----------

mlflow.set_tracking_uri("https://dagshub.com/enelene/Walmart-Recruiting---Store-Sales-Forecasting.mlflow")

os.environ['MLFLOW_TRACKING_USERNAME'] = 'enelene'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cbe8109dbe80931664d754dbd476356414fa62a0'

REGISTERED_MODEL_NAME = "LightGBM-Walmart-Sales-Pipeline"
MODEL_STAGE = "None" 

model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
print(f"Loading model from: {model_uri}")

loaded_pipeline = mlflow.pyfunc.load_model(model_uri)
print("Model pipeline loaded successfully.")



# COMMAND ----------

RAW_DATA_DIR = '/dbfs/FileStore/walmart_project/data/raw'
original_test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'test.csv'))
print(f"Original test file loaded. It has {len(original_test_df)} rows.")

features_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'features.csv'))
stores_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'stores.csv'))

raw_test_data_to_process = original_test_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
raw_test_data_to_process = raw_test_data_to_process.merge(stores_df, on='Store', how='left')

raw_test_data_to_process['Weekly_Sales'] = 0
if len(raw_test_data_to_process) != len(original_test_df):
    print("WARNING: Row count changed after merging. Check the merge logic.")
else:
    print("Row count is correct after merging.")

print("Generating predictions...")
predictions = loaded_pipeline.predict(raw_test_data_to_process)
print("Predictions generated successfully.")

if len(predictions) != len(original_test_df):
     print(f"FATAL ERROR: Prediction count ({len(predictions)}) does not match original row count ({len(original_test_df)}).")
     dbutils.notebook.exit("Prediction length mismatch.")
else:
    print("Prediction count matches original row count. Ready to create submission file.")


# COMMAND ----------

print(len(predictions))

# COMMAND ----------

original_test_df.reset_index(drop=True, inplace=True)

# 2. Create the Id column from this clean dataframe.
id_column = original_test_df['Store'].astype(str) + '_' + \
            original_test_df['Dept'].astype(str) + '_' + \
            original_test_df['Date'].astype(str)

# 3. Create the final DataFrame from a dictionary. This is the safest method.
submission_df = pd.DataFrame({
    'Id': id_column,
    'Weekly_Sales': predictions
})

# --- Add explicit length check before saving ---
print(f"Length of original test data: {len(original_test_df)}")
print(f"Length of predictions array:  {len(predictions)}")
print(f"Length of final submission DF:  {len(submission_df)}")

if len(submission_df) != 115064:
    dbutils.notebook.exit(f"FATAL ERROR: Final DataFrame has {len(submission_df)} rows, but expected 115064.")

# An important business rule: sales cannot be negative.
submission_df['Weekly_Sales'] = submission_df['Weekly_Sales'].clip(lower=0)

# --- 4.2: Save the File ---
SUBMISSION_DIR = '/dbfs/FileStore/walmart_project/submissions'
if not os.path.exists(SUBMISSION_DIR):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

submission_file_path = os.path.join(SUBMISSION_DIR, 'submission_lightgbm_v2_final.csv')
submission_df.to_csv(submission_file_path, index=False)

print(f"\nSubmission file created successfully at: {submission_file_path}")
print("\nSubmission file head:")
display(submission_df)


# COMMAND ----------

# Define the path of the file you want to download
dbfs_path = "/FileStore/walmart_project/submissions/submission_lightgbm_v2_final.csv"

# Copy the file from the internal DBFS to the publicly accessible /FileStore/ path
# This makes it downloadable via a URL
dbutils.fs.cp(dbfs_path, "dbfs:/FileStore/submission_to_download.csv")

