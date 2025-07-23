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

mlflow.set_tracking_uri("https://dagshub.com/enelene/Walmart-Recruiting---Store-Sales-Forecasting.mlflow")

os.environ['MLFLOW_TRACKING_USERNAME'] = 'enelene'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cbe8109dbe80931664d754dbd476356414fa62a0'

model_name = "LightGBM-Walmart-Sales-Pipeline"
model_uri = f"models:/{model_name}/latest"

print(f"Loading model pipeline from: {model_uri}")
try:
    # mlflow.pyfunc.load_model will download the model artifacts and reconstruct the pipeline.
    loaded_model_pipeline = mlflow.pyfunc.load_model(model_uri)
    print("Model pipeline loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    # In a real workflow, you might want to exit the notebook here.
    # dbutils.notebook.exit("Model loading failed.")



# COMMAND ----------


RAW_DATA_DIR = '/dbfs/FileStore/walmart_project/data/raw'
try:
    # Load the raw competition test set
    raw_test_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'test.csv'))
    
    # We also need the features and stores data to merge, just like in preprocessing.
    features_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'features.csv'))
    stores_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'stores.csv'))

    # Merge the raw test data
    test_data_to_predict = raw_test_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
    test_data_to_predict = test_data_to_predict.merge(stores_df, on='Store', how='left')
    
    print("Raw test data loaded and merged successfully.")
    print(f"Shape of the data to predict: {test_data_to_predict.shape}")

except FileNotFoundError as e:
    print(f"Error loading raw test data: {e}")
    # dbutils.notebook.exit("Raw data not found.")

# COMMAND ----------


print("\nMaking predictions on the test set...")
predictions = loaded_model_pipeline.predict(test_data_to_predict)
print("Predictions complete.")

# --- 5. Create Submission File ---
# The final step is to format the predictions into the required submission format.

# Create a submission DataFrame
submission_df = test_data_to_predict[['Store', 'Dept', 'Date']].copy()
submission_df['Weekly_Sales'] = predictions

# The competition requires an 'Id' column in the format 'Store_Dept_Date'
submission_df['Date'] = pd.to_datetime(submission_df['Date']).dt.strftime('%Y-%m-%d')
submission_df['Id'] = submission_df['Store'].astype(str) + '_' + submission_df['Dept'].astype(str) + '_' + submission_df['Date']

# Select and reorder columns for the final file
final_submission = submission_df[['Id', 'Weekly_Sales']]

# Ensure there are no negative predictions
final_submission['Weekly_Sales'] = final_submission['Weekly_Sales'].clip(lower=0)

# Save the submission file
SUBMISSION_DIR = '/dbfs/FileStore/walmart_project/submissions'
if not os.path.exists(SUBMISSION_DIR):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

submission_path = os.path.join(SUBMISSION_DIR, 'lgbm_submission.csv')
final_submission.to_csv(submission_path, index=False)

print(f"\nSubmission file created successfully at: {submission_path}")
print("Sample of the submission file:")
print(final_submission.head())


# COMMAND ----------


