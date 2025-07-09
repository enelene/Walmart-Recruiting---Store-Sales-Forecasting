# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import pandas as pd
import numpy as np
import os
import pmdarima as pm
from sklearn.ensemble import RandomForestRegressor
from tqdm.notebook import tqdm
import warnings


# COMMAND ----------

RAW_DATA_DIR = '/dbfs/FileStore/walmart_project/data/raw'
try:
    # We need the full raw training data to build models for each series
    train_df_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, 'train.csv'))
    test_df_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, 'test.csv'))
    features_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'features.csv'))
    stores_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'stores.csv'))
    print("Successfully loaded all raw data files.")
except FileNotFoundError:
    dbutils.notebook.exit("Raw data files not found. Please run the initial data prep notebook.")


# COMMAND ----------

from src.preprocessing import advanced_feature_engineering

print("Merging and preprocessing all data...")
# Combine train and test for consistent feature engineering
test_df_raw['Weekly_Sales'] = np.nan # Add dummy column
combined_df = pd.concat([train_df_raw, test_df_raw])

# Merge features
combined_df = combined_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
combined_df = combined_df.merge(stores_df, on='Store', how='left')

# Process the entire dataset at once
# Note: The 'Weekly_Sales' for the test set portion will be NaN
processed_df = advanced_feature_engineering(combined_df)
print("Full dataset processed.")


# COMMAND ----------

# MAGIC %md
# MAGIC ITERATIVE INFERENCE WORKFLOW 

# COMMAND ----------


test_pairs = test_df_raw[['Store', 'Dept']].drop_duplicates().to_records(index=False)
all_predictions = []

# Loop through every single pair that needs a prediction
for store_id, dept_id in tqdm(test_pairs):
    
    # --- 1. Isolate Data for this Series ---
    series_df = processed_df[(processed_df['Store'] == store_id) & (processed_df['Dept'] == dept_id)].copy()
    series_df.set_index('Date', inplace=True)
    series_df = series_df.asfreq('W-FRI') # Ensure consistent frequency
    
    # Separate the historical data (for training) and the future data (for prediction)
    train_history = series_df[series_df['Weekly_Sales'].notna()]
    predict_data = series_df[series_df['Weekly_Sales'].isna()]
    
    # Skip if there's no history to train on
    if len(train_history) < 52: # Need at least one year of data
        # If we can't train a model, predict the mean of the last few weeks
        simple_prediction = train_history['Weekly_Sales'].tail(4).mean()
        predictions_for_series = pd.Series(simple_prediction, index=predict_data.index)
        all_predictions.append(predictions_for_series)
        continue

    # --- 2. Prepare Data for this Series ---
    y_train = train_history['Weekly_Sales']
    
    exog_features_all = [col for col in train_history.columns if col not in ['Store', 'Dept', 'Weekly_Sales', 'Store_Dept']]
    X_train_all = train_history[exog_features_all]
    X_predict = predict_data[exog_features_all]
    
    # Convert booleans to integers
    bool_cols = X_train_all.select_dtypes(include='bool').columns
    X_train_all[bool_cols] = X_train_all[bool_cols].astype(int)
    X_predict[bool_cols] = X_predict[bool_cols].astype(int)

    # --- 3. Two-Stage Modeling (Feature Selection + auto_arima) ---
    try:
        # Stage 1: Feature Selection
        feature_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        feature_selector.fit(X_train_all, y_train)
        importances = pd.DataFrame({
            'feature': X_train_all.columns,
            'importance': feature_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        N_TOP_FEATURES = 6
        top_features = importances.head(N_TOP_FEATURES)['feature'].tolist()
        
        X_train_selected = X_train_all[top_features]
        X_predict_selected = X_predict[top_features]

        # Stage 2: Guided auto_arima
        auto_model = pm.auto_arima(
            y=y_train, X=X_train_selected, m=52, seasonal=True, d=1, D=1,
            stepwise=True, suppress_warnings=True, error_action='ignore', trace=False
        )
        
        # --- 4. Predict and Store ---
        predictions = auto_model.predict(n_periods=len(X_predict), X=X_predict_selected)
        predictions_for_series = pd.Series(predictions, index=X_predict.index)
        all_predictions.append(predictions_for_series)

    except Exception as e:
        # Fallback for any series that still fails
        simple_prediction = train_history['Weekly_Sales'].tail(4).mean()
        predictions_for_series = pd.Series(simple_prediction, index=predict_data.index)
        all_predictions.append(predictions_for_series)


# COMMAND ----------

# MAGIC %md
# MAGIC CREATE AND SAVE SUBMISSION FILE

# COMMAND ----------

final_predictions_series = pd.concat(all_predictions).sort_index()

# Create the submission DataFrame
submission_df = pd.DataFrame(final_predictions_series, columns=['Weekly_Sales']).reset_index()
submission_df.rename(columns={'index': 'Date'}, inplace=True)

# Merge with original test data to get Store and Dept
submission_df = test_df_raw[['Store', 'Dept', 'Date']].merge(submission_df, on='Date', how='left')

# Create the final 'Id' column
submission_df['Id'] = submission_df['Store'].astype(str) + '_' + submission_df['Dept'].astype(str) + '_' + submission_df['Date'].astype(str)
submission_df['Weekly_Sales'] = submission_df['Weekly_Sales'].clip(lower=0).fillna(0)

# Save the file
SUBMISSION_DIR = '/dbfs/FileStore/walmart_project/submissions'
if not os.path.exists(SUBMISSION_DIR):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

submission_file_path = os.path.join(SUBMISSION_DIR, 'submission_sarimax.csv')
submission_df[['Id', 'Weekly_Sales']].to_csv(submission_file_path, index=False)

print(f"Submission file created successfully at: {submission_file_path}")
display(submission_df[['Id', 'Weekly_Sales']].head())

