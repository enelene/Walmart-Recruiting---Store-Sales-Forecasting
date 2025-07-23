# Databricks notebook source
# MAGIC %md
# MAGIC # SECTION 1: IMPORTS & LIBRARY INSTALLATION

# COMMAND ----------

!pip install -r requirements.txt

# COMMAND ----------

!pip install joblib
from joblib import Parallel, delayed

# COMMAND ----------

import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
import mlflow
import mlflow.statsmodels
import dagshub

# COMMAND ----------

print("--- Setting up MLflow and Dagshub ---")
warnings.filterwarnings("ignore")
try:
    dagshub.init(repo_owner='enelene', repo_name='Walmart-Recruiting---Store-Sales-Forecasting', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/enelene/Walmart-Recruiting---Store-Sales-Forecasting.mlflow")
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'enelene'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cbe8109dbe80931664d754dbd476356414fa62a0'
    EXPERIMENT_NAME = "SARIMAX_Full_Production_Run"
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow experiment set to: '{EXPERIMENT_NAME}'")
except Exception as e:
    print(f"MLflow/Dagshub setup failed: {e}")


# COMMAND ----------

print("\n--- Loading All Preprocessed Data ---")
PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
# We need the full history (train + validation) for final model training
TRAIN_PATH = os.path.join(PROCESSED_DIR, 'train_final.csv')
VALIDATION_PATH = os.path.join(PROCESSED_DIR, 'validation_final.csv')
TEST_PATH = os.path.join(PROCESSED_DIR, 'test_final.csv')

try:
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['Date'])
    validation_df = pd.read_csv(VALIDATION_PATH, parse_dates=['Date'])
    test_df = pd.read_csv(TEST_PATH, parse_dates=['Date'])
    full_train_df = pd.concat([train_df, validation_df])
    print("Successfully loaded and combined all data.")
except FileNotFoundError:
    print("ERROR: Data not found. Please run the preprocessing script first.")


# COMMAND ----------

# --- 3. Define the Training Function & Metrics ---
def train_and_forecast_one_series(store, dept, train_group, test_group):
    order = (1, 1, 1)
    seasonal_order = (0, 1, 1, 52)
    exog_features = [col for col in train_group.columns if col not in ['Weekly_Sales', 'Store_Dept', 'Store', 'Dept', 'Date']]
    exog_features = [f for f in exog_features if f in test_group.columns]
    predictions = []
    if train_group.empty:
        for date_index in test_group.index:
            predictions.append({'Store': store, 'Dept': dept, 'Date': date_index, 'Weekly_Sales': 0})
        return predictions
    y_train = train_group['Weekly_Sales'].asfreq('W-FRI').fillna(method='ffill').fillna(method='bfill').fillna(0)
    X_train = train_group[exog_features].asfreq('W-FRI').fillna(method='ffill').fillna(method='bfill').fillna(0)
    X_test = test_group[exog_features].asfreq('W-FRI').fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    try:
        model = SARIMAX(endog=y_train, exog=X_train, order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=len(X_test), exog=X_test)
        predicted_sales = forecast.predicted_mean
        for date_index, prediction in predicted_sales.items():
            predictions.append({'Store': store, 'Dept': dept, 'Date': date_index, 'Weekly_Sales': prediction})
    except Exception:
        last_known_sale = y_train.iloc[-1] if not y_train.empty else 0
        for date_index in test_group.index:
            predictions.append({'Store': store, 'Dept': dept, 'Date': date_index, 'Weekly_Sales': last_known_sale})
    return predictions

# COMMAND ----------


def wmae(y_true, y_pred, is_holiday):
    weights = np.where(np.array(is_holiday, dtype=bool), 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# --- 4. Prepare Data Chunks for Parallel Processing ---
store_dept_to_predict = test_df[['Store', 'Dept']].drop_duplicates().to_records(index=False)
job_args = []

print("\n--- Preparing data chunks for parallel execution ---")
for store, dept in tqdm(store_dept_to_predict):
    train_group = full_train_df[(full_train_df['Store'] == store) & (full_train_df['Dept'] == dept)].set_index('Date')
    test_group = test_df[(test_df['Store'] == store) & (test_df['Dept'] == dept)].set_index('Date')
    job_args.append((store, dept, train_group, test_group))

# --- 5. Main Parallel Execution Loop with MLflow Logging ---
print(f"\n--- Starting PARALLEL SARIMAX forecasting for {len(job_args)} combinations ---")


# COMMAND ----------


# --- 5. Main Parallel Execution Loop with MLflow Logging ---
print(f"\n--- Starting PARALLEL SARIMAX forecasting for {len(job_args)} combinations ---")

with mlflow.start_run(run_name="Full_SARIMAX_Prediction_Job_Robust") as parent_run:
    mlflow.log_param("total_series_to_predict", len(job_args))
    mlflow.log_param("model_type", "SARIMAX (Parallel)")
    mlflow.log_param("order", str((1, 1, 1)))
    mlflow.log_param("seasonal_order", str((0, 1, 1, 52))) # Log the new robust order

    results_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(train_and_forecast_one_series)(*args) for args in tqdm(job_args)
    )

    all_predictions = [item for sublist in results_list for item in sublist]

    # --- 6. Create and Log Final Submission File ---
    print("\n--- Creating Final Submission File ---")
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date']).dt.strftime('%Y-%m-%d')
    predictions_df['Id'] = predictions_df['Store'].astype(str) + '_' + predictions_df['Dept'].astype(str) + '_' + predictions_df['Date']
    predictions_df['Weekly_Sales'] = predictions_df['Weekly_Sales'].clip(lower=0)
    submission_df = predictions_df[['Id', 'Weekly_Sales']]
    
    SUBMISSION_DIR = '/dbfs/FileStore/walmart_project/submissions'
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, 'parallel_sarimax_submission_robust.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print("\n--- Logging submission file as an artifact to MLflow ---")
    mlflow.log_artifact(submission_path, "submission")

    print(f"\nSubmission file created successfully at: {submission_path}")
    print("Job complete. Check MLflow for the parent run and artifact.")


# COMMAND ----------


# --- 4. Main Experiment Run with MLflow ---
with mlflow.start_run(run_name="Full_SARIMAX_Prediction_Job") as parent_run:
    mlflow.log_param("model_type", "SARIMAX (Parallel)")
    mlflow.log_param("order", str((1, 1, 1)))
    mlflow.log_param("seasonal_order", str((1, 1, 1, 52)))

    # --- STAGE 1: EVALUATION ON VALIDATION SET ---
    print("\n--- Stage 1: Evaluating performance on the validation set ---")
    val_job_args = []
    store_dept_val = validation_df[['Store', 'Dept']].drop_duplicates().to_records(index=False)
    for store, dept in tqdm(store_dept_val):
        train_chunk = train_df[(train_df['Store'] == store) & (train_df['Dept'] == dept)].set_index('Date')
        val_chunk = validation_df[(validation_df['Store'] == store) & (validation_df['Dept'] == dept)].set_index('Date')
        if not train_chunk.empty:
            val_job_args.append((store, dept, train_chunk, val_chunk))

    val_results_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(train_and_forecast_one_series)(*args) for args in tqdm(val_job_args)
    )
    val_predictions = [item for sublist in val_results_list for item in sublist]
    val_preds_df = pd.DataFrame(val_predictions)

    # Merge predictions with actuals to calculate metrics
    eval_df = pd.merge(validation_df, val_preds_df, on=['Store', 'Dept', 'Date'], suffixes=('_actual', '_pred'))
    
    # Calculate overall metrics
    wmae_score = wmae(eval_df['Weekly_Sales_actual'], eval_df['Weekly_Sales_pred'], eval_df['IsHoliday'])
    mae_score = mae(eval_df['Weekly_Sales_actual'], eval_df['Weekly_Sales_pred'])
    rmse_score = rmse(eval_df['Weekly_Sales_actual'], eval_df['Weekly_Sales_pred'])
    mape_score = mape(eval_df['Weekly_Sales_actual'], eval_df['Weekly_Sales_pred'])

    print("\n--- Validation Metrics ---")
    print(f"Overall WMAE: {wmae_score:.2f}")
    print(f"Overall MAE:  {mae_score:.2f}")
    print(f"Overall RMSE: {rmse_score:.2f}")
    print(f"Overall MAPE: {mape_score:.2f}%")

    # Log metrics to MLflow
    mlflow.log_metric("validation_wmae", wmae_score)
    mlflow.log_metric("validation_mae", mae_score)
    mlflow.log_metric("validation_rmse", rmse_score)
    mlflow.log_metric("validation_mape", mape_score)

    # --- STAGE 2: GENERATE FINAL SUBMISSION FILE ---
    print("\n--- Stage 2: Generating final predictions on the test set ---")
    test_job_args = []
    store_dept_to_predict = test_df[['Store', 'Dept']].drop_duplicates().to_records(index=False)
    for store, dept in tqdm(store_dept_to_predict):
        train_group = full_train_df[(full_train_df['Store'] == store) & (full_train_df['Dept'] == dept)].set_index('Date')
        test_group = test_df[(test_df['Store'] == store) & (test_df['Dept'] == dept)].set_index('Date')
        test_job_args.append((store, dept, train_group, test_group))

    results_list = Parallel(n_jobs=-1, backend="loky")(
        delayed(train_and_forecast_one_series)(*args) for args in tqdm(test_job_args)
    )
    all_predictions = [item for sublist in results_list for item in sublist]

    # --- Create and Log Submission File ---
    print("\n--- Creating and Logging Final Submission File ---")
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date']).dt.strftime('%Y-%m-%d')
    predictions_df['Id'] = predictions_df['Store'].astype(str) + '_' + predictions_df['Dept'].astype(str) + '_' + predictions_df['Date']
    predictions_df['Weekly_Sales'] = predictions_df['Weekly_Sales'].clip(lower=0)
    submission_df = predictions_df[['Id', 'Weekly_Sales']]
    
    SUBMISSION_DIR = '/dbfs/FileStore/walmart_project/submissions'
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, 'parallel_sarimax_submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    mlflow.log_artifact(submission_path, "submission")
    print(f"\nSubmission file created and logged successfully at: {submission_path}")


# COMMAND ----------

# MAGIC %md
# MAGIC comparison of arima sarima and sarimax 

# COMMAND ----------


# --- 3. Select a Robust Candidate Series ---
print("\n--- Selecting a stable candidate series for comparison ---")

# Calculate statistics for each series
series_stats = full_train_df.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'count']).dropna()
# Calculate Coefficient of Variation (CoV)
series_stats['cov'] = series_stats['std'] / series_stats['mean']
# Filter for series that have the full history (143 weeks) and positive sales
series_stats = series_stats[(series_stats['count'] >= 143) & (series_stats['mean'] > 0)]

# Select the most stable series (lowest CoV)
best_candidate = series_stats.sort_values(by='cov', ascending=True).index[0]
STORE_ID, DEPT_ID = best_candidate
print(f"Selected candidate with lowest CoV: Store {STORE_ID}, Department {DEPT_ID}")

# --- Prepare Data for the Selected Series ---
# Split data into train and validation for the selected series
full_ts_df = full_train_df[(full_train_df['Store'] == STORE_ID) & (full_train_df['Dept'] == DEPT_ID)].set_index('Date').sort_index()
train_size = int(len(full_ts_df) * 0.8) # Use an 80/20 split for this demonstration
train_ts_df, validation_ts_df = full_ts_df.iloc[:train_size], full_ts_df.iloc[train_size:]

# Prepare endogenous variable (y)
y_train = train_ts_df['Weekly_Sales'].asfreq('W-FRI').fillna(method='ffill')
y_val = validation_ts_df['Weekly_Sales'].asfreq('W-FRI').fillna(method='ffill')

# Prepare exogenous variables (X) for SARIMAX
exog_features_to_exclude = [col for col in train_ts_df.columns if 'Sales_' in col]
exog_features_to_exclude += ['Weekly_Sales', 'Store_Dept', 'Store', 'Dept', 'Weekly_Returns']
exog_features = [col for col in train_ts_df.columns if col not in exog_features_to_exclude]
X_train = train_ts_df[exog_features].asfreq('W-FRI').fillna(method='ffill')
X_val = validation_ts_df[exog_features].asfreq('W-FRI').fillna(method='ffill')


# COMMAND ----------


# --- 4. Define Evaluation Metrics ---
def wmae(y_true, y_pred, is_holiday):
    weights = np.where(np.array(is_holiday, dtype=bool), 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


# COMMAND ----------

import pmdarima as pm

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

all_forecasts = {}
all_metrics = {}

# -- Model 1: Auto ARIMA (non-seasonal) --
with mlflow.start_run(run_name=f"AutoARIMA_S{STORE_ID}_D{DEPT_ID}"):
    print("\n--- Training Auto ARIMA (non-seasonal) model ---")
    arima_model = pm.auto_arima(y_train, seasonal=False, stepwise=True, suppress_warnings=True, trace=False, error_action='ignore')
    print(f"Best ARIMA order: {arima_model.order}")
    all_forecasts['ARIMA'] = pd.Series(arima_model.predict(n_periods=len(y_val)), index=y_val.index)

# -- Model 2: Auto SARIMA (seasonal) --
with mlflow.start_run(run_name=f"AutoSARIMA_S{STORE_ID}_D{DEPT_ID}"):
    print("\n--- Training Auto SARIMA (seasonal) model ---")
    sarima_model = pm.auto_arima(y_train, seasonal=True, m=52, stepwise=True, suppress_warnings=True, trace=False, error_action='ignore')
    print(f"Best SARIMA order: {sarima_model.order}, seasonal_order: {sarima_model.seasonal_order}")
    all_forecasts['SARIMA'] = pd.Series(sarima_model.predict(n_periods=len(y_val)), index=y_val.index)




# COMMAND ----------


# --- 6. Calculate Metrics and Compare ---
is_holiday_val = validation_ts_df['IsHoliday']
for model_name, forecast in all_forecasts.items():
    all_metrics[model_name] = {
        'WMAE': wmae(y_val, forecast, is_holiday_val),
        'MAE': mae(y_val, forecast),
        'RMSE': rmse(y_val, forecast),
        'MAPE': mape(y_val, forecast)
    }

metrics_df = pd.DataFrame(all_metrics).T
print("\n--- Model Performance Comparison ---")
print(metrics_df.round(2))

# Log metrics to the last run (SARIMAX) for a summary
for model_name, metrics in all_metrics.items():
    for metric_name, value in metrics.items():
        mlflow.log_metric(f"{model_name}_{metric_name}", value)

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

# --- 7. Visualize the Results ---
print("\n--- Generating Forecast Comparison Plot ---")
plt.figure(figsize=(18, 9))
plt.plot(y_train.index, y_train, label='Training Data', color='gray', alpha=0.7)
plt.plot(y_val.index, y_val, label='Actual Sales (Validation)', color='blue', linewidth=2.5)

colors = ['orange', 'green', 'red']
for i, (model_name, forecast) in enumerate(all_forecasts.items()):
    wmae_score = all_metrics[model_name]['WMAE']
    plt.plot(forecast.index, forecast, label=f'{model_name} Forecast (WMAE: {wmae_score:.2f})', color=colors[i], linestyle='--')

plt.title(f'Model Comparison for Store {STORE_ID}, Dept {DEPT_ID}', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weekly Sales', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
