# Databricks notebook source
# MAGIC %md
# MAGIC # SECTION 1: IMPORTS & LIBRARY INSTALLATION

# COMMAND ----------

!pip install -r requirements.txt

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import mlflow.statsmodels
import os
import matplotlib.pyplot as plt
import pmdarima as pm

from sklearn.metrics import mean_absolute_error


# COMMAND ----------

# MAGIC %md
# MAGIC LOAD PROCESSED DATA FROM DBFS

# COMMAND ----------

PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
TRAIN_PATH = os.path.join(PROCESSED_DIR, 'train_processed_final.csv')

try:
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=['Date'])
    print("Successfully loaded processed training data from DBFS.")
except FileNotFoundError:
    print(f"ERROR: Processed data not found at '{TRAIN_PATH}'.")
    dbutils.notebook.exit("Data preparation notebook must be run first.")


# COMMAND ----------

# MAGIC %md
# MAGIC MLFLOW SETUP AND DATA SELECTION

# COMMAND ----------

mlflow.set_tracking_uri("https://dagshub.com/enelene/Walmart-Recruiting---Store-Sales-Forecasting.mlflow")

os.environ['MLFLOW_TRACKING_USERNAME'] = 'enelene'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cbe8109dbe80931664d754dbd476356414fa62a0'

EXPERIMENT_NAME = "ARIMA_SARIMA_Training"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment set to: '{EXPERIMENT_NAME}'")



# COMMAND ----------

# Select a Single Time Series for Modeling 
STORE_ID = 1
DEPT_ID = 1
print(f"\nModeling for Store {STORE_ID}, Department {DEPT_ID}")

ts_df = train_df[(train_df['Store'] == STORE_ID) & (train_df['Dept'] == DEPT_ID)].copy()
ts = ts_df[['Date', 'Weekly_Sales', 'IsHoliday']].set_index('Date').sort_index()
ts = ts.asfreq('W-FRI')
ts['Weekly_Sales'] = ts['Weekly_Sales'].fillna(method='ffill')
ts['IsHoliday'] = ts['IsHoliday'].fillna(method='ffill')

# Train-Test Split
train_size = int(len(ts) * 0.8)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]
print(f"Train size: {len(train)}, Test size: {len(test)}")

# COMMAND ----------

# MAGIC %md
# MAGIC AUTOMATED SARIMA MODELING

# COMMAND ----------

with mlflow.start_run(run_name=f"AutoSARIMA_S{STORE_ID}_D{DEPT_ID}"):
    print("\n--- Finding best SARIMA order with auto_arima ---")
    
    # Use auto_arima to find the best (p,d,q)(P,D,Q,m) parameters
    # m=52 tells it to look for yearly seasonality in the weekly data.
    # stepwise=True makes the search much faster.
    auto_model = pm.auto_arima(
        train['Weekly_Sales'],
        start_p=1, start_q=1,
        test='adf',       # Use adf test to find 'd'
        max_p=3, max_q=3, # Maximum p and q
        m=52,             # The period for seasonal differencing
        d=None,           # Let the model determine 'd'
        seasonal=True,    # Fit a seasonal model
        start_P=0,
        D=None,           # Let the model determine 'D'
        trace=True,
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True
    )

    # Print the summary of the best model found
    print("\n--- Best Model Summary ---")
    print(auto_model.summary())

    # Get the best parameters found
    best_order = auto_model.order
    best_seasonal_order = auto_model.seasonal_order
    
    print(f"\nBest Order (p,d,q): {best_order}")
    print(f"Best Seasonal Order (P,D,Q,m): {best_seasonal_order}")

    # Log the discovered parameters to MLflow
    mlflow.log_param("model_type", "AutoSARIMA")
    mlflow.log_param("store_id", STORE_ID)
    mlflow.log_param("dept_id", DEPT_ID)
    mlflow.log_param("best_order", str(best_order))
    mlflow.log_param("best_seasonal_order", str(best_seasonal_order))

    # --- Evaluate the Best Model ---
    print("\n--- Evaluating the best model on the test set ---")
    
    # Get predictions for the test set period
    predictions = auto_model.predict(n_periods=len(test))
    
    # Create a pandas Series with the correct index for evaluation
    predictions = pd.Series(predictions, index=test.index)

    # Calculate WMAE
    def wmae(y_true, y_pred, is_holiday):
        weights = np.where(is_holiday, 5, 1)
        return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

    score_sarima = wmae(test['Weekly_Sales'], predictions, test['IsHoliday'])
    print(f"Auto-SARIMA Model WMAE: {score_sarima:.2f}")
    mlflow.log_metric("wmae", score_sarima)

    # Log the final fitted model to Dagshub
    mlflow.pmdarima.log_model(
        auto_model,
        artifact_path="auto-sarima-model",
        registered_model_name=f"AutoSARIMA-S{STORE_ID}-D{DEPT_ID}"
    )

    # Log a forecast plot
    fig, ax = plt.subplots(figsize=(15, 6))
    train['Weekly_Sales'].plot(ax=ax, label='Train')
    test['Weekly_Sales'].plot(ax=ax, label='Test')
    predictions.plot(ax=ax, label='Auto-SARIMA Forecast')
    ax.legend()
    plt.title("Auto-SARIMA Forecast vs Actuals")
    mlflow.log_figure(fig, "auto_sarima_forecast_plot.png")

print("\nAuto-SARIMA experiment complete. Check your Dagshub dashboard.")


# COMMAND ----------

# MAGIC %md
# MAGIC CANDIDATE SELECTION & DATA PREPARATION

# COMMAND ----------

series_stats = train_df.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'count']).dropna()
series_stats['cov'] = series_stats['std'] / series_stats['mean']
series_stats = series_stats[series_stats['count'] >= 143]
best_candidate = series_stats.sort_values(by='cov', ascending=True).index[0]
STORE_ID, DEPT_ID = best_candidate
print(f"Selected a robust candidate series: Store {STORE_ID}, Department {DEPT_ID}")

#Prepare Data for SARIMAX 
ts_df = train_df[(train_df['Store'] == STORE_ID) & (train_df['Dept'] == DEPT_ID)].copy()
y = ts_df.set_index('Date')['Weekly_Sales'].asfreq('W-FRI').fillna(method='ffill')

# Define and clean exogenous features
exog_features = [col for col in ts_df.columns if col not in ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Store_Dept']]
X_exog_raw = ts_df.set_index('Date')[exog_features].asfreq('W-FRI').fillna(method='ffill')
bool_cols = X_exog_raw.select_dtypes(include='bool').columns
X_exog = X_exog_raw.copy()
X_exog[bool_cols] = X_exog[bool_cols].astype(int)
print("All exogenous features are numeric.")

# Train-Test Split 
train_size = int(len(y) * 0.8)
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
X_train, X_test = X_exog.iloc[:train_size], X_exog.iloc[train_size:]
print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

# COMMAND ----------

# We will use the ARIMA function directly from statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# COMMAND ----------


with mlflow.start_run(run_name=f"Guided_SARIMAX_S{STORE_ID}_D{DEPT_ID}"):
    print("\n--- Finding best SARIMAX order with a guided auto_arima ---")
    
    # Guide auto_arima by pre-setting d and D and limiting the search space.
    # This is much more stable and faster.
    auto_model = pm.auto_arima(
        y=y_train,
        X=X_train,
        m=52,
        seasonal=True,
        d=1,  # Force one non-seasonal difference
        D=1,  # Force one seasonal difference
        start_p=0, start_q=0, start_P=0, start_Q=0,
        max_p=2, max_q=2, max_P=1, max_Q=1, # Limit the search space
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )

    print("\n--- Best Model Summary ---")
    print(auto_model.summary())

    mlflow.log_param("model_type", "Guided_AutoSARIMAX")
    mlflow.log_param("best_order", str(auto_model.order))
    mlflow.log_param("best_seasonal_order", str(auto_model.seasonal_order))

    # --- Evaluate the Best Model ---
    print("\n--- Evaluating the best model on the test set ---")
    predictions = auto_model.predict(n_periods=len(y_test), X=X_test)
    
    def wmae(y_true, y_pred, is_holiday):
        weights = np.where(is_holiday, 5, 1)
        return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

    # --- THE FIX for the WMAE ValueError ---
    # Get the 'IsHoliday' values directly from the test set dataframe.
    is_holiday_test = X_test['IsHoliday'].values
    
    print(f"Shape of y_test: {y_test.shape}")
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of is_holiday_test: {is_holiday_test.shape}")

    score_sarimax = wmae(y_test, predictions, is_holiday_test)
    
    print(f"Guided Auto-SARIMAX Model WMAE: {score_sarimax:.2f}")
    mlflow.log_metric("wmae", score_sarimax)

    # Log the final fitted model to Dagshub
    mlflow.pmdarima.log_model(
        auto_model,
        artifact_path="guided-sarimax-model",
        registered_model_name=f"Guided-SARIMAX-S{STORE_ID}-D{DEPT_ID}"
    )

print("\nGuided Auto-SARIMAX experiment complete.")


# COMMAND ----------

print("\n--- SECTION 3: CANDIDATE SELECTION & DATA PREPARATION ---")

# --- 3.1: Find a Good Candidate Time Series ---
series_stats = train_df.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'count']).dropna()
series_stats['cov'] = series_stats['std'] / series_stats['mean']
series_stats = series_stats[series_stats['count'] >= 143]
best_candidate = series_stats.sort_values(by='cov', ascending=True).index[0]
STORE_ID, DEPT_ID = best_candidate
print(f"Selected candidate series: Store {STORE_ID}, Department {DEPT_ID}")

# --- 3.2: Prepare Data for SARIMAX ---
ts_df = train_df[(train_df['Store'] == STORE_ID) & (train_df['Dept'] == DEPT_ID)].copy()
y = ts_df.set_index('Date')['Weekly_Sales'].asfreq('W-FRI').fillna(method='ffill')
exog_features_all = [col for col in ts_df.columns if col not in ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Store_Dept']]
X_exog_raw = ts_df.set_index('Date')[exog_features_all].asfreq('W-FRI').fillna(method='ffill')
bool_cols = X_exog_raw.select_dtypes(include='bool').columns
X_exog_all = X_exog_raw.copy()
X_exog_all[bool_cols] = X_exog_all[bool_cols].astype(int)
print("All potential exogenous features are prepared.")

# --- 3.3: Train-Test Split for the entire dataset ---
train_size = int(len(y) * 0.8)
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
X_train_all, X_test_all = X_exog_all.iloc[:train_size], X_exog_all.iloc[train_size:]
print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# COMMAND ----------

print("\n--- SECTION 4: TWO-STAGE AUTOMATED SARIMAX MODELING ---")

with mlflow.start_run(run_name=f"FeatureSelected_SARIMAX_S{STORE_ID}_D{DEPT_ID}"):
    # --- STAGE 1: Automated Feature Selection ---
    print("\n--- Stage 1: Finding most important features with RandomForest ---")
    
    # Use a simple RandomForest to rank features
    # We use the training data to find the best features
    feature_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    feature_selector.fit(X_train_all, y_train)
    
    # Create a DataFrame of feature importances
    importances = pd.DataFrame({
        'feature': X_train_all.columns,
        'importance': feature_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select the top N features
    N_TOP_FEATURES = 6
    top_features = importances.head(N_TOP_FEATURES)['feature'].tolist()
    
    print(f"Top {N_TOP_FEATURES} features selected: {top_features}")
    mlflow.log_param("selected_features", top_features)

    # Create new training and test sets with only the top features
    X_train_selected = X_train_all[top_features]
    X_test_selected = X_test_all[top_features]

    # --- STAGE 2: Guided auto_arima with Selected Features ---
    print("\n--- Stage 2: Finding best SARIMAX order with selected features ---")
    
    auto_model = pm.auto_arima(
        y=y_train,
        X=X_train_selected, # Use only the best features
        m=52,
        seasonal=True,
        d=1, D=1, # Guide the model
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )

    print("\n--- Best Model Summary ---")
    print(auto_model.summary())

    mlflow.log_param("model_type", "Feature-Selected_AutoSARIMAX")
    mlflow.log_param("best_order", str(auto_model.order))
    mlflow.log_param("best_seasonal_order", str(auto_model.seasonal_order))

    # --- Evaluate the Best Model ---
    print("\n--- Evaluating the best model on the test set ---")
    predictions = auto_model.predict(n_periods=len(y_test), X=X_test_selected)
    
    def wmae(y_true, y_pred, is_holiday):
        weights = np.where(is_holiday, 5, 1)
        return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

    is_holiday_test = X_test_all['IsHoliday'].values
    score_sarimax = wmae(y_test, predictions, is_holiday_test)
    
    print(f"Feature-Selected Auto-SARIMAX Model WMAE: {score_sarimax:.2f}")
    mlflow.log_metric("wmae", score_sarimax)

    # Log the final fitted model to Dagshub
    mlflow.pmdarima.log_model(
        auto_model,
        artifact_path="fs-sarimax-model",
        registered_model_name=f"FS-SARIMAX-S{STORE_ID}-D{DEPT_ID}"
    )

print("\nFeature-Selected Auto-SARIMAX experiment complete.")


# COMMAND ----------

# SECTION 3: CANDIDATE SELECTION
# ==============================================================================
print("\n--- SECTION 3: CANDIDATE SELECTION ---")

# --- 3.1: Find the Top 5 Best Candidate Time Series ---
series_stats = train_df.groupby(['Store', 'Dept'])['Weekly_Sales'].agg(['mean', 'std', 'count']).dropna()
series_stats['cov'] = series_stats['std'] / series_stats['mean']
# Ensure the series has the full history
series_stats = series_stats[series_stats['count'] >= 143]
# Select the top 5 most stable (lowest CoV) series
top_5_candidates = series_stats.sort_values(by='cov', ascending=True).head(5).index.tolist()
top_5_candidates.remove((37, 13))

print(f"Selected Top 5 candidate series: {top_5_candidates}")

# COMMAND ----------

print("\n--- SECTION 4: LOOPED SARIMAX MODELING FOR TOP 5 ---")

# We will start one parent MLflow run to hold the results of all 5 models.
with mlflow.start_run(run_name="SARIMAX_Top_5_Evaluation") as parent_run:
    
    wmae_scores = []
    mlflow.log_param("candidate_series", str(top_5_candidates))

    # Loop through each of the top 5 candidates
    for i, (store_id, dept_id) in enumerate(top_5_candidates):
        
        # Start a nested run for this specific series
        with mlflow.start_run(run_name=f"SARIMAX_S{store_id}_D{dept_id}", nested=True):
            
            print(f"\n--- Processing Model {i+1}/5: Store {store_id}, Dept {dept_id} ---")
            
            # --- 1. Prepare Data for this specific series ---
            ts_df = train_df[(train_df['Store'] == store_id) & (train_df['Dept'] == dept_id)].copy()
            y = ts_df.set_index('Date')['Weekly_Sales'].asfreq('W-FRI').fillna(method='ffill')
            exog_features_all = [col for col in ts_df.columns if col not in ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Store_Dept']]
            X_exog_raw = ts_df.set_index('Date')[exog_features_all].asfreq('W-FRI').fillna(method='ffill')
            bool_cols = X_exog_raw.select_dtypes(include='bool').columns
            X_exog_all = X_exog_raw.copy()
            X_exog_all[bool_cols] = X_exog_all[bool_cols].astype(int)

            train_size = int(len(y) * 0.8)
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            X_train_all, X_test_all = X_exog_all.iloc[:train_size], X_exog_all.iloc[train_size:]

            # --- 2. Automated Feature Selection ---
            feature_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            feature_selector.fit(X_train_all, y_train)
            importances = pd.DataFrame({
                'feature': X_train_all.columns,
                'importance': feature_selector.feature_importances_
            }).sort_values('importance', ascending=False)
            
            N_TOP_FEATURES = 6
            top_features = importances.head(N_TOP_FEATURES)['feature'].tolist()
            X_train_selected = X_train_all[top_features]
            X_test_selected = X_test_all[top_features]
            
            mlflow.log_param("selected_features", top_features)

            # --- 3. Guided auto_arima ---
            auto_model = pm.auto_arima(
                y=y_train, X=X_train_selected, m=52, seasonal=True, d=1, D=1,
                stepwise=True, suppress_warnings=True, error_action='ignore', trace=False
            )
            
            mlflow.log_param("best_order", str(auto_model.order))
            mlflow.log_param("best_seasonal_order", str(auto_model.seasonal_order))

            # --- 4. Evaluate and Log ---
            predictions = auto_model.predict(n_periods=len(y_test), X=X_test_selected)
            
            def wmae(y_true, y_pred, is_holiday):
                weights = np.where(is_holiday, 5, 1)
                return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

            is_holiday_test = X_test_all['IsHoliday'].values
            score_sarimax = wmae(y_test, predictions, is_holiday_test)
            wmae_scores.append(score_sarimax)
            
            print(f"WMAE for S{store_id}-D{dept_id}: {score_sarimax:.2f}")
            mlflow.log_metric("wmae", score_sarimax)
            mlflow.pmdarima.log_model(auto_model, artifact_path=f"sarimax_S{store_id}_D{dept_id}")

    # --- After the loop, calculate and log the average score ---
    average_wmae = np.mean(wmae_scores)
    print(f"\n----------------------------------------------------")
    print(f"Average WMAE across Top 5 Models: {average_wmae:.2f}")
    print(f"----------------------------------------------------")
    mlflow.log_metric("average_wmae_top5", average_wmae)

print("\nSARIMAX evaluation for top 5 series is complete.")

