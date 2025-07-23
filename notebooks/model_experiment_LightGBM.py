# Databricks notebook source
!pip install -r requirements.txt

# COMMAND ----------

!pip install dagshub

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import optuna
import os
import dagshub

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from src.preprocessing import advanced_feature_engineering

# COMMAND ----------

# MAGIC %md
# MAGIC # LOAD PROCESSED DATA

# COMMAND ----------

PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
TRAIN_PATH = os.path.join(PROCESSED_DIR, 'train_processed_final.csv')
RAW_DATA_DIR = '/dbfs/FileStore/walmart_project/data/raw'

try:
    train_df = pd.read_csv(TRAIN_PATH)
    raw_train_for_example = pd.read_csv(os.path.join(RAW_DATA_DIR, 'train.csv'))
    print("Successfully loaded processed training data from DBFS.")
except FileNotFoundError:
    print(f"ERROR: Processed data not found at '{TRAIN_PATH}'.")
    dbutils.notebook.exit("Data preparation notebook must be run first.")


# COMMAND ----------

from src.preprocessing import advanced_feature_engineering

# COMMAND ----------

# MAGIC %md
# MAGIC # MLFLOW SETUP FOR DAGSHUB AND MODEL PREPARATION

# COMMAND ----------

dagshub.init(repo_owner='enelene', repo_name='Walmart-Recruiting---Store-Sales-Forecasting', mlflow=True)


# COMMAND ----------

mlflow.set_tracking_uri("https://dagshub.com/enelene/Walmart-Recruiting---Store-Sales-Forecasting.mlflow")

os.environ['MLFLOW_TRACKING_USERNAME'] = 'enelene'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cbe8109dbe80931664d754dbd476356414fa62a0'

EXPERIMENT_NAME = "LightGBM_Training"
print(f"Experiment set to: '{EXPERIMENT_NAME}'")
# Define Features (X) and Target (y)
TARGET = 'Weekly_Sales'
features_to_drop = [TARGET, 'Date']
features = [col for col in train_df.columns if col not in features_to_drop]

categorical_features = [
    'Store', 'Dept', 'IsHoliday', 'Year', 'Month', 'WeekOfYear', 'HasMarkdown', 
    'Store_Dept', 'Type_A', 'Type_B', 'Type_C', 'IsSuperBowlWeek', 
    'IsLaborDayWeek', 'IsThanksgivingWeek', 'IsChristmasWeek'
]

categorical_features = [f for f in categorical_features if f in train_df.columns]

for col in categorical_features:
    train_df[col] = train_df[col].astype('category')

X = train_df[features]
y = train_df[TARGET]

def wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# COMMAND ----------

# MAGIC %md
# MAGIC # MLFLOW EXPERIMENT RUNS

# COMMAND ----------

# TimeSeriesSplit is the correct way to cross-validate time-series data.
tscv = TimeSeriesSplit(n_splits=3)

# COMMAND ----------

# This objective function now performs cross-validation *inside* the tuning loop
# for a more robust hyperparameter search.
def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    params = {
        'objective': 'regression_l1',  # MAE
        'metric': 'mae',
        'n_estimators': 1500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    wmae_scores = []
    # Loop through each cross-validation split
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='mae',
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  categorical_feature=categorical_features)
                  
        preds = model.predict(X_val)
        is_holiday_val = X_val['IsHoliday'].astype(bool)
        score = wmae(y_val, preds, is_holiday_val)
        wmae_scores.append(score)

    return np.mean(wmae_scores)



# COMMAND ----------

with mlflow.start_run(run_name="LGBM_Tuning_and_Final_Model") as run:
    print("--- Starting Hyperparameter Tuning with Optuna (using full CV) ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30) # Increase trials for a more thorough search
    
    best_params = study.best_params
    best_wmae = study.best_value
    
    print(f"Best WMAE from tuning: {best_wmae:.4f}")
    print(f"Best Parameters: {best_params}")
    
    mlflow.log_params(best_params)
    mlflow.log_metric("best_wmae_tuned", best_wmae)
    
    # --- Train and Log Final Model ---
    print("\n--- Training Final Model on All Data ---")
    
    # Add n_estimators and random_state to the best params
    final_params = best_params
    final_params['n_estimators'] = 2500 # Use a higher number for the final model
    final_params['random_state'] = 42
    
    final_model = lgb.LGBMRegressor(**final_params)
    
    # Train on the entire dataset (X, y)
    final_model.fit(X, y, categorical_feature=categorical_features)
    print("Final model training complete.")
    
    # --- Package the Model and Preprocessing into a Pipeline ---
    
    

# COMMAND ----------

training_categories = {
    col: X[col].cat.categories for col in categorical_features if hasattr(X[col], 'cat')
}

class WalmartSalesPipeline(mlflow.pyfunc.PythonModel):
    def __init__(self, model, feature_engineering_fn, training_columns, categories):
        self.model = model
        self._feature_engineering_fn = feature_engineering_fn
        self._training_columns = training_columns
        self._categories = categories

    def predict(self, context, model_input):
        # 1. Apply the same feature engineering
        processed_input = self._feature_engineering_fn(model_input)
        
        # 2. Enforce consistent categories
        for col, cats in self._categories.items():
            if col in processed_input.columns:
                processed_input[col] = pd.Categorical(processed_input[col], categories=cats)

        # 3. Ensure all training columns are present and in the correct order
        processed_input = processed_input.reindex(columns=self._training_columns, fill_value=0)
        
        # 4. Predict
        return self.model.predict(processed_input)

print("\n--- Logging and Registering Final Model Pipeline ---")

# The input example should be in the RAW format, before preprocessing
input_example = raw_train_for_example.head(5).drop('Weekly_Sales', axis=1, errors='ignore')

mlflow.pyfunc.log_model(
    artifact_path="lightgbm-sales-pipeline",
    python_model=WalmartSalesPipeline(final_model, advanced_feature_engineering, features, training_categories),
    registered_model_name="LightGBM-Walmart-Sales-Pipeline",
    input_example=input_example
)

print("Model Pipeline successfully logged and registered!")
