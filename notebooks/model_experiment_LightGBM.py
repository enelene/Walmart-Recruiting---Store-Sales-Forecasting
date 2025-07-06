# Databricks notebook source
!pip install -r requirements.txt

# COMMAND ----------

import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import optuna
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from src.preprocessing import advanced_feature_engineering

# COMMAND ----------

# MAGIC %md
# MAGIC # LOAD PROCESSED DATA

# COMMAND ----------

PROCESSED_DIR = '/dbfs/FileStore/walmart_project/data/processed'
TRAIN_PATH = os.path.join(PROCESSED_DIR, 'train_processed_final.csv')

try:
    train_df = pd.read_csv(TRAIN_PATH)
    print("Successfully loaded processed training data from DBFS.")
except FileNotFoundError:
    print(f"ERROR: Processed data not found at '{TRAIN_PATH}'.")
    dbutils.notebook.exit("Data preparation notebook must be run first.")


# COMMAND ----------

# MAGIC %md
# MAGIC # MLFLOW SETUP FOR DAGSHUB AND MODEL PREPARATION

# COMMAND ----------

import dagshub
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

X = train_df[features]
y = train_df[TARGET]

def wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# COMMAND ----------

# MAGIC %md
# MAGIC # MLFLOW EXPERIMENT RUNS

# COMMAND ----------

with mlflow.start_run(run_name="LGBM_Baseline"):
    print("\n--- Starting Run: LGBM_Baseline ---")
    model = lgb.LGBMRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    wmae_scores = []
    for train_index, val_index in tscv.split(X):
        X_t, X_v = X.iloc[train_index], X.iloc[val_index]
        y_t, y_v = y.iloc[train_index], y.iloc[val_index]
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        score = wmae(y_v, preds, X_v['IsHoliday'].astype(bool))
        wmae_scores.append(score)
    avg_wmae = np.mean(wmae_scores)
    print(f"Baseline Average WMAE: {avg_wmae:.2f}")
    mlflow.log_metric("avg_wmae_cv", avg_wmae)

# == Run 2: Hyperparameter Tuning with Optuna ==
with mlflow.start_run(run_name="LGBM_Hyperparameter_Tuning"):
    print("\n--- Starting Run: LGBM_Hyperparameter_Tuning ---")
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'verbose': -1, 'n_jobs': -1, 'seed': 42
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val)
        return wmae(y_val, preds, X_val['IsHoliday'].astype(bool))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    mlflow.log_params(best_params)
    mlflow.log_metric("best_wmae_tuned", study.best_value)

# == Run 3: Final Model & Registration ==
with mlflow.start_run(run_name="LGBM_Final_Pipeline"):
    print("\n--- Starting Run: LGBM_Final_Pipeline ---")
    final_params = best_params
    final_params['n_estimators'] = 2000
    final_params['random_state'] = 42
    mlflow.log_params(final_params)

    final_model = lgb.LGBMRegressor(**final_params)
    print("Training final model on all data...")
    final_model.fit(X, y)
    print("Training complete.")

    class WalmartSalesPipeline(mlflow.pyfunc.PythonModel):
        def __init__(self, model, feature_engineering_fn, training_columns):
            self.model = model
            self._feature_engineering_fn = feature_engineering_fn
            self._training_columns = training_columns
        
        def predict(self, context, model_input):
            processed_input = self._feature_engineering_fn(model_input)
            processed_input = processed_input.reindex(columns=self._training_columns, fill_value=0)
            return self.model.predict(processed_input)

    print("Logging and registering the final model pipeline to Dagshub...")
    mlflow.pyfunc.log_model(
        artifact_path="lightgbm-full-pipeline",
        python_model=WalmartSalesPipeline(final_model, advanced_feature_engineering, features),
        registered_model_name="LightGBM-Walmart-Sales-Pipeline",
        input_example=X.head(5)
    )
    print("Model Pipeline successfully logged and registered to Dagshub MLflow!")


# COMMAND ----------

# MAGIC %md
# MAGIC 2nd Iteration with some advanced featueres

# COMMAND ----------

mlflow.set_tracking_uri("https://dagshub.com/enelene/Walmart-Recruiting---Store-Sales-Forecasting.mlflow")

os.environ['MLFLOW_TRACKING_USERNAME'] = 'enelene'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'cbe8109dbe80931664d754dbd476356414fa62a0'

EXPERIMENT_NAME = "LightGBM_Training_V2_Advanced_Features"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment set to: '{EXPERIMENT_NAME}'")

# --- Prepare Data ---
TARGET = 'Weekly_Sales'
features_to_drop = [TARGET, 'Date']
categorical_features = ['Store', 'Dept', 'IsHoliday', 'Year', 'Month', 'WeekOfYear', 'HasMarkdown', 'Store_Dept', 'Type_A', 'Type_B', 'Type_C']
features = [col for col in train_df.columns if col not in features_to_drop]

# Convert categorical features for LightGBM
for col in categorical_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype('category')

X = train_df[features]
y = train_df[TARGET]

def wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# COMMAND ----------

with mlflow.start_run(run_name="LGBM_Baseline_V2"):
    print("\n--- Starting Run: LGBM_Baseline_V2 ---")
    model = lgb.LGBMRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    wmae_scores = []
    for train_index, val_index in tscv.split(X):
        X_t, X_v = X.iloc[train_index], X.iloc[val_index]
        y_t, y_v = y.iloc[train_index], y.iloc[val_index]
        # Tell LightGBM which features are categorical
        model.fit(X_t, y_t, categorical_feature=[col for col in categorical_features if col in X_t.columns])
        preds = model.predict(X_v)
        score = wmae(y_v, preds, X_v['IsHoliday'].astype(bool))
        wmae_scores.append(score)
    avg_wmae = np.mean(wmae_scores)
    print(f"Baseline Average WMAE (V2 Features): {avg_wmae:.2f}")
    mlflow.log_metric("avg_wmae_cv", avg_wmae)


# COMMAND ----------

with mlflow.start_run(run_name="LGBM_Hyperparameter_Tuning_V2"):
    print("\n--- Starting Run: LGBM_Hyperparameter_Tuning_V2 ---")
    train_idx, val_idx = list(tscv.split(X))[-1]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbose': -1, 'n_jobs': -1, 'seed': 42
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  eval_metric='mae', 
                  callbacks=[lgb.early_stopping(50, verbose=False)],
                  categorical_feature=[col for col in categorical_features if col in X_train.columns])
        preds = model.predict(X_val)
        return wmae(y_val, preds, X_val['IsHoliday'].astype(bool))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25) # Increased trials slightly
    best_params = study.best_params
    mlflow.log_params(best_params)
    mlflow.log_metric("best_wmae_tuned", study.best_value)


# COMMAND ----------

with mlflow.start_run(run_name="LGBM_Final_Pipeline_V2"):
    print("\n--- Starting Run: LGBM_Final_Pipeline_V2_---")
    final_params = best_params
    final_params['n_estimators'] = 2500
    final_params['random_state'] = 42
    mlflow.log_params(final_params)

    final_model = lgb.LGBMRegressor(**final_params)
    print("Training final model on all data...")
    # Pass the categorical features to fit for best performance
    final_model.fit(X, y, categorical_feature=[col for col in categorical_features if col in X.columns])
    print("Training complete.")

    # Store the learned categories from the training data
    training_categories = {
        col: X[col].cat.categories for col in categorical_features if col in X.columns and hasattr(X[col], 'cat')
    }

    class WalmartSalesPipeline(mlflow.pyfunc.PythonModel):
        def __init__(self, model, feature_engineering_fn, training_columns, categories):
            self.model = model
            self._feature_engineering_fn = feature_engineering_fn
            self._training_columns = training_columns
            self._categories = categories # Store the categories dict

        def predict(self, context, model_input):
            processed_input = self._feature_engineering_fn(model_input)
            for col, cats in self._categories.items():
                if col in processed_input.columns:
                    processed_input[col] = pd.Categorical(processed_input[col], categories=cats)

            # Reindex to ensure column order and handle any missing columns
            processed_input = processed_input.reindex(columns=self._training_columns, fill_value=0)
            
            return self.model.predict(processed_input)

    raw_train_df = pd.read_csv('/dbfs/FileStore/walmart_project/data/raw/train.csv')
    
    print("Logging and registering the fixed model pipeline to Dagshub...")
    mlflow.pyfunc.log_model(
        artifact_path="lightgbm-full-pipeline-v2-fixed",
        python_model=WalmartSalesPipeline(final_model, advanced_feature_engineering, features, training_categories),
        registered_model_name="LightGBM-Walmart-Sales-Pipeline", 
        input_example=raw_train_df.head(5).drop('Weekly_Sales', axis=1) 
    )
    print("Model Pipeline successfully logged and registered!")

