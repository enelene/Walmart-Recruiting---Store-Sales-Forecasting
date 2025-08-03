import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
import wandb
from wandb.integration.xgboost import WandbCallback
import os # To handle file paths

# Import your preprocessing function
from src.preprocessing import advanced_feature_engineering

# --- 0. Weights & Biases Login ---
wandb.login()

# --- 1. Load Raw Data ---
print("--- Loading Raw Data ---")
# Adjust these paths based on where your script is run from
# If you run `python main.py` from `your_project/`, these paths are correct
train_df = pd.read_csv('data/train.csv')
features_df = pd.read_csv('data/features.csv')
stores_df = pd.read_csv('data/stores.csv')

# Merge the datasets (assuming common columns like 'Store', 'Date', 'IsHoliday')
# This merge order is typical for the Walmart competition
df = pd.merge(train_df, stores_df, on='Store', how='left')
df = pd.merge(df, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')

print("Raw data loaded and merged.")
print(f"Initial DataFrame shape: {df.shape}")
print(df.head())

# --- 2. Apply Advanced Feature Engineering ---
print("\n--- Applying Advanced Feature Engineering ---")
processed_df = advanced_feature_engineering(df.copy()) # Pass a copy to avoid modifying original df

print("Feature engineering complete.")
print(f"Processed DataFrame shape: {processed_df.shape}")
print(processed_df.head())

# --- 3. Prepare Data for XGBoost ---
# Define features (X) and target (y)
# Exclude original 'Date', 'IsHoliday', 'Weekly_Sales' (if present), 'Store_Dept' (if using it as category directly)
# and original markdown columns, as new ones are derived
feature_cols = [col for col in processed_df.columns if col not in ['Date', 'Weekly_Sales', 'IsHoliday', 'Store_Dept_cat']] # Assuming Store_Dept will be passed as a category type, not integer after encoding if needed
target_col = 'Weekly_Sales'

# Handle 'Store_Dept' as a proper category in XGBoost
# For XGBoost's native DMatrix, you need to convert it to an integer or one-hot encode it.
# For sklearn API, it often handles pandas category dtype directly if specified.
# Let's convert it to categorical dtype for XGBoost and get the feature names list without it.
processed_df['Store_Dept_cat'] = processed_df['Store_Dept'].astype('category')
# To get numerical codes for DMatrix
processed_df['Store_Dept_code'] = processed_df['Store_Dept_cat'].cat.codes

# Update feature_cols to include the code and exclude the original string category
feature_cols = [col for col in processed_df.columns if col not in ['Date', 'Weekly_Sales', 'IsHoliday', 'Store_Dept', 'Store_Dept_cat']]
feature_cols.append('Store_Dept_code') # Add the numerical code for DMatrix

X = processed_df[feature_cols]
y = processed_df[target_col]

# Identify categorical features for XGBoost (if using native DMatrix)
categorical_features_indices = [X.columns.get_loc(col) for col in X.select_dtypes(include='category').columns]


# For the `Store_Dept_code`, if it was already treated as a numeric column and then you want to tell XGBoost it's categorical:
# You'd need to convert to category *before* splitting for sklearn API, or use `enable_categorical=True` with `DMatrix`.
# For simplicity, let's treat `Store_Dept_code` as another numerical feature here, or ensure it's encoded properly.

# Let's refine feature_cols for clarity based on your preprocessing output
# `Type` is already one-hot encoded, `Store_Dept` is a new categorical
final_feature_columns = [
    'Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsBlackFridayWeek',
    'IsSuperBowlWeek', 'IsLaborDayWeek', 'IsChristmasWeek', 'Year', 'Month', 'WeekOfYear', 'Day',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'HasMarkdown',
    'Type_A', 'Type_B', 'Type_C', # Assuming these are your one-hot encoded Type columns
    'Store_Dept_code' # The numerical representation of Store_Dept
]

# Add lag and rolling features if they exist after handling initial NaNs from them
for lag in [1, 2, 4, 13, 26, 52]:
    if f'Sales_Lag_{lag}' in processed_df.columns:
        final_feature_columns.append(f'Sales_Lag_{lag}')
rolling_features = ['Sales_Roll_Mean_4', 'Sales_Roll_Std_4', 'Sales_Roll_Min_4', 'Sales_Roll_Max_4']
for rf in rolling_features:
    if rf in processed_df.columns:
        final_feature_columns.append(rf)

# Make sure all final_feature_columns actually exist in processed_df
final_feature_columns = [col for col in final_feature_columns if col in processed_df.columns]

X = processed_df[final_feature_columns]
y = processed_df['Weekly_Sales']


# Important: Handle NaNs introduced by lag/rolling features after splitting, if any
# Your `advanced_feature_engineering` function attempts to fill all NaNs.
# However, the very first few rows for each Store/Dept group will still have NaNs
# for lag features. These should ideally be dropped or filled appropriately *before* training.
# Since your function `fillna`s everything, we'll trust that, but be aware.
# If you don't drop/fill these, XGBoost will raise errors or perform poorly.

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix for native XGBoost API
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print("Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 4. Define and Train the Model (Regression Example) ---
print("\n--- Starting Model Training ---")

# Base configuration for this model run
base_config = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.01, # Slower learning rate
    "n_estimators": 1500,   # More estimators due to slower LR
    "max_depth": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "seed": 42,
    "feature_set_version": "Advanced_V2_AllFeatures" # Unique identifier for this feature set
}

with wandb.init(project="Walmart-Sales-Forecasting", job_type="model-training", config=base_config, reinit=True) as run:
    params = dict(run.config)
    del params['n_estimators']
    del params['feature_set_version'] # This is just for logging config, not an XGBoost param

    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=run.config.n_estimators,
        evals=[(dtrain, 'train'), (dtest, 'validation')],
        callbacks=[WandbCallback(log_model=True, log_feature_importance=True)],
        verbose_eval=100
    )

    # Make predictions
    y_pred = xgb_model.predict(dtest)

    # Evaluate and log final metrics
    rmse_final = np.sqrt(mean_squared_error(y_test, y_pred))
    wandb.log({"final_rmse": rmse_final})

    print(f"Model Training Complete. Final RMSE: {rmse_final:.4f}")

# --- End of script ---
print("\nScript execution finished.")
