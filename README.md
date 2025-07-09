# Walmart-Recruiting---Store-Sales-Forecasting
Kaggle competition as Final project for ML

Elene's experiments: 
https://dagshub.com/enelene/Walmart-Recruiting---Store-Sales-Forecasting
https://wandb.ai/egabe21-free-university-of-tbilisi-/Walmart-Sales-Forecasting-DL?nw=nwuseregabe21

File 1: 00_initial_data_exploration.ipynb
Purpose: The single most important setup script. Its only job is to be run once to download the raw data from Kaggle, perform all the complex feature engineering using the logic from src/preprocessing.py, and save the final, clean train_processed_final.csv and test_processed_final.csv files. This ensures that all subsequent modeling notebooks start from the exact same, clean data source.

Key Moments:

Data Merging: It correctly performs a left merge to combine stores.csv and features.csv with the main train.csv and test.csv files. This enriches sales data with contextual information.

Output: The key output is the final, processed CSV files in the data/processed directory, which all other notebooks will use.

File 2: 01_advanced_data_exploration.py
Purpose: To visually explore the processed data to gain insights and justify your modeling decisions. This notebook is for analysis.

Key Moments:

Sales Heatmap: This visualization proves the existence of strong yearly seasonality by showing that sales patterns (e.g., high in Nov/Dec, lower in Jan/Feb) are consistent across the years.
![image](https://github.com/user-attachments/assets/cea03bf4-a4b1-48b1-b7aa-1ffd9ce29790)
![image](https://github.com/user-attachments/assets/91d0743a-6243-44a5-9de9-96ecac52497e)
Because the trend is not flat, the data is non-stationary. (useful for an ARIMA model)
Seasonal: For a SARIMA model, this tells you two things: you must use seasonal differencing (D=1), and seasonal period is m=52 (since it's weekly data repeating annually).
Residuals: This is the random noise or error that's left over after the trend and seasonality have been removed.

Correlation Matrix: This is a critical plot. It shows that there is no strong linear relationship between features like Temperature or CPI and Weekly_Sales. Thats why a simple linear regression model would fail and why we needed more complex models like LightGBM or Deep Learning, which can find non-linear patterns.
![image](https://github.com/user-attachments/assets/2ff41d4f-593c-4d3f-909a-539267f72f3d)


Department Boxplot: This plot is the best visual justification for groupby(['Store', 'Dept']) logic. It clearly shows that different departments have vastly different sales volumes and volatility, proving that they must be treated as separate entities.

![image](https://github.com/user-attachments/assets/268c4225-f55e-4216-b97e-e3ef4e703335)


File 3: model_experiment_LightGBM.py
Purpose: To train, tune, and evaluate a global LightGBM model. This represents the "feature-based" approach to forecasting.

Key Moments:

Global Model: training one single model on the entire dataset. The model learns from all 3,000+ time series simultaneously, allowing it to discover global patterns (e.g., "all electronics departments get a sales boost during the Super Bowl").

TimeSeriesSplit: This is the correct way to do cross-validation for time series data. It ensures that you always train on past data and validate on future data, preventing data leakage.

Hyperparameter Tuning with Optuna: used a optimization library to automatically find the best settings for model. Optuna ran many trials to find the learning_rate and num_leaves that produced the lowest error.

Categorical Feature Handling: give LightGBM info about which columns to treat as categories (Store, Dept, etc.). This is more efficient and often more effective than one-hot encoding for tree-based models.

MLflow Pipeline (WalmartSalesPipeline): a custom pyfunc pipeline that bundles preprocessing function with trained model. This creates a single, robust artifact that can take raw data as input and produce predictions, which is essential for deployment and inference.

File 4: model_experiment_ARIMA.py
Purpose: To train, tune, and evaluate a local SARIMAX model for a sample of the best candidate series. This represents the "classical statistical" approach.

Key Moments:

Local Model: This notebook demonstrates the opposite approach of LightGBM. It builds a separate, specialized model for each individual Store-Dept pair.

Candidate Selection: data-driven approach to find the most stable, complete time series to model, giving the architecture its best chance to succeed.

Two-Stage Automated Modeling:

Stage 1 (Feature Selection): used a RandomForest to automatically find the top 6 most important external features for that specific time series.

Stage 2 (Guided auto_arima): then used auto_arima to find the best time series parameters (p,d,q,P,D,Q), but made its job easier and more stable by only giving it the 6 features it needed.

The "X" in SARIMAX: used SARIMAX, where the "X" stands for eXogenous variables. This means using the classic ARIMA model by giving it the external features selected in Stage 1.

Files 5 & 6: model_experiment_NBEATS.py & model_experiment_TFT_darts.py
Purpose: To explore deep learning solutions using the darts library, which proved to be more stable in my environment. 
Key Moments:

Pivoting to a New Library (darts): We initially attempted to use pytorch-forecasting, but encountered persistent, environment-specific versioning errors. To overcome this and ensure project completion, we pivoted to the more stable darts library.

Global vs. Local in Deep Learning:  N-BEATS model was a local model (trained on one series) and performed poorly, confirming the limitations of this approach. TFT model was a global model that learned from all series at once.

The TimeSeries Object: darts uses a special TimeSeries object that bundles the data and its timeline together.

The Power of TFT: the Temporal Fusion Transformer is the most advanced model you tested. It's a global model like LightGBM, but its internal "attention mechanism" is specifically designed to understand time and automatically learn which features and past time steps are most important, making it a very powerful tool for this kind of complex, multivariate problem.

Files 7 & 8: The Inference Notebooks
Purpose: To load a final, trained model from your experiment tracking server (Dagshub) and use it to generate a submission.csv file for Kaggle.

Key Moments:

Loading from the Model Registry: mlflow.pyfunc.load_model("models:/...")

End-to-End Pipeline: For the LightGBM inference, loaded the WalmartSalesPipeline. it takes raw test data, performs all the advanced feature engineering internally, and outputs the final predictions. 

Iterative Inference (for SARIMAX): For the SARIMAX submission, had to create a loop that re-trained a new model for every single row in the test set. You should present this as a key finding: while the local model approach is interesting, it is computationally impractical for real-world deployment, as it took a very long time to generate the submission file.
