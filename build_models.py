# build_models.py

# Import necessary libraries
import pandas as pd
from prophet import Prophet
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to preprocess payments data
def preprocess_payments(df):
    df = df.copy()
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df = df.groupby(pd.Grouper(key='TransactionDate', freq='M')).agg({'TransactionAmount': 'sum'}).reset_index()
    df.rename(columns={'TransactionDate': 'ds', 'TransactionAmount': 'y'}, inplace=True)
    return df

# Function to preprocess visits data
def preprocess_visits(df):
    df = df.copy()
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df = df.groupby(pd.Grouper(key='TransactionDate', freq='M')).size().reset_index(name='y')
    df.rename(columns={'TransactionDate': 'ds'}, inplace=True)
    return df

# Function to calculate RMSE and MAPE
def calculate_metrics(actual_df, forecast_df):
    merged_df = pd.merge(actual_df, forecast_df[['ds', 'yhat']], on='ds')
    rmse = np.sqrt(np.mean((merged_df['y'] - merged_df['yhat']) ** 2))
    mape = np.mean(np.abs((merged_df['y'] - merged_df['yhat']) / merged_df['y'])) * 100
    return {'RMSE': rmse, 'MAPE': mape}

# Function to calculate CAGR
def calculate_cagr(start_value, end_value, periods_in_years):
    if start_value <= 0 or end_value <= 0 or periods_in_years <= 0:
        return None
    cagr = ((end_value / start_value) ** (1 / periods_in_years) - 1) * 100
    return cagr

# Function to validate DataFrame
def validate_dataframe(df, df_name):
    if df.empty:
        logger.error(f"{df_name} DataFrame is empty.")
        raise ValueError(f"{df_name} DataFrame is empty.")
    if not {'ds', 'y'}.issubset(df.columns):
        logger.error(f"{df_name} DataFrame must contain 'ds' and 'y' columns.")
        raise ValueError(f"{df_name} DataFrame must contain 'ds' and 'y' columns.")

# Read the data
try:
    data_path = os.path.join(BASE_DIR, 'payment.xlsx')
    payments_df = pd.read_excel(data_path)
    payments_df = payments_df[payments_df['EncStatus'] == 'CHK']
    logger.info("Successfully read 'payment.xlsx' and filtered by EncStatus 'CHK'.")
except FileNotFoundError:
    logger.error("The file 'payment.xlsx' was not found.")
    raise
except Exception as e:
    logger.error(f"Error reading 'payment.xlsx': {e}")
    raise

# Filter data for UPFH Family Clinic - West Jordan
west_jordan_df = payments_df[payments_df['Facility'] == 'UPFH Family Clinic - West Jordan']
logger.info(f"Records for 'UPFH Family Clinic - West Jordan': {len(west_jordan_df)}")

# Preprocess data
west_jordan_payments = preprocess_payments(west_jordan_df)
west_jordan_visits = preprocess_visits(west_jordan_df)
logger.info("Preprocessed payments and visits data for West Jordan Clinic.")

entire_payments = preprocess_payments(payments_df)
entire_visits = preprocess_visits(payments_df)
logger.info("Preprocessed payments and visits data for Entire Payments Data Set.")

# Validate DataFrames
validate_dataframe(west_jordan_payments, "west_jordan_payments")
validate_dataframe(west_jordan_visits, "west_jordan_visits")
validate_dataframe(entire_payments, "entire_payments")
validate_dataframe(entire_visits, "entire_visits")

# Function to fit Prophet model and forecast
def fit_and_forecast(df, periods, freq='M'):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

# Forecast periods (e.g., 12 months)
forecast_periods = 12

# West Jordan Clinic forecasts
payments_forecast_wj = fit_and_forecast(west_jordan_payments, periods=forecast_periods)
visits_forecast_wj = fit_and_forecast(west_jordan_visits, periods=forecast_periods)

# Entire Data Set forecasts
payments_forecast_entire = fit_and_forecast(entire_payments, periods=forecast_periods)
visits_forecast_entire = fit_and_forecast(entire_visits, periods=forecast_periods)

# Calculate metrics for West Jordan Clinic
payments_metrics_wj = calculate_metrics(west_jordan_payments, payments_forecast_wj)
visits_metrics_wj = calculate_metrics(west_jordan_visits, visits_forecast_wj)

# Calculate metrics for Entire Data Set
payments_metrics_entire = calculate_metrics(entire_payments, payments_forecast_entire)
visits_metrics_entire = calculate_metrics(entire_visits, visits_forecast_entire)

# Save forecasts
payments_forecast_wj.to_csv(os.path.join(BASE_DIR, 'payments_forecast_wj.csv'), index=False)
visits_forecast_wj.to_csv(os.path.join(BASE_DIR, 'visits_forecast_wj.csv'), index=False)
payments_forecast_entire.to_csv(os.path.join(BASE_DIR, 'payments_forecast_entire.csv'), index=False)
visits_forecast_entire.to_csv(os.path.join(BASE_DIR, 'visits_forecast_entire.csv'), index=False)

# Save metrics
pd.DataFrame([payments_metrics_wj]).to_csv(os.path.join(BASE_DIR, 'payments_metrics_wj.csv'), index=False)
pd.DataFrame([visits_metrics_wj]).to_csv(os.path.join(BASE_DIR, 'visits_metrics_wj.csv'), index=False)
pd.DataFrame([payments_metrics_entire]).to_csv(os.path.join(BASE_DIR, 'payments_metrics_entire.csv'), index=False)
pd.DataFrame([visits_metrics_entire]).to_csv(os.path.join(BASE_DIR, 'visits_metrics_entire.csv'), index=False)

# Save actual data for plotting
west_jordan_payments.to_csv(os.path.join(BASE_DIR, 'west_jordan_payments.csv'), index=False)
west_jordan_visits.to_csv(os.path.join(BASE_DIR, 'west_jordan_visits.csv'), index=False)
entire_payments.to_csv(os.path.join(BASE_DIR, 'entire_payments.csv'), index=False)
entire_visits.to_csv(os.path.join(BASE_DIR, 'entire_visits.csv'), index=False)

logger.info("Forecasts, metrics, and actual data have been saved to CSV files.")
