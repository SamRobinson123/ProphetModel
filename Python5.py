# Import necessary libraries
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import logging
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to create a Plotly graph for Prophet forecast with actual and forecasted data
def create_prophet_forecast_graph(actual_df, forecast_df, title, y_title, actual_name, forecast_name, rmse=None,
                                  mape=None, cagr=None):
    """
    Creates a Plotly graph for Prophet forecasts, displaying actual and forecasted data with confidence intervals.

    Parameters:
    - actual_df (DataFrame): Actual historical data with 'ds' and 'y' columns.
    - forecast_df (DataFrame): Forecasted data from Prophet with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    - title (str): Title of the graph.
    - y_title (str): Y-axis title.
    - actual_name (str): Label for actual data.
    - forecast_name (str): Label for forecasted data.
    - rmse (float, optional): Root Mean Squared Error metric.
    - mape (float, optional): Mean Absolute Percentage Error metric.
    - cagr (float, optional): Compound Annual Growth Rate metric.

    Returns:
    - fig (Figure): Plotly Figure object.
    """
    # Create the figure
    fig = go.Figure()

    # Add actual data (blue line with markers)
    fig.add_trace(go.Scatter(
        x=actual_df['ds'],
        y=actual_df['y'],
        mode='lines+markers',
        name=actual_name,
        line=dict(color='blue', width=2),
        marker=dict(size=5)
    ))

    # Add forecasted data (green line)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name=forecast_name,
        line=dict(color='green', width=3)
    ))

    # Add upper confidence interval (dashed green line)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        name='Upper Confidence Interval',
        line=dict(color='lightgreen', dash='dash'),
        showlegend=False
    ))

    # Add lower confidence interval (dashed green line)
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines',
        name='Lower Confidence Interval',
        line=dict(color='lightgreen', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(144, 238, 144, 0.2)',
        showlegend=False
    ))

    # Prepare the title with metrics if provided
    metrics_text = ""
    if rmse is not None and mape is not None:
        metrics_text += f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%"
    if cagr is not None:
        if metrics_text:
            metrics_text += ", "
        metrics_text += f"CAGR: {cagr:.2f}%"
    if metrics_text:
        title_with_metrics = f"{title}<br>{metrics_text}"
    else:
        title_with_metrics = title

    # Update the layout for titles and axes
    fig.update_layout(
        title=title_with_metrics,
        xaxis_title="Date",
        yaxis_title=y_title,
        template='plotly_white',
        height=600,
        margin=dict(l=60, r=60, t=100, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig

# Function to preprocess payments data
def preprocess_payments(df):
    """
    Preprocesses the payments DataFrame by renaming columns, converting dates, removing invalid entries,
    and aggregating data monthly.

    Parameters:
    - df (DataFrame): Raw payments data.

    Returns:
    - df (DataFrame): Preprocessed payments data with 'ds' and 'y' columns.
    """
    if 'EncDate' not in df.columns or 'NetPayment' not in df.columns:
        logger.error("The DataFrame does not contain required columns: 'EncDate', 'NetPayment'.")
        raise ValueError("Missing required columns in payments DataFrame.")

    # Drop unnecessary columns
    columns_to_drop = ['UserFirstName', 'UserLastName', 'ProviderName', 'ResourceName']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Rename columns
    df = df.rename(columns={'EncDate': 'ds', 'NetPayment': 'y'})

    # Ensure datetime
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

    # Drop NaNs and select relevant columns
    df = df[['ds', 'y']].dropna()

    # Remove non-positive payments
    df = df[df['y'] > 0]
    logger.info(f"Number of records after removing non-positive payments: {len(df)}")

    # Aggregate monthly using Month-End frequency ('ME')
    df = df.groupby(pd.Grouper(key='ds', freq='ME')).sum().reset_index()

    return df

# Function to preprocess visits data
def preprocess_visits(df):
    """
    Preprocesses the visits DataFrame by renaming columns, converting dates, removing invalid entries,
    and aggregating data monthly.

    Parameters:
    - df (DataFrame): Raw visits data.

    Returns:
    - df (DataFrame): Preprocessed visits data with 'ds' and 'y' columns.
    """
    if 'EncDate' not in df.columns:
        logger.error("The DataFrame does not contain the 'EncDate' column.")
        raise ValueError("Missing 'EncDate' column in visits DataFrame.")

    # Rename 'EncDate' to 'ds'
    df = df.rename(columns={'EncDate': 'ds'})

    # Ensure datetime
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['ds'])

    # Count visits using Month-End frequency ('ME')
    df = df.groupby(pd.Grouper(key='ds', freq='ME')).size().reset_index(name='y')

    # Remove negative visit counts (if any)
    df = df[df['y'] >= 0]
    logger.info(f"Number of records after ensuring non-negative visits: {len(df)}")

    return df

# Function to train Prophet model for payments
def train_prophet_payments(df, periods=60):
    """
    Trains a Prophet model on the payments data and generates forecasts.

    Parameters:
    - df (DataFrame): Preprocessed payments data with 'ds' and 'y' columns.
    - periods (int): Number of future periods to forecast (default: 60 months).

    Returns:
    - model (Prophet): Trained Prophet model.
    - forecast (DataFrame): Forecasted data.
    """
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='ME')
    forecast = model.predict(future)
    return model, forecast

# Function to train Prophet model for visits
def train_prophet_visits(df, periods=60):
    """
    Trains a Prophet model on the visits data and generates forecasts.

    Parameters:
    - df (DataFrame): Preprocessed visits data with 'ds' and 'y' columns.
    - periods (int): Number of future periods to forecast (default: 60 months).

    Returns:
    - model (Prophet): Trained Prophet model.
    - forecast (DataFrame): Forecasted data.
    """
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='ME')
    forecast = model.predict(future)
    return model, forecast

# Function to calculate performance metrics
def calculate_metrics(df, forecast):
    """
    Calculates performance metrics (RMSE and MAPE) between actual and forecasted data.

    Parameters:
    - df (DataFrame): Actual historical data with 'y' column.
    - forecast (DataFrame): Forecasted data with 'yhat' column.

    Returns:
    - metrics (dict): Dictionary containing RMSE and MAPE.
    """
    actual = df['y']
    predicted = forecast['yhat'][:len(actual)]
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(actual, predicted) * 100  # Convert to percentage
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

# Function to calculate CAGR
def calculate_cagr(start_value, end_value, periods_in_years):
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    Parameters:
    - start_value (float): Initial value.
    - end_value (float): Final value.
    - periods_in_years (float): Number of years.

    Returns:
    - cagr (float or None): CAGR percentage or None if undefined.
    """
    if start_value <= 0:
        return None  # CAGR is undefined if the start value is zero or negative
    cagr = ((end_value / start_value) ** (1 / periods_in_years) - 1) * 100
    return cagr

# Function to validate DataFrame
def validate_dataframe(df, name):
    """
    Validates the DataFrame for Prophet forecasting.

    Parameters:
    - df (DataFrame): DataFrame to validate.
    - name (str): Name of the DataFrame (for logging).

    Raises:
    - TypeError: If df is not a DataFrame.
    - ValueError: If required columns are missing, contain NaNs, or have insufficient rows.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(f"{name} is not a DataFrame.")
        raise TypeError(f"{name} must be a pandas DataFrame.")

    required_columns = {'ds', 'y'}
    if not required_columns.issubset(df.columns):
        logger.error(f"{name} does not contain required columns: {required_columns}")
        raise ValueError(f"{name} must contain columns: {required_columns}")

    if df[['ds', 'y']].isnull().any().any():
        logger.error(f"{name} contains NaN values in 'ds' or 'y' columns.")
        raise ValueError(f"{name} cannot contain NaN values in 'ds' or 'y' columns.")

    if len(df) < 2:
        logger.error(f"{name} has less than 2 records.")
        raise ValueError(f"{name} must have at least 2 records for Prophet forecasting.")

# Read the data
try:
    payments_df = pd.read_excel("payment.xlsx")
    payments_df = payments_df[payments_df['EncStatus'] == 'CHK']
    logger.info("Successfully read payment.xlsx and filtered by EncStatus 'CHK'.")
except FileNotFoundError:
    logger.error("The file 'payment.xlsx' was not found.")
    raise
except Exception as e:
    logger.error(f"Error reading 'payment.xlsx': {e}")
    raise

# Filter data for UPFH Family Clinic - West Jordan
if 'Facility' not in payments_df.columns:
    logger.error("The DataFrame does not contain the 'Facility' column.")
    raise ValueError("Missing 'Facility' column in payments DataFrame.")

west_jordan_df = payments_df[payments_df['Facility'] == 'UPFH Family Clinic - West Jordan']
logger.info(f"Number of records for UPFH Family Clinic - West Jordan: {len(west_jordan_df)}")

if west_jordan_df.empty:
    logger.error("No data found for 'UPFH Family Clinic - West Jordan'. Please check the 'Facility' column values.")
    raise ValueError("No data for 'UPFH Family Clinic - West Jordan'.")

# Verify columns in west_jordan_df
logger.info("Columns in west_jordan_df: {}".format(west_jordan_df.columns.tolist()))

# Preprocess data for West Jordan Clinic
try:
    west_jordan_payments = preprocess_payments(west_jordan_df)
    west_jordan_visits = preprocess_visits(west_jordan_df)
    logger.info("Successfully preprocessed payments and visits data for West Jordan Clinic.")
except Exception as e:
    logger.error(f"Error during data preprocessing for West Jordan Clinic: {e}")
    raise

# Validate DataFrames for West Jordan Clinic
validate_dataframe(west_jordan_payments, "west_jordan_payments")
validate_dataframe(west_jordan_visits, "west_jordan_visits")

# Preprocess data for Entire Payments Data Set
try:
    entire_payments = preprocess_payments(payments_df)
    entire_visits = preprocess_visits(payments_df)
    logger.info("Successfully preprocessed payments and visits data for Entire Payments Data Set.")
except Exception as e:
    logger.error(f"Error during data preprocessing for Entire Payments Data Set: {e}")
    raise

# Validate DataFrames for Entire Payments Data Set
validate_dataframe(entire_payments, "entire_payments")
validate_dataframe(entire_visits, "entire_visits")

# Train Prophet models and get forecasts for West Jordan Clinic (60 months)
try:
    payments_model_wj, payments_forecast_wj = train_prophet_payments(west_jordan_payments, periods=60)
    visits_model_wj, visits_forecast_wj = train_prophet_visits(west_jordan_visits, periods=60)
    logger.info("Successfully trained Prophet models and generated forecasts for West Jordan Clinic.")
except Exception as e:
    logger.error(f"Error during Prophet model training or forecasting for West Jordan Clinic: {e}")
    raise

# Train Prophet models and get forecasts for Entire Payments Data Set (60 months)
try:
    payments_model_entire, payments_forecast_entire = train_prophet_payments(entire_payments, periods=60)
    visits_model_entire, visits_forecast_entire = train_prophet_visits(entire_visits, periods=60)
    logger.info("Successfully trained Prophet models and generated forecasts for Entire Payments Data Set.")
except Exception as e:
    logger.error(f"Error during Prophet model training or forecasting for Entire Payments Data Set: {e}")
    raise

# Calculate metrics for West Jordan Clinic
try:
    payments_metrics_wj = calculate_metrics(west_jordan_payments, payments_forecast_wj)
    visits_metrics_wj = calculate_metrics(west_jordan_visits, visits_forecast_wj)
    logger.info("Successfully calculated performance metrics for West Jordan Clinic.")
except Exception as e:
    logger.error(f"Error calculating metrics for West Jordan Clinic: {e}")
    raise

# Calculate metrics for Entire Payments Data Set
try:
    payments_metrics_entire = calculate_metrics(entire_payments, payments_forecast_entire)
    visits_metrics_entire = calculate_metrics(entire_visits, visits_forecast_entire)
    logger.info("Successfully calculated performance metrics for Entire Payments Data Set.")
except Exception as e:
    logger.error(f"Error calculating metrics for Entire Payments Data Set: {e}")
    raise

# Calculate Historical Visitor CAGR for West Jordan Clinic (This will be used to initialize the input field)
payments_cagr_wj = calculate_cagr(
    start_value=west_jordan_payments['y'].iloc[0],
    end_value=west_jordan_payments['y'].iloc[-1],
    periods_in_years=5  # 60 months = 5 years
)

visits_cagr_wj = calculate_cagr(
    start_value=west_jordan_visits['y'].iloc[0],
    end_value=west_jordan_visits['y'].iloc[-1],
    periods_in_years=5
)

if visits_cagr_wj is None:
    initial_visitor_cagr = 0  # Default to 0% if historical CAGR is undefined
    logger.warning("Historical Visitor CAGR is undefined. Defaulting to 0%.")
else:
    initial_visitor_cagr = visits_cagr_wj
    logger.info(f"Historical Visitor CAGR (West Jordan Clinic, 2020-2025): {initial_visitor_cagr:.2f}%")

# Calculate Historical Visitor CAGR for Entire Payments Data Set (if needed)
payments_cagr_entire = calculate_cagr(
    start_value=entire_payments['y'].iloc[0],
    end_value=entire_payments['y'].iloc[-1],
    periods_in_years=5
)

visits_cagr_entire = calculate_cagr(
    start_value=entire_visits['y'].iloc[0],
    end_value=entire_visits['y'].iloc[-1],
    periods_in_years=5
)

if visits_cagr_entire is None:
    initial_visits_cagr_entire = 0  # Default to 0% if historical CAGR is undefined
    logger.warning("Historical Visitor CAGR for Entire Data Set is undefined. Defaulting to 0%.")
else:
    initial_visits_cagr_entire = visits_cagr_entire
    logger.info(f"Historical Visitor CAGR (Entire Data Set, 2020-2025): {initial_visits_cagr_entire:.2f}%")

# Initialize the Dash app before defining any callbacks
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# Define the Dash app layout
app.layout = dbc.Container([
    html.H1("Clinic Forecasting and Break-Even Analysis Dashboard", className="text-center text-primary mb-4"),
    html.Hr(),

    # Alert for errors
    dbc.Alert(
        id='error-message',
        is_open=False,
        duration=4000,
        color='danger',
        children=[]
    ),

    # Input Parameters (Assumptions)
    dbc.Row([
        dbc.Col([
            html.H3("Assumptions", className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    # Average Cost Per Visitor Text Input
                    dbc.Row([
                        dbc.Col(html.Label("Average Cost Per Visitor ($):"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='cost-per-visit',
                                type='number',
                                value=220,
                                min=0,
                                step=10,
                                style={'width': '100%'}
                            ),
                            width=6,
                        ),
                    ], className="mb-3"),
                    # Average Payment Per Visitor Text Input
                    dbc.Row([
                        dbc.Col(html.Label("Average Payment Per Visitor ($):"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='avg-payment-per-visitor',
                                type='number',
                                value=500,  # Default average payment
                                min=0,
                                step=10,
                                style={'width': '100%'}
                            ),
                            width=6,
                        ),
                    ], className="mb-3"),
                    # Initial Visitors Assumption Input
                    dbc.Row([
                        dbc.Col(html.Label("Initial Visitors:"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='initial-visitors',
                                type='number',
                                value=264,  # Updated initial visitors
                                min=0,
                                step=1,
                                style={'width': '100%'}
                            ),
                            width=6,
                        ),
                    ], className="mb-3"),
                    # Start-Up Costs Assumption Input
                    dbc.Row([
                        dbc.Col(html.Label("Start-Up Costs ($):"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='startup-costs',
                                type='number',
                                value=10000,  # Default start-up cost
                                min=0,
                                step=100,
                                style={'width': '100%'}
                            ),
                            width=6,
                        ),
                    ], className="mb-3"),
                    # New Row: Visitor CAGR Growth Rate Assumption Input
                    dbc.Row([
                        dbc.Col(html.Label("Visitor CAGR Growth Rate (%):"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='visitor-cagr',  # New input ID
                                type='number',
                                value=initial_visitor_cagr,  # **Set to historical CAGR**
                                min=-100,  # Minimum reasonable CAGR
                                max=100,    # Maximum reasonable CAGR
                                step=0.1,   # Step size for finer control
                                style={'width': '100%'}
                            ),
                            width=6,
                        ),
                    ], className="mb-3"),
                    # Refresh Button
                    dbc.Row([
                        dbc.Col(
                            dbc.Button("Refresh Assumptions", id="refresh-button", color="primary", className="mt-2"),
                            width=12,
                            className="text-center"
                        ),
                    ], className="mb-3"),
                ])
            ], className="mb-4"),
        ], width=12),
    ]),

    # Break-Even Analysis Section
    dbc.Row([
        dbc.Col([
            html.H3("BEA Analysis - West Jordan Clinic", className="mb-3"),
            dcc.Graph(id='break-even-graph'),
            dbc.Card([
                dbc.CardHeader(html.H5("Break-Even Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("BEP (Visits):"), width=3),
                        dbc.Col(id='bea-bep-visits', width=3),
                        dbc.Col(html.Strong("BEP ($):"), width=3),
                        dbc.Col(id='bea-bep-dollars', width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("Time to BEP:"), width=3),
                        dbc.Col(id='bea-time-to-bep', width=3),
                    ])
                ])
            ], className="mb-4"),
        ], width=12),
    ]),

    # Individual Forecast Graphs with Metrics
    dbc.Row([
        # Payments Forecast - West Jordan Clinic
        dbc.Col([
            html.H3("Payments Forecast (West Jordan Clinic)", className="mb-3"),
            dcc.Graph(id='payments-forecast-wj'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("RMSE:"), width=3),
                        dbc.Col(id='payments-rmse-wj', width=3),
                        dbc.Col(html.Strong("MAPE:"), width=3),
                        dbc.Col(id='payments-mape-wj', width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=3),
                        dbc.Col(id='payments-cagr-wj', width=3),
                    ], className="mb-2"),
                    # Average Payment per Visitor Metric
                    dbc.Row([
                        dbc.Col(html.Strong("Avg Payment per Visitor ($):"), width=3),
                        dbc.Col(id='payments-avg-payment-wj', width=3),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),

        # Visits Forecast - West Jordan Clinic
        dbc.Col([
            html.H3("Visits Forecast (West Jordan Clinic)", className="mb-3"),
            dcc.Graph(id='visits-forecast-wj'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("RMSE:"), width=3),
                        dbc.Col(id='visits-rmse-wj', width=3),
                        dbc.Col(html.Strong("MAPE:"), width=3),
                        dbc.Col(id='visits-mape-wj', width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=3),
                        dbc.Col(id='visits-cagr-wj', width=3),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),
    ]),

    dbc.Row([
        # Payments Forecast - Entire Payments Data Set
        dbc.Col([
            html.H3("Payments Forecast (Entire Data Set)", className="mb-3"),
            dcc.Graph(id='payments-forecast-entire'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("RMSE:"), width=3),
                        dbc.Col(id='payments-rmse-entire', width=3),
                        dbc.Col(html.Strong("MAPE:"), width=3),
                        dbc.Col(id='payments-mape-entire', width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=3),
                        dbc.Col(id='payments-cagr-entire', width=3),
                    ], className="mb-2"),
                    # Average Payment per Visitor Metric
                    dbc.Row([
                        dbc.Col(html.Strong("Avg Payment per Visitor ($):"), width=3),
                        dbc.Col(id='payments-avg-payment-entire', width=3),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),

        # Visits Forecast - Entire Payments Data Set
        dbc.Col([
            html.H3("Visits Forecast (Entire Data Set)", className="mb-3"),
            dcc.Graph(id='visits-forecast-entire'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("RMSE:"), width=3),
                        dbc.Col(id='visits-rmse-entire', width=3),
                        dbc.Col(html.Strong("MAPE:"), width=3),
                        dbc.Col(id='visits-mape-entire', width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=3),
                        dbc.Col(id='visits-cagr-entire', width=3),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),
    ]),

    # Download Buttons
    dbc.Row([
        dbc.Col([
            dbc.Button("Download Payments Forecast (West Jordan)", id="download-payments-wj-button", color="primary",
                       className="mr-2"),
            dcc.Download(id="download-payments-wj-data"),
        ], width=6),
        dbc.Col([
            dbc.Button("Download Visits Forecast (West Jordan)", id="download-visits-wj-button", color="primary",
                       className="mr-2"),
            dcc.Download(id="download-visits-wj-data"),
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Button("Download Payments Forecast (Entire Data Set)", id="download-payments-entire-button",
                       color="primary", className="mr-2"),
            dcc.Download(id="download-payments-entire-data"),
        ], width=6),
        dbc.Col([
            dbc.Button("Download Visits Forecast (Entire Data Set)", id="download-visits-entire-button",
                       color="primary", className="mr-2"),
            dcc.Download(id="download-visits-entire-data"),
        ], width=6),
    ], className="mb-4"),

    # Help Modal
    dbc.Row([
        dbc.Col([
            dbc.Button("Help", id="open-help", n_clicks=0, className="mb-4"),
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Help")),
                dbc.ModalBody(
                    "This dashboard provides forecasts and break-even analysis for the West Jordan Clinic and the entire Payments data set. "
                    "Adjust the 'Average Cost Per Visitor', 'Average Payment Per Visitor', set the 'Initial Visitors', input the 'Start-Up Costs', and specify the 'Visitor CAGR Growth Rate' to see how they affect the financial projections. "
                    "Click the 'Refresh Assumptions' button to apply your changes."
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-help", className="ml-auto")
                ),
            ], id="help-modal", is_open=False),
        ], width=12),
    ]),

], fluid=True)

# Define callbacks to update graphs and metrics
@app.callback(
    [
        Output('payments-forecast-wj', 'figure'),
        Output('visits-forecast-wj', 'figure'),
        Output('payments-rmse-wj', 'children'),
        Output('payments-mape-wj', 'children'),
        Output('payments-cagr-wj', 'children'),
        Output('payments-avg-payment-wj', 'children'),  # New Output
        Output('visits-rmse-wj', 'children'),
        Output('visits-mape-wj', 'children'),
        Output('visits-cagr-wj', 'children'),
        Output('break-even-graph', 'figure'),
        Output('bea-bep-visits', 'children'),
        Output('bea-bep-dollars', 'children'),
        Output('bea-time-to-bep', 'children'),
        Output('payments-forecast-entire', 'figure'),
        Output('visits-forecast-entire', 'figure'),
        Output('payments-rmse-entire', 'children'),
        Output('payments-mape-entire', 'children'),
        Output('payments-cagr-entire', 'children'),
        Output('payments-avg-payment-entire', 'children'),  # New Output
        Output('visits-rmse-entire', 'children'),
        Output('visits-mape-entire', 'children'),
        Output('visits-cagr-entire', 'children'),
        Output('error-message', 'children'),  # Error message
        Output('error-message', 'is_open'),  # Alert visibility
    ],
    [
        Input('refresh-button', 'n_clicks'),  # Input for Refresh Button
    ],
    [
        State('cost-per-visit', 'value'),
        State('avg-payment-per-visitor', 'value'),  # New State
        State('initial-visitors', 'value'),
        State('startup-costs', 'value'),
        State('visitor-cagr', 'value'),  # New State for Visitor CAGR
    ]
)
def update_graphs_and_metrics(n_clicks, cost_per_visit, avg_payment_per_visitor, initial_visitors, startup_costs, visitor_cagr):
    """
    Callback function to update all graphs and metrics based on user inputs when the refresh button is clicked.

    Parameters:
    - n_clicks (int): Number of times the refresh button has been clicked.
    - cost_per_visit (float): Average cost per visitor input by the user.
    - avg_payment_per_visitor (float): Average payment per visitor input by the user.
    - initial_visitors (int): Initial visitors assumption input by the user.
    - startup_costs (float): Start-Up costs input by the user.
    - visitor_cagr (float): Visitor CAGR Growth Rate input by the user.

    Returns:
    - Tuple containing updated figures and metrics for all outputs.
    """
    try:
        # Validate and sanitize inputs
        cost_per_visit = float(cost_per_visit) if cost_per_visit is not None else 0
        avg_payment_per_visitor = float(avg_payment_per_visitor) if avg_payment_per_visitor is not None else 0
        initial_visitors = int(initial_visitors) if initial_visitors is not None else 0
        startup_costs = float(startup_costs) if startup_costs is not None else 0
        visitor_cagr = float(visitor_cagr) if visitor_cagr is not None else 0  # Validate Visitor CAGR

        # Input Validation
        if cost_per_visit < 0:
            cost_per_visit = 0
            logger.warning("Negative input for 'Average Cost Per Visitor'. Defaulting to $0.")

        if avg_payment_per_visitor < 0:
            avg_payment_per_visitor = 0
            logger.warning("Negative input for 'Average Payment Per Visitor'. Defaulting to $0.")

        if initial_visitors < 0:
            initial_visitors = 0
            logger.warning("Negative input for 'Initial Visitors'. Defaulting to 0.")

        if startup_costs < 0:
            startup_costs = 0
            logger.warning("Negative input for 'Start-Up Costs'. Defaulting to $0.")

        # Validate Visitor CAGR Input
        if visitor_cagr < -100 or visitor_cagr > 100:
            raise ValueError("Visitor CAGR Growth Rate must be between -100% and 100%.")

        # Update Payments Forecast Graph for West Jordan Clinic
        payments_fig_wj = create_prophet_forecast_graph(
            actual_df=west_jordan_payments,
            forecast_df=payments_forecast_wj,
            title="Payments Forecast for West Jordan Clinic",
            y_title="Payments ($)",
            actual_name="Actual Payments",
            forecast_name="Forecasted Payments",
            rmse=payments_metrics_wj['RMSE'],
            mape=payments_metrics_wj['MAPE'],
            cagr=None  # CAGR will be updated below
        )

        # Calculate Average Payment per Visitor for West Jordan Clinic
        merged_wj = west_jordan_payments.merge(west_jordan_visits, on='ds', suffixes=('_payments', '_visits'))
        merged_wj['avg_payment_per_visitor'] = merged_wj['y_payments'] / merged_wj['y_visits']
        average_payment_wj = merged_wj['avg_payment_per_visitor'].mean()
        logger.info(f"Calculated Average Payment per Visitor for West Jordan Clinic: ${average_payment_wj:.2f}")

        # Update Visits Forecast Graph for West Jordan Clinic
        visits_fig_wj = create_prophet_forecast_graph(
            actual_df=west_jordan_visits,
            forecast_df=visits_forecast_wj,
            title="Visits Forecast for West Jordan Clinic",
            y_title="Visits",
            actual_name="Actual Visits",
            forecast_name="Forecasted Visits",
            rmse=visits_metrics_wj['RMSE'],
            mape=visits_metrics_wj['MAPE'],
            cagr=None  # CAGR will be updated below
        )

        # Calculate Average Payment per Visitor for Entire Data Set
        merged_entire = entire_payments.merge(entire_visits, on='ds', suffixes=('_payments', '_visits'))
        merged_entire['avg_payment_per_visitor'] = merged_entire['y_payments'] / merged_entire['y_visits']
        average_payment_entire = merged_entire['avg_payment_per_visitor'].mean()
        logger.info(f"Calculated Average Payment per Visitor for Entire Data Set: ${average_payment_entire:.2f}")

        # ---- BEA Analysis ----

        # Identify the last date of actual payments data
        last_actual_date = west_jordan_payments['ds'].max()
        logger.info(f"Last actual payment date: {last_actual_date}")

        # Separate forecasted data (dates after last_actual_date)
        future_payments_forecast_wj = payments_forecast_wj[payments_forecast_wj['ds'] > last_actual_date].reset_index(drop=True)
        future_visits_forecast_wj = visits_forecast_wj[visits_forecast_wj['ds'] > last_actual_date].reset_index(drop=True)

        logger.info(f"Number of forecasted months for BEA: {len(future_visits_forecast_wj)}")

        # Check if there is any forecasted data
        if future_visits_forecast_wj.empty:
            raise ValueError("No forecasted data available for BEA.")

        # Use User-Provided Visitor CAGR for BEA Analysis
        # Convert CAGR to a monthly growth rate for visits
        monthly_growth_rate_visits = (1 + visitor_cagr / 100) ** (1/12) - 1
        logger.info(f"Converted Visitor CAGR to monthly growth rate for visits: {monthly_growth_rate_visits:.4f}")

        # Initialize projected visitors list with initial_visitors
        projected_visits_wj = [initial_visitors]

        # Apply monthly growth rate to project future visitors
        for month in range(len(future_visits_forecast_wj)):
            next_visits = projected_visits_wj[-1] * (1 + monthly_growth_rate_visits)
            projected_visits_wj.append(next_visits)

        # Remove the last element to match the forecast period
        projected_visits_wj = projected_visits_wj[:-1]

        # Create a pandas Series for projected_visits_wj with the correct dates
        projected_visits_wj = pd.Series(projected_visits_wj, index=future_visits_forecast_wj['ds'])

        logger.info(f"Projected Visits for BEA (first 5 months):\n{projected_visits_wj.head()}")

        # Calculate Revenue and Variable Costs based on projected visitors
        revenue_wj_future = avg_payment_per_visitor * projected_visits_wj
        variable_costs_wj_future = cost_per_visit * projected_visits_wj
        net_profit_wj_future = revenue_wj_future - variable_costs_wj_future

        # **Incorporate Start-Up Costs Assumption by Adding to Variable Costs in the First Month**
        if startup_costs > 0 and not variable_costs_wj_future.empty:
            variable_costs_wj_future.iloc[0] += startup_costs  # Add Start-Up Costs to Variable Costs in the first forecasted month
            logger.info(f"Incorporated start-up costs assumption: ${startup_costs:,.2f} added to Variable Costs in the first forecasted month.")

        # Recalculate Net Profit after adding Start-Up Costs to Variable Costs
        net_profit_wj_future = revenue_wj_future - variable_costs_wj_future

        # Determine Break-Even Point (First month where Net Profit >= 0)
        bep_indices_wj = net_profit_wj_future[net_profit_wj_future >= 0].index
        if not bep_indices_wj.empty:
            bep_date_wj = bep_indices_wj[0]
            try:
                # Get the positional index of the BEP date
                bep_pos_wj = net_profit_wj_future.index.get_loc(bep_date_wj)
                bep_month_wj = bep_pos_wj + 1  # +1 because index starts at 0
                bep_visits_wj = projected_visits_wj.iloc[bep_pos_wj]
                bep_dollars_wj = revenue_wj_future.iloc[bep_pos_wj]
                time_to_bep_wj = bep_month_wj  # in months
                logger.info(f"BEP achieved in month {bep_month_wj}: {time_to_bep_wj} months")
            except Exception as e:
                logger.error(f"Error determining BEP position: {e}")
                raise
        else:
            bep_date_wj = None
            bep_pos_wj = None
            bep_month_wj = None
            bep_visits_wj = None
            bep_dollars_wj = None
            time_to_bep_wj = None
            logger.info("BEP not achieved within forecast period")

        # Create Break-Even Analysis Graph with Revenue, Costs, Net Profit, and Visits
        bea_fig = go.Figure()

        # Add Total Revenue
        bea_fig.add_trace(go.Scatter(
            x=future_payments_forecast_wj['ds'],
            y=revenue_wj_future,
            mode='lines',
            name='Total Revenue',
            line=dict(color='green')
        ))

        # Add Total Variable Costs
        bea_fig.add_trace(go.Scatter(
            x=future_payments_forecast_wj['ds'],
            y=variable_costs_wj_future,
            mode='lines',
            name='Total Variable Costs',
            line=dict(color='red')
        ))

        # Add Net Profit
        bea_fig.add_trace(go.Scatter(
            x=future_payments_forecast_wj['ds'],
            y=net_profit_wj_future,
            mode='lines+markers',
            name='Net Profit',
            line=dict(color='orange'),
            marker=dict(size=5)
        ))

        # Add Projected Visits on Secondary Y-Axis
        bea_fig.add_trace(go.Scatter(
            x=future_visits_forecast_wj['ds'],
            y=projected_visits_wj,
            mode='lines',
            name='Projected Visits',
            line=dict(color='blue', dash='dot'),
            yaxis='y2'
        ))

        # Add Break-Even Point Indicator if achieved
        if bep_month_wj and bep_date_wj is not None and not pd.isna(bep_date_wj):
            bea_fig.add_vline(x=bep_date_wj, line_width=2, line_dash="dash", line_color="purple")
            bea_fig.add_annotation(
                x=bep_date_wj,
                y=net_profit_wj_future.iloc[bep_pos_wj],
                text=f"BEP Achieved: {int(bep_visits_wj)} visits<br>${bep_dollars_wj:,.2f}",
                showarrow=True,
                arrowhead=1
            )

        # Update layout with secondary y-axis
        bea_fig.update_layout(
            title="BEA Analysis",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            yaxis2=dict(
                title="Visits",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            template='plotly_white',
            height=600,
            margin=dict(l=60, r=100, t=100, b=60),  # Adjust right margin for yaxis2
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            xaxis=dict(range=[future_payments_forecast_wj['ds'].min().date(), future_payments_forecast_wj['ds'].max().date()])
        )

        # Update Payments Forecast Graph for Entire Payments Data Set
        payments_fig_entire = create_prophet_forecast_graph(
            actual_df=entire_payments,
            forecast_df=payments_forecast_entire,
            title="Payments Forecast for Entire Data Set",
            y_title="Payments ($)",
            actual_name="Actual Payments",
            forecast_name="Forecasted Payments",
            rmse=payments_metrics_entire['RMSE'],
            mape=payments_metrics_entire['MAPE'],
            cagr=None  # CAGR will be updated below
        )

        # Update Visits Forecast Graph for Entire Payments Data Set
        visits_fig_entire = create_prophet_forecast_graph(
            actual_df=entire_visits,
            forecast_df=visits_forecast_entire,
            title="Visits Forecast for Entire Data Set",
            y_title="Visits",
            actual_name="Actual Visits",
            forecast_name="Forecasted Visits",
            rmse=visits_metrics_entire['RMSE'],
            mape=visits_metrics_entire['MAPE'],
            cagr=None  # CAGR will be updated below
        )

        # **Calculate CAGR for West Jordan Clinic (Historical)**
        if payments_cagr_wj is None:
            payments_cagr_wj_text = "N/A"
        else:
            payments_cagr_wj_text = f"{payments_cagr_wj:.2f}%"

        if visits_cagr_wj is None:
            visits_cagr_wj_text = "N/A"
        else:
            visits_cagr_wj_text = f"{visits_cagr_wj:.2f}%"

        # Calculate CAGR for Entire Payments Data Set
        if payments_cagr_entire is None:
            payments_cagr_entire_text = "N/A"
        else:
            payments_cagr_entire_text = f"{payments_cagr_entire:.2f}%"

        if visits_cagr_entire is None:
            visits_cagr_entire_text = "N/A"
        else:
            visits_cagr_entire_text = f"{visits_cagr_entire:.2f}%"

        # **Update graphs with CAGR**
        payments_fig_wj = create_prophet_forecast_graph(
            actual_df=west_jordan_payments,
            forecast_df=payments_forecast_wj,
            title="Payments Forecast for West Jordan Clinic",
            y_title="Payments ($)",
            actual_name="Actual Payments",
            forecast_name="Forecasted Payments",
            rmse=payments_metrics_wj['RMSE'],
            mape=payments_metrics_wj['MAPE'],
            cagr=payments_cagr_wj if payments_cagr_wj is not None else 0  # Handle None
        )

        visits_fig_wj = create_prophet_forecast_graph(
            actual_df=west_jordan_visits,
            forecast_df=visits_forecast_wj,
            title="Visits Forecast for West Jordan Clinic",
            y_title="Visits",
            actual_name="Actual Visits",
            forecast_name="Forecasted Visits",
            rmse=visits_metrics_wj['RMSE'],
            mape=visits_metrics_wj['MAPE'],
            cagr=visitor_cagr  # Use User-Defined CAGR
        )

        payments_fig_entire = create_prophet_forecast_graph(
            actual_df=entire_payments,
            forecast_df=payments_forecast_entire,
            title="Payments Forecast for Entire Data Set",
            y_title="Payments ($)",
            actual_name="Actual Payments",
            forecast_name="Forecasted Payments",
            rmse=payments_metrics_entire['RMSE'],
            mape=payments_metrics_entire['MAPE'],
            cagr=payments_cagr_entire if payments_cagr_entire is not None else 0  # Handle None
        )

        visits_fig_entire = create_prophet_forecast_graph(
            actual_df=entire_visits,
            forecast_df=visits_forecast_entire,
            title="Visits Forecast for Entire Data Set",
            y_title="Visits",
            actual_name="Actual Visits",
            forecast_name="Forecasted Visits",
            rmse=visits_metrics_entire['RMSE'],
            mape=visits_metrics_entire['MAPE'],
            cagr=visits_cagr_entire if visits_cagr_entire is not None else 0  # Handle None
        )

        # Return all updated figures and metrics with no error
        return (
            payments_fig_wj,  # payments-forecast-wj.figure
            visits_fig_wj,    # visits-forecast-wj.figure
            f"{payments_metrics_wj['RMSE']:.2f}",  # payments-rmse-wj.children
            f"{payments_metrics_wj['MAPE']:.2f}%",  # payments-mape-wj.children
            f"{visitor_cagr:.2f}%",          # payments-cagr-wj.children (Updated to User CAGR)
            f"${average_payment_wj:.2f}",        # payments-avg-payment-wj.children (New)
            f"{visits_metrics_wj['RMSE']:.2f}",    # visits-rmse-wj.children
            f"{visits_metrics_wj['MAPE']:.2f}%",   # visits-mape-wj.children
            f"{visitor_cagr:.2f}%",             # visits-cagr-wj.children (Updated to User CAGR)
            bea_fig,                              # break-even-graph.figure
            f"{bep_visits_wj:.0f}" if isinstance(bep_visits_wj, float) else bep_visits_wj,  # bea-bep-visits.children
            f"${bep_dollars_wj:,.2f}" if isinstance(bep_dollars_wj, float) else bep_dollars_wj,  # bea-bep-dollars.children
            f"{time_to_bep_wj} months" if isinstance(time_to_bep_wj, int) else "N/A",  # bea-time-to-bep.children
            payments_fig_entire,  # payments-forecast-entire.figure
            visits_fig_entire,     # visits-forecast-entire.figure
            f"{payments_metrics_entire['RMSE']:.2f}",  # payments-rmse-entire.children
            f"{payments_metrics_entire['MAPE']:.2f}%",  # payments-mape-entire.children
            payments_cagr_entire_text,          # payments-cagr-entire.children
            f"${average_payment_entire:.2f}",        # payments-avg-payment-entire.children (New)
            f"{visits_metrics_entire['RMSE']:.2f}",    # visits-rmse-entire.children
            f"{visits_metrics_entire['MAPE']:.2f}%",   # visits-mape-entire.children
            visits_cagr_entire_text,             # visits-cagr-entire.children
            "",  # error-message.children (No error message)
            False,  # error-message.is_open (Alert is closed)
        )
    except KeyError as e:
        logger.error(f"KeyError in callback: {e}")
        # Return empty figures and default values in case of error
        empty_fig = go.Figure()
        return (
            empty_fig,  # payments-forecast-wj.figure
            empty_fig,  # visits-forecast-wj.figure
            "", "", "", "", "", "", "",
            empty_fig, "", "", "",
            empty_fig, empty_fig, "", "", "", "",
            "", "", "", "",
            f"An error occurred: {e}", True
        )
    except Exception as e:
        logger.error(f"Error in callback: {e}")
        # Return empty figures and default values in case of error
        empty_fig = go.Figure()
        return (
            empty_fig,  # payments-forecast-wj.figure
            empty_fig,  # visits-forecast-wj.figure
            "", "", "", "", "", "", "",
            empty_fig, "", "", "",
            empty_fig, empty_fig, "", "", "", "",
            "", "", "", "",
            f"An error occurred: {e}", True
        )

# Define callbacks for Download Buttons
@app.callback(
    Output("download-payments-wj-data", "data"),
    Input("download-payments-wj-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_payments_wj(n_clicks):
    """
    Callback to download the Payments Forecast data for West Jordan Clinic as a CSV file.

    Parameters:
    - n_clicks (int): Number of times the download button has been clicked.

    Returns:
    - CSV file containing the Payments Forecast data for West Jordan Clinic.
    """
    return dcc.send_data_frame(payments_forecast_wj.to_csv, "payments_forecast_wj.csv")

@app.callback(
    Output("download-visits-wj-data", "data"),
    Input("download-visits-wj-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_visits_wj(n_clicks):
    """
    Callback to download the Visits Forecast data for West Jordan Clinic as a CSV file.

    Parameters:
    - n_clicks (int): Number of times the download button has been clicked.

    Returns:
    - CSV file containing the Visits Forecast data for West Jordan Clinic.
    """
    return dcc.send_data_frame(visits_forecast_wj.to_csv, "visits_forecast_wj.csv")

@app.callback(
    Output("download-payments-entire-data", "data"),
    Input("download-payments-entire-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_payments_entire(n_clicks):
    """
    Callback to download the Payments Forecast data for the Entire Data Set as a CSV file.

    Parameters:
    - n_clicks (int): Number of times the download button has been clicked.

    Returns:
    - CSV file containing the Payments Forecast data for the Entire Data Set.
    """
    return dcc.send_data_frame(payments_forecast_entire.to_csv, "payments_forecast_entire.csv")

@app.callback(
    Output("download-visits-entire-data", "data"),
    Input("download-visits-entire-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_visits_entire(n_clicks):
    """
    Callback to download the Visits Forecast data for the Entire Data Set as a CSV file.

    Parameters:
    - n_clicks (int): Number of times the download button has been clicked.

    Returns:
    - CSV file containing the Visits Forecast data for the Entire Data Set.
    """
    return dcc.send_data_frame(visits_forecast_entire.to_csv, "visits_forecast_entire.csv")

# Define callbacks for Help Modal
@app.callback(
    Output("help-modal", "is_open"),
    [Input("open-help", "n_clicks"), Input("close-help", "n_clicks")],
    [State("help-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    """
    Callback to toggle the visibility of the Help Modal.

    Parameters:
    - n1 (int): Number of clicks on the "Help" button.
    - n2 (int): Number of clicks on the "Close" button within the modal.
    - is_open (bool): Current state of the modal's visibility.

    Returns:
    - bool: Updated state of the modal's visibility.
    """
    if n1 or n2:
        return not is_open
    return is_open

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
