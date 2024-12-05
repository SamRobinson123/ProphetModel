# Import necessary libraries
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to create a Plotly graph for Prophet forecast with actual and forecasted data
def create_prophet_forecast_graph(actual_df, forecast_df, title, y_title, actual_name, forecast_name, rmse=None,
                                  mape=None, cagr=None):
    fig = go.Figure()

    # Add actual data
    fig.add_trace(go.Scatter(
        x=actual_df['ds'],
        y=actual_df['y'],
        mode='lines+markers',
        name=actual_name,
        line=dict(color='blue', width=2),
        marker=dict(size=5)
    ))

    # Add forecasted data
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name=forecast_name,
        line=dict(color='green', width=3)
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        name='Upper Confidence Interval',
        line=dict(color='lightgreen', dash='dash'),
        showlegend=False
    ))
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
    title_with_metrics = f"{title}<br>{metrics_text}" if metrics_text else title

    # Update the layout
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
    if 'EncDate' not in df.columns or 'NetPayment' not in df.columns:
        logger.error("Missing required columns in payments DataFrame.")
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
    logger.info(f"Records after removing non-positive payments: {len(df)}")

    # Aggregate monthly
    df = df.groupby(pd.Grouper(key='ds', freq='M')).sum().reset_index()

    # Downcast data types
    df['y'] = pd.to_numeric(df['y'], downcast='float')

    return df

# Function to preprocess visits data
def preprocess_visits(df):
    if 'EncDate' not in df.columns:
        logger.error("Missing 'EncDate' column in visits DataFrame.")
        raise ValueError("Missing 'EncDate' column in visits DataFrame.")

    # Rename and ensure datetime
    df = df.rename(columns={'EncDate': 'ds'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds'])

    # Count visits monthly
    df = df.groupby(pd.Grouper(key='ds', freq='M')).size().reset_index(name='y')

    # Remove negative visit counts
    df = df[df['y'] >= 0]
    logger.info(f"Records after ensuring non-negative visits: {len(df)}")

    # Downcast data types
    df['y'] = pd.to_numeric(df['y'], downcast='integer')

    return df

# Function to calculate CAGR
def calculate_cagr(start_value, end_value, periods_in_years):
    if start_value <= 0:
        return None
    cagr = ((end_value / start_value) ** (1 / periods_in_years) - 1) * 100
    return cagr

# Function to validate DataFrame
def validate_dataframe(df, name):
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

# Read and preprocess the data
try:
    payments_df = pd.read_excel("payment.xlsx")
    payments_df = payments_df[payments_df['EncStatus'] == 'CHK']
    logger.info("Successfully read 'payment.xlsx' and filtered by EncStatus 'CHK'.")
except FileNotFoundError:
    logger.error("The file 'payment.xlsx' was not found.")
    raise
except Exception as e:
    logger.error(f"Error reading 'payment.xlsx': {e}")
    raise

# Filter data for UPFH Family Clinic - West Jordan
if 'Facility' not in payments_df.columns:
    logger.error("Missing 'Facility' column in payments DataFrame.")
    raise ValueError("Missing 'Facility' column in payments DataFrame.")

west_jordan_df = payments_df[payments_df['Facility'] == 'UPFH Family Clinic - West Jordan']
logger.info(f"Records for 'UPFH Family Clinic - West Jordan': {len(west_jordan_df)}")

if west_jordan_df.empty:
    logger.error("No data found for 'UPFH Family Clinic - West Jordan'.")
    raise ValueError("No data for 'UPFH Family Clinic - West Jordan'.")

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

# Load precomputed forecasts
try:
    payments_forecast_wj = pd.read_csv('payments_forecast_wj.csv', parse_dates=['ds'])
    visits_forecast_wj = pd.read_csv('visits_forecast_wj.csv', parse_dates=['ds'])
    payments_forecast_entire = pd.read_csv('payments_forecast_entire.csv', parse_dates=['ds'])
    visits_forecast_entire = pd.read_csv('visits_forecast_entire.csv', parse_dates=['ds'])
    logger.info("Loaded precomputed forecasts.")
except FileNotFoundError:
    logger.error("Precomputed forecast files not found.")
    raise

# Calculate metrics (Precomputed or set to None)
payments_metrics_wj = {'RMSE': None, 'MAPE': None}
visits_metrics_wj = {'RMSE': None, 'MAPE': None}
payments_metrics_entire = {'RMSE': None, 'MAPE': None}
visits_metrics_entire = {'RMSE': None, 'MAPE': None}

# Calculate Historical CAGR
payments_cagr_wj = calculate_cagr(
    start_value=west_jordan_payments['y'].iloc[0],
    end_value=west_jordan_payments['y'].iloc[-1],
    periods_in_years=5  # Adjust as per your data
)

visits_cagr_wj = calculate_cagr(
    start_value=west_jordan_visits['y'].iloc[0],
    end_value=west_jordan_visits['y'].iloc[-1],
    periods_in_years=5
)

initial_visitor_cagr = visits_cagr_wj if visits_cagr_wj is not None else 0
if visits_cagr_wj is None:
    logger.warning("Historical Visitor CAGR is undefined. Defaulting to 0%.")
else:
    logger.info(f"Historical Visitor CAGR (West Jordan Clinic): {initial_visitor_cagr:.2f}%")

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

initial_visits_cagr_entire = visits_cagr_entire if visits_cagr_entire is not None else 0
if visits_cagr_entire is None:
    logger.warning("Historical Visitor CAGR for Entire Data Set is undefined. Defaulting to 0%.")
else:
    logger.info(f"Historical Visitor CAGR (Entire Data Set): {initial_visits_cagr_entire:.2f}%")

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# Precompute forecast graphs
payments_fig_wj = create_prophet_forecast_graph(
    actual_df=west_jordan_payments,
    forecast_df=payments_forecast_wj,
    title="Payments Forecast for West Jordan Clinic",
    y_title="Payments ($)",
    actual_name="Actual Payments",
    forecast_name="Forecasted Payments",
    cagr=payments_cagr_wj if payments_cagr_wj is not None else 0
)

visits_fig_wj = create_prophet_forecast_graph(
    actual_df=west_jordan_visits,
    forecast_df=visits_forecast_wj,
    title="Visits Forecast for West Jordan Clinic",
    y_title="Visits",
    actual_name="Actual Visits",
    forecast_name="Forecasted Visits",
    cagr=initial_visitor_cagr
)

payments_fig_entire = create_prophet_forecast_graph(
    actual_df=entire_payments,
    forecast_df=payments_forecast_entire,
    title="Payments Forecast for Entire Data Set",
    y_title="Payments ($)",
    actual_name="Actual Payments",
    forecast_name="Forecasted Payments",
    cagr=payments_cagr_entire if payments_cagr_entire is not None else 0
)

visits_fig_entire = create_prophet_forecast_graph(
    actual_df=entire_visits,
    forecast_df=visits_forecast_entire,
    title="Visits Forecast for Entire Data Set",
    y_title="Visits",
    actual_name="Actual Visits",
    forecast_name="Forecasted Visits",
    cagr=visits_cagr_entire if visits_cagr_entire is not None else 0
)

# Calculate Average Payment per Visitor (Precomputed)
merged_wj = west_jordan_payments.merge(west_jordan_visits, on='ds', suffixes=('_payments', '_visits'))
merged_wj['avg_payment_per_visitor'] = merged_wj['y_payments'] / merged_wj['y_visits']
average_payment_wj = merged_wj['avg_payment_per_visitor'].mean()

merged_entire = entire_payments.merge(entire_visits, on='ds', suffixes=('_payments', '_visits'))
merged_entire['avg_payment_per_visitor'] = merged_entire['y_payments'] / merged_entire['y_visits']
average_payment_entire = merged_entire['avg_payment_per_visitor'].mean()

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
                                value=500,
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
                                value=264,
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
                                value=10000,
                                min=0,
                                step=100,
                                style={'width': '100%'}
                            ),
                            width=6,
                        ),
                    ], className="mb-3"),
                    # Visitor CAGR Growth Rate Assumption Input
                    dbc.Row([
                        dbc.Col(html.Label("Visitor CAGR Growth Rate (%):"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='visitor-cagr',
                                type='number',
                                value=initial_visitor_cagr,
                                min=-100,
                                max=100,
                                step=0.1,
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
            dcc.Graph(figure=payments_fig_wj, id='payments-forecast-wj'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    # Average Payment per Visitor Metric
                    dbc.Row([
                        dbc.Col(html.Strong("Avg Payment per Visitor ($):"), width=6),
                        dbc.Col(f"${average_payment_wj:.2f}", width=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=6),
                        dbc.Col(f"{payments_cagr_wj:.2f}%" if payments_cagr_wj else "N/A", width=6),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),

        # Visits Forecast - West Jordan Clinic
        dbc.Col([
            html.H3("Visits Forecast (West Jordan Clinic)", className="mb-3"),
            dcc.Graph(figure=visits_fig_wj, id='visits-forecast-wj'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=6),
                        dbc.Col(f"{initial_visitor_cagr:.2f}%", width=6),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),
    ]),

    dbc.Row([
        # Payments Forecast - Entire Payments Data Set
        dbc.Col([
            html.H3("Payments Forecast (Entire Data Set)", className="mb-3"),
            dcc.Graph(figure=payments_fig_entire, id='payments-forecast-entire'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    # Average Payment per Visitor Metric
                    dbc.Row([
                        dbc.Col(html.Strong("Avg Payment per Visitor ($):"), width=6),
                        dbc.Col(f"${average_payment_entire:.2f}", width=6),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=6),
                        dbc.Col(f"{payments_cagr_entire:.2f}%" if payments_cagr_entire else "N/A", width=6),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),

        # Visits Forecast - Entire Payments Data Set
        dbc.Col([
            html.H3("Visits Forecast (Entire Data Set)", className="mb-3"),
            dcc.Graph(figure=visits_fig_entire, id='visits-forecast-entire'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=6),
                        dbc.Col(f"{visits_cagr_entire:.2f}%" if visits_cagr_entire else "N/A", width=6),
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

# Callback functions remain unchanged

# Run the server
if __name__ == "__main__":
    import os
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
