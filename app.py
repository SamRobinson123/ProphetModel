# app.py

# Import necessary libraries
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import logging
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to create a Plotly graph for Prophet forecast
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

# Function to calculate CAGR
def calculate_cagr(start_value, end_value, periods_in_years):
    if start_value <= 0 or end_value <= 0 or periods_in_years <= 0:
        return None
    cagr = ((end_value / start_value) ** (1 / periods_in_years) - 1) * 100
    return cagr

# Load precomputed forecasts
try:
    payments_forecast_wj = pd.read_csv(os.path.join(BASE_DIR, 'payments_forecast_wj.csv'), parse_dates=['ds'])
    visits_forecast_wj = pd.read_csv(os.path.join(BASE_DIR, 'visits_forecast_wj.csv'), parse_dates=['ds'])
    payments_forecast_entire = pd.read_csv(os.path.join(BASE_DIR, 'payments_forecast_entire.csv'), parse_dates=['ds'])
    visits_forecast_entire = pd.read_csv(os.path.join(BASE_DIR, 'visits_forecast_entire.csv'), parse_dates=['ds'])
    logger.info("Loaded precomputed forecasts.")
except FileNotFoundError as e:
    logger.error(f"Error loading forecasts: {e}")
    raise

# Load metrics
payments_metrics_wj = pd.read_csv(os.path.join(BASE_DIR, 'payments_metrics_wj.csv')).to_dict('records')[0]
visits_metrics_wj = pd.read_csv(os.path.join(BASE_DIR, 'visits_metrics_wj.csv')).to_dict('records')[0]
payments_metrics_entire = pd.read_csv(os.path.join(BASE_DIR, 'payments_metrics_entire.csv')).to_dict('records')[0]
visits_metrics_entire = pd.read_csv(os.path.join(BASE_DIR, 'visits_metrics_entire.csv')).to_dict('records')[0]
logger.info("Loaded precomputed metrics.")

# Load actual data
west_jordan_payments = pd.read_csv(os.path.join(BASE_DIR, 'west_jordan_payments.csv'), parse_dates=['ds'])
west_jordan_visits = pd.read_csv(os.path.join(BASE_DIR, 'west_jordan_visits.csv'), parse_dates=['ds'])
entire_payments = pd.read_csv(os.path.join(BASE_DIR, 'entire_payments.csv'), parse_dates=['ds'])
entire_visits = pd.read_csv(os.path.join(BASE_DIR, 'entire_visits.csv'), parse_dates=['ds'])

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

# Calculate Average Payment per Visitor
merged_wj = west_jordan_payments.merge(west_jordan_visits, on='ds', suffixes=('_payments', '_visits'))
merged_wj['avg_payment_per_visitor'] = merged_wj['y_payments'] / merged_wj['y_visits']
average_payment_wj = merged_wj['avg_payment_per_visitor'].mean()

merged_entire = entire_payments.merge(entire_visits, on='ds', suffixes=('_payments', '_visits'))
merged_entire['avg_payment_per_visitor'] = merged_entire['y_payments'] / merged_entire['y_visits']
average_payment_entire = merged_entire['avg_payment_per_visitor'].mean()

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# Generate forecast graphs
payments_fig_wj = create_prophet_forecast_graph(
    actual_df=west_jordan_payments,
    forecast_df=payments_forecast_wj,
    title="Payments Forecast (West Jordan Clinic)",
    y_title="Payments ($)",
    actual_name="Actual Payments",
    forecast_name="Forecasted Payments",
    rmse=payments_metrics_wj['RMSE'],
    mape=payments_metrics_wj['MAPE'],
    cagr=payments_cagr_wj
)

visits_fig_wj = create_prophet_forecast_graph(
    actual_df=west_jordan_visits,
    forecast_df=visits_forecast_wj,
    title="Visits Forecast (West Jordan Clinic)",
    y_title="Visits",
    actual_name="Actual Visits",
    forecast_name="Forecasted Visits",
    rmse=visits_metrics_wj['RMSE'],
    mape=visits_metrics_wj['MAPE'],
    cagr=visits_cagr_wj
)

payments_fig_entire = create_prophet_forecast_graph(
    actual_df=entire_payments,
    forecast_df=payments_forecast_entire,
    title="Payments Forecast (Entire Data Set)",
    y_title="Payments ($)",
    actual_name="Actual Payments",
    forecast_name="Forecasted Payments",
    rmse=payments_metrics_entire['RMSE'],
    mape=payments_metrics_entire['MAPE'],
    cagr=payments_cagr_entire
)

visits_fig_entire = create_prophet_forecast_graph(
    actual_df=entire_visits,
    forecast_df=visits_forecast_entire,
    title="Visits Forecast (Entire Data Set)",
    y_title="Visits",
    actual_name="Actual Visits",
    forecast_name="Forecasted Visits",
    rmse=visits_metrics_entire['RMSE'],
    mape=visits_metrics_entire['MAPE'],
    cagr=visits_cagr_entire
)

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
                    # Average Cost Per Visitor Input
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
                    # Average Payment Per Visitor Input
                    dbc.Row([
                        dbc.Col(html.Label("Average Payment Per Visitor ($):"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='avg-payment-per-visitor',
                                type='number',
                                value=round(average_payment_wj, 2),
                                min=0,
                                step=10,
                                style={'width': '100%'}
                            ),
                            width=6,
                        ),
                    ], className="mb-3"),
                    # Initial Visitors Input
                    dbc.Row([
                        dbc.Col(html.Label("Initial Visitors:"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='initial-visitors',
                                type='number',
                                value=int(west_jordan_visits['y'].iloc[-1]),
                                min=0,
                                step=1,
                                style={'width': '100%'}
                            ),
                            width=6,
                        ),
                    ], className="mb-3"),
                    # Start-Up Costs Input
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
                    # Visitor CAGR Growth Rate Input
                    dbc.Row([
                        dbc.Col(html.Label("Visitor CAGR Growth Rate (%):"), width=6),
                        dbc.Col(
                            dcc.Input(
                                id='visitor-cagr',
                                type='number',
                                value=round(visits_cagr_wj, 2) if visits_cagr_wj else 0,
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
            html.H3("Break-Even Analysis - West Jordan Clinic", className="mb-3"),
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

    # Forecast Graphs and Metrics
    dbc.Row([
        # Payments Forecast - West Jordan Clinic
        dbc.Col([
            html.H3("Payments Forecast (West Jordan Clinic)", className="mb-3"),
            dcc.Graph(figure=payments_fig_wj, id='payments-forecast-wj'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("RMSE:"), width=3),
                        dbc.Col(f"{payments_metrics_wj['RMSE']:.2f}", width=3),
                        dbc.Col(html.Strong("MAPE:"), width=3),
                        dbc.Col(f"{payments_metrics_wj['MAPE']:.2f}%", width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=3),
                        dbc.Col(f"{payments_cagr_wj:.2f}%", width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("Avg Payment per Visitor ($):"), width=3),
                        dbc.Col(f"${average_payment_wj:.2f}", width=3),
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
                        dbc.Col(html.Strong("RMSE:"), width=3),
                        dbc.Col(f"{visits_metrics_wj['RMSE']:.2f}", width=3),
                        dbc.Col(html.Strong("MAPE:"), width=3),
                        dbc.Col(f"{visits_metrics_wj['MAPE']:.2f}%", width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=3),
                        dbc.Col(f"{visits_cagr_wj:.2f}%", width=3),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),
    ]),

    # Forecast Graphs for Entire Data Set
    dbc.Row([
        # Payments Forecast - Entire Data Set
        dbc.Col([
            html.H3("Payments Forecast (Entire Data Set)", className="mb-3"),
            dcc.Graph(figure=payments_fig_entire, id='payments-forecast-entire'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("RMSE:"), width=3),
                        dbc.Col(f"{payments_metrics_entire['RMSE']:.2f}", width=3),
                        dbc.Col(html.Strong("MAPE:"), width=3),
                        dbc.Col(f"{payments_metrics_entire['MAPE']:.2f}%", width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=3),
                        dbc.Col(f"{payments_cagr_entire:.2f}%", width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("Avg Payment per Visitor ($):"), width=3),
                        dbc.Col(f"${average_payment_entire:.2f}", width=3),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),

        # Visits Forecast - Entire Data Set
        dbc.Col([
            html.H3("Visits Forecast (Entire Data Set)", className="mb-3"),
            dcc.Graph(figure=visits_fig_entire, id='visits-forecast-entire'),
            dbc.Card([
                dbc.CardHeader(html.H5("Metrics")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(html.Strong("RMSE:"), width=3),
                        dbc.Col(f"{visits_metrics_entire['RMSE']:.2f}", width=3),
                        dbc.Col(html.Strong("MAPE:"), width=3),
                        dbc.Col(f"{visits_metrics_entire['MAPE']:.2f}%", width=3),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col(html.Strong("CAGR:"), width=3),
                        dbc.Col(f"{visits_cagr_entire:.2f}%", width=3),
                    ], className="mb-2"),
                ])
            ], className="mb-4"),
        ], width=6),
    ]),

    # Download Buttons
    dbc.Row([
        dbc.Col([
            dbc.Button("Download Payments Forecast (West Jordan)", id="download-payments-wj-button", color="primary"),
            dcc.Download(id="download-payments-wj-data"),
        ], width=6),
        dbc.Col([
            dbc.Button("Download Visits Forecast (West Jordan)", id="download-visits-wj-button", color="primary"),
            dcc.Download(id="download-visits-wj-data"),
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Button("Download Payments Forecast (Entire Data Set)", id="download-payments-entire-button",
                       color="primary"),
            dcc.Download(id="download-payments-entire-data"),
        ], width=6),
        dbc.Col([
            dbc.Button("Download Visits Forecast (Entire Data Set)", id="download-visits-entire-button",
                       color="primary"),
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
                    "This dashboard provides forecasts and break-even analysis for the West Jordan Clinic and the entire payments data set. "
                    "Adjust the assumptions to see how they affect the financial projections. "
                    "Click the 'Refresh Assumptions' button to apply your changes."
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-help", className="ml-auto")
                ),
            ], id="help-modal", is_open=False),
        ], width=12),
    ]),

], fluid=True)

# Callback to update the break-even graph and metrics
@app.callback(
    Output('break-even-graph', 'figure'),
    Output('bea-bep-visits', 'children'),
    Output('bea-bep-dollars', 'children'),
    Output('bea-time-to-bep', 'children'),
    Input('refresh-button', 'n_clicks'),
    State('cost-per-visit', 'value'),
    State('avg-payment-per-visitor', 'value'),
    State('initial-visitors', 'value'),
    State('startup-costs', 'value'),
    State('visitor-cagr', 'value'),
)
def update_break_even_graph(n_clicks, cost_per_visit, avg_payment_per_visitor, initial_visitors, startup_costs, visitor_cagr):
    try:
        # Calculate monthly visitors over 60 months
        months = np.arange(1, 61)
        visitor_growth_rate = visitor_cagr / 100
        monthly_growth_rate = (1 + visitor_growth_rate) ** (1/12) - 1
        visitors = initial_visitors * (1 + monthly_growth_rate) ** months

        # Calculate revenue and costs
        revenue = visitors * avg_payment_per_visitor
        costs = visitors * cost_per_visit + startup_costs

        # Cumulative revenue and costs
        cumulative_revenue = np.cumsum(revenue)
        cumulative_costs = np.cumsum(costs)

        # Find break-even point
        profit = cumulative_revenue - cumulative_costs
        bep_index = np.argmax(profit >= 0)
        if profit[bep_index] < 0:
            time_to_bep = "Not reached within 60 months"
            bep_visits = "N/A"
            bep_dollars = "N/A"
        else:
            time_to_bep = f"{months[bep_index]} months"
            bep_visits = f"{int(visitors[bep_index]):,}"
            bep_dollars = f"${int(cumulative_revenue[bep_index]):,}"

        # Create the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_revenue,
            mode='lines',
            name='Cumulative Revenue',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_costs,
            mode='lines',
            name='Cumulative Costs',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Break-Even Analysis',
            xaxis_title='Months',
            yaxis_title='Amount ($)',
            template='plotly_white',
            height=600,
            margin=dict(l=60, r=60, t=50, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        return fig, bep_visits, bep_dollars, time_to_bep

    except Exception as e:
        logger.error(f"Error in update_break_even_graph callback: {e}")
        return no_update, no_update, no_update, no_update

# Callbacks for the download buttons
@app.callback(
    Output("download-payments-wj-data", "data"),
    Input("download-payments-wj-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_payments_wj(n_clicks):
    return dcc.send_data_frame(payments_forecast_wj.to_csv, "payments_forecast_wj.csv")

@app.callback(
    Output("download-visits-wj-data", "data"),
    Input("download-visits-wj-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_visits_wj(n_clicks):
    return dcc.send_data_frame(visits_forecast_wj.to_csv, "visits_forecast_wj.csv")

@app.callback(
    Output("download-payments-entire-data", "data"),
    Input("download-payments-entire-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_payments_entire(n_clicks):
    return dcc.send_data_frame(payments_forecast_entire.to_csv, "payments_forecast_entire.csv")

@app.callback(
    Output("download-visits-entire-data", "data"),
    Input("download-visits-entire-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_visits_entire(n_clicks):
    return dcc.send_data_frame(visits_forecast_entire.to_csv, "visits_forecast_entire.csv")

# Callback for Help modal
@app.callback(
    Output("help-modal", "is_open"),
    [Input("open-help", "n_clicks"), Input("close-help", "n_clicks")],
    [State("help-modal", "is_open")],
)
def toggle_help_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Remove or comment out the app.run_server() block when using Gunicorn
# if __name__ == "__main__":
#     app.run_server(debug=False)
