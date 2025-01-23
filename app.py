# inflation_break_even.py

import math
import logging
import os
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicDataPreprocessor:
    """
    Loads and forecasts 'payment.xlsx' for a given facility, 5-year forecast.
    """
    def __init__(self, file_path="payment.xlsx", facility_name="UPFH Family Clinic - West Jordan", periods=60):
        self.file_path = file_path
        self.facility_name = facility_name
        self.periods = periods

        self.df_raw = None
        self.visits_df_wj = None
        self.payments_df_wj = None
        self.visits_forecast_wj = None
        self.payments_forecast_wj = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        df = pd.read_excel(self.file_path)
        logger.info(f"Read {len(df)} rows from '{self.file_path}'.")

        if "EncStatus" not in df.columns:
            raise ValueError("Missing 'EncStatus' column.")

        df = df[df["EncStatus"] == "CHK"]
        logger.info(f"Records after filtering EncStatus=='CHK': {len(df)}")
        self.df_raw = df

    def preprocess_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        if "EncDate" not in df.columns:
            raise ValueError("Missing 'EncDate' for visits.")
        df = df.rename(columns={"EncDate":"ds"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds"])
        df = df.groupby(pd.Grouper(key="ds", freq="ME")).size().reset_index(name="y")
        df = df[df["y"] >= 0]
        return df

    def preprocess_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        if "EncDate" not in df.columns or "NetPayment" not in df.columns:
            raise ValueError("Missing 'EncDate' or 'NetPayment'.")
        df = df.rename(columns={"EncDate":"ds","NetPayment":"y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds","y"])
        df = df[df["y"] > 0]
        df = df.groupby(pd.Grouper(key="ds", freq="ME"))["y"].sum().reset_index()
        return df

    def train_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=self.periods, freq="M")
        forecast = model.predict(future)
        return forecast

    def process_data(self):
        """Load, filter facility, preprocess visits/payments, 5-year forecast."""
        self.load_data()

        if "Facility" not in self.df_raw.columns:
            raise ValueError("Missing 'Facility' column in data.")
        facility_df = self.df_raw[self.df_raw["Facility"] == self.facility_name]
        if facility_df.empty:
            raise ValueError(f"No data for facility '{self.facility_name}'.")

        self.visits_df_wj = self.preprocess_visits(facility_df)
        self.payments_df_wj = self.preprocess_payments(facility_df)

        self.visits_forecast_wj = self.train_prophet(self.visits_df_wj)
        self.payments_forecast_wj = self.train_prophet(self.payments_df_wj)

def calculate_forecast_metrics(actual, predicted) -> dict:
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse**0.5
    mape = mean_absolute_percentage_error(actual, predicted)*100
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }

def create_prophet_forecast_figure(actual_df, forecast_df, title, y_title,
                                   actual_name="Actual", forecast_name="Forecast") -> go.Figure:
    fig = go.Figure()
    # Actual
    fig.add_trace(go.Scatter(
        x=actual_df["ds"],
        y=actual_df["y"],
        mode="lines+markers",
        name=actual_name,
        line=dict(color="blue", width=2),
        marker=dict(size=5)
    ))
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        mode="lines",
        name=forecast_name,
        line=dict(color="green", width=3)
    ))
    # Confidence intervals
    if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat_upper"],
            mode="lines",
            line=dict(color="lightgreen", dash="dash"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat_lower"],
            mode="lines",
            line=dict(color="lightgreen", dash="dash"),
            fill="tonexty",
            fillcolor="rgba(144,238,144,0.2)",
            showlegend=False
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        template="plotly_white",
        height=400
    )
    return fig

def calculate_break_even_forecast(
    df_forecast: pd.DataFrame,
    cost_per_visit: float,
    monthly_base_overhead: float,
    physician_threshold: float,
    monthly_physician_cost: float,
    cost_inflation_rate: float,    # separate inflation for overhead & physician cost
    payments_inflation_rate: float, # separate inflation for payments
    startup_costs: float=0.0,
    monthly_grant_income: float=0.0,
    payments_factor: float=1.0
) -> pd.DataFrame:
    """
    1. Apply monthly inflation to payments, then scale by 'payments_factor'.
    2. monthly_revenue = inflated & scaled payments + grant
    3. variable_cost = visits_yhat * cost_per_visit
    4. overhead & physician cost also inflated by cost_inflation.
    5. monthly_profit, cumulative_profit => break-even date.
    """

    df = df_forecast.copy()
    df = df.sort_values("ds").reset_index(drop=True)

    # Convert annual rates to monthly
    cost_monthly_inflation = (1 + cost_inflation_rate)**(1/12) - 1
    pay_monthly_inflation = (1 + payments_inflation_rate)**(1/12) - 1

    # 1) Payment inflation
    def inflated_payments(row):
        idx = row.name
        # each row is month i => scale by pay_monthly_inflation
        # e.g. row['payments_yhat'] * (1 + pay_monthly_inflation)**idx
        factor = (1 + pay_monthly_inflation)**idx
        return row["payments_yhat"] * factor

    df["payments_inflated"] = df.apply(inflated_payments, axis=1)
    # then apply factor
    df["payments_scaled"] = df["payments_inflated"] * payments_factor

    # monthly revenue
    df["monthly_revenue"] = df["payments_scaled"] + monthly_grant_income

    # variable cost
    df["variable_cost"] = df["visits_yhat"] * cost_per_visit

    # overhead & physician cost inflated by cost_inflation
    def compute_fixed(row):
        idx = row.name
        factor = (1 + cost_monthly_inflation)**idx
        overhead_i = monthly_base_overhead * factor
        # number of physicians
        n_phys = math.ceil(row["visits_yhat"] / physician_threshold)
        phys_i = n_phys * (monthly_physician_cost * factor)
        return overhead_i + phys_i

    df["fixed_cost"] = df.apply(compute_fixed, axis=1)

    df["monthly_profit"] = df["monthly_revenue"] - (df["variable_cost"] + df["fixed_cost"])

    # cumulative
    cumul = []
    running = -startup_costs
    for p in df["monthly_profit"]:
        running += p
        cumul.append(running)
    df["cumulative_profit"] = cumul

    # break-even
    mask = df["cumulative_profit"] >= 0
    be_date = None
    if mask.any():
        idx = mask.idxmax()
        be_date = df.loc[idx, "ds"]
    df.attrs["break_even_date"] = be_date

    return df


# ---------- Load & Merge Forecasts ---------------
preprocessor = None
try:
    preprocessor = ClinicDataPreprocessor(
        file_path="payment.xlsx",
        facility_name="UPFH Family Clinic - West Jordan",
        periods=60  # 5 year
    )
    preprocessor.process_data()
    logger.info("Data loaded & forecasted.")
except Exception as e:
    logger.error(f"Error: {e}")

forecast_merged_wj = pd.DataFrame(columns=["ds","visits_yhat","payments_yhat"])
visits_metrics = {}
payments_metrics = {}
visits_fig = go.Figure()
payments_fig = go.Figure()

if (
    preprocessor and
    preprocessor.visits_forecast_wj is not None and
    preprocessor.payments_forecast_wj is not None and
    not preprocessor.visits_forecast_wj.empty and
    not preprocessor.payments_forecast_wj.empty
):
    v_fore = preprocessor.visits_forecast_wj.rename(columns={"yhat":"visits_yhat"})
    p_fore = preprocessor.payments_forecast_wj.rename(columns={"yhat":"payments_yhat"})
    v_fore = v_fore[["ds","visits_yhat"]].sort_values("ds")
    p_fore = p_fore[["ds","payments_yhat"]].sort_values("ds")

    forecast_merged_wj = v_fore.merge(p_fore, on="ds", how="inner").sort_values("ds")

    # In-sample metrics
    hist_len_v = len(preprocessor.visits_df_wj)
    if hist_len_v <= len(v_fore):
        actual_v = preprocessor.visits_df_wj["y"]
        pred_v   = v_fore["visits_yhat"][:hist_len_v]
        visits_metrics = calculate_forecast_metrics(actual_v, pred_v)

    hist_len_p = len(preprocessor.payments_df_wj)
    if hist_len_p <= len(p_fore):
        actual_p = preprocessor.payments_df_wj["y"]
        pred_p   = p_fore["payments_yhat"][:hist_len_p]
        payments_metrics = calculate_forecast_metrics(actual_p, pred_p)

    visits_fig = create_prophet_forecast_figure(
        preprocessor.visits_df_wj,
        preprocessor.visits_forecast_wj,
        title="Visits Forecast (5 Years)",
        y_title="Visits"
    )
    payments_fig = create_prophet_forecast_figure(
        preprocessor.payments_df_wj,
        preprocessor.payments_forecast_wj,
        title="Payments Forecast (5 Years)",
        y_title="Payments ($)"
    )

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def metrics_card(title, m):
    if not m:
        return dbc.Card([dbc.CardBody("No metrics available.")])
    return dbc.Card([
        dbc.CardHeader(html.H5(title)),
        dbc.CardBody([
            html.Div(f"RMSE: {m.get('RMSE',0):.2f}"),
            html.Div(f"MAPE: {m.get('MAPE',0):.2f}%"),
            html.Div(f"MAE: {m.get('MAE',0):.2f}"),
            html.Div(f"MSE: {m.get('MSE',0):.2f}")
        ])
    ], className="mb-3")

app.layout = dbc.Container([
    html.H1("Break-Even Analysis (Separate Cost & Payments Inflation)", className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H3("Visits Forecast"),
            dcc.Graph(figure=visits_fig),
            metrics_card("Visits Metrics", visits_metrics)
        ], width=6),
        dbc.Col([
            html.H3("Payments Forecast"),
            dcc.Graph(figure=payments_fig),
            metrics_card("Payments Metrics", payments_metrics)
        ], width=6),
    ]),

    html.Hr(),
    html.H2("Break-Even Analysis", className="mt-4 mb-3"),

    # First row of user inputs
    dbc.Row([
        dbc.Col([
            html.Label("Cost per Visit (Variable)"),
            dcc.Input(id="cost-per-visit", type="number", value=100, step=10, style={"width":"100%"})
        ], width=4),
        dbc.Col([
            html.Label("Monthly Base Overhead (Fixed)"),
            dcc.Input(id="monthly-base-overhead", type="number", value=20000, step=1000, style={"width":"100%"})
        ], width=4),
        dbc.Col([
            html.Label("Physician Threshold (Visits)"),
            dcc.Input(id="physician-threshold", type="number", value=500, step=50, style={"width":"100%"})
        ], width=4),
    ], className="mb-3"),

    # Second row of user inputs
    dbc.Row([
        dbc.Col([
            html.Label("Monthly Physician Cost"),
            dcc.Input(id="monthly-physician-cost", type="number", value=10000, step=1000, style={"width":"100%"})
        ], width=4),
        dbc.Col([
            html.Label("Cost Inflation Rate (annual, e.g. 0.03)"),
            dcc.Input(id="cost-inflation", type="number", value=0.03, step=0.01, style={"width":"100%"})
        ], width=4),
        dbc.Col([
            html.Label("Payments Inflation Rate (annual, e.g. 0.02)"),
            dcc.Input(id="payments-inflation", type="number", value=0.02, step=0.01, style={"width":"100%"})
        ], width=4),
    ], className="mb-3"),

    # Third row
    dbc.Row([
        dbc.Col([
            html.Label("Start-up Costs"),
            dcc.Input(id="startup-costs", type="number", value=50000, step=5000, style={"width":"100%"})
        ], width=4),
        dbc.Col([
            html.Label("Grant/Gov Monthly Income"),
            dcc.Input(id="monthly-grant-income", type="number", value=50000, step=100, style={"width":"100%"})
        ], width=4),
        dbc.Col([
            html.Label("Payments Factor (1.0 => no change)"),
            dcc.Input(id="payments-factor", type="number", value=1.1, step="any", style={"width":"100%"})
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Button("Update Model", id="update-button", color="primary", className="mt-4")
        ], width=12),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id="break-even-graph")
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Break-Even Date:"),
            html.Div(id="break-even-output", style={"fontWeight":"bold","marginTop":"10px"})
        ], width=12),
    ])
], fluid=True)

@app.callback(
    [Output("break-even-graph", "figure"),
     Output("break-even-output", "children")],
    Input("update-button", "n_clicks"),
    [
        State("cost-per-visit","value"),
        State("monthly-base-overhead","value"),
        State("physician-threshold","value"),
        State("monthly-physician-cost","value"),
        State("cost-inflation","value"),
        State("payments-inflation","value"),
        State("startup-costs","value"),
        State("monthly-grant-income","value"),
        State("payments-factor","value")
    ]
)
def update_break_even(
    n_clicks,
    cost_visit,
    base_overhead,
    threshold,
    physician_cost,
    cost_inflation,
    pay_inflation,
    startup,
    grant_income,
    pay_factor
):
    if n_clicks is None or forecast_merged_wj.empty:
        fig = go.Figure()
        return fig, "No forecast data or not updated yet."

    # 1) Calculate break-even with separate inflation for cost & payments
    be_df = calculate_break_even_forecast(
        df_forecast=forecast_merged_wj,
        cost_per_visit=cost_visit,
        monthly_base_overhead=base_overhead,
        physician_threshold=threshold,
        monthly_physician_cost=physician_cost,
        cost_inflation_rate=cost_inflation,
        payments_inflation_rate=pay_inflation,
        startup_costs=startup,
        monthly_grant_income=grant_income,
        payments_factor=pay_factor
    )

    be_date = be_df.attrs.get("break_even_date")
    if be_date is None:
        be_text = "No break-even within forecast horizon."
    else:
        be_text = f"Break-even in {be_date.strftime('%B %Y')}"

    # 2) Build figure (pastel color approach)
    fig = go.Figure()

    # Cumulative Profit
    fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["cumulative_profit"],
        mode="lines",
        line=dict(color="#4B8BBE", width=3),
        fill="tozeroy",
        fillcolor="rgba(75,139,190,0.2)",
        name="Cumulative Profit"
    ))

    # Zero line
    x_min, x_max = be_df["ds"].min(), be_df["ds"].max()
    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[0,0],
        mode="lines",
        line=dict(color="#FF6E6C", dash="dash", width=2),
        name="Zero Line"
    ))

    # Break-even date
    if be_date:
        y_min = be_df["cumulative_profit"].min()
        y_max = be_df["cumulative_profit"].max()
        fig.add_trace(go.Scatter(
            x=[be_date, be_date],
            y=[y_min, y_max],
            mode="lines",
            line=dict(color="#72CCA7", dash="dot", width=3),
            name="Break-Even Date"
        ))

    # Secondary axis lines: monthly_revenue, etc.
    fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["monthly_revenue"],
        mode="lines",
        line=dict(color="#ffd97d", width=3),
        name="Monthly Revenue",
        yaxis="y2"
    ))
    # If needed, keep track of the inflated payment lines separate, but
    # here we just show monthly_revenue total.

    fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["fixed_cost"],
        mode="lines",
        line=dict(color="#f2b5d4", width=3),
        name="Fixed Cost",
        yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["variable_cost"],
        mode="lines",
        line=dict(color="#c49bbb", width=3),
        name="Variable Cost",
        yaxis="y2"
    ))

    fig.update_layout(
        title="Break-Even Analysis (Separate Cost & Payments Inflation)",
        title_x=0.5,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Cumulative Profit ($)",
        height=600,

        yaxis2=dict(
            title="Revenue / Costs ($)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig, be_text

if __name__ == "__main__":
    app.run_server(debug=True)
