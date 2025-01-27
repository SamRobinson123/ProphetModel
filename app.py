import math
import logging
import os
from datetime import datetime

import dash
import openpyxl  # For reading Excel files
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
    Loads data from PaymentData2024.xlsx, calculates 'Net Revenue',
    and trains a Prophet model on *monthly* data *only in 2024*.
    Then forecasts 5 years (60 months) beyond 2024.
    """

    def __init__(
        self,
        file_path="PaymentData2024.xlsx",
        facility_name="UPFH Family Clinic - West Jordan",
        periods=60  # 60 months forecast => 5 years
    ):
        """
        :param file_path: Path to your Excel file
        :param facility_name: Which clinic to filter on
        :param periods: How many *monthly* periods to forecast (60 => 5 years)
        """
        self.file_path = file_path
        self.facility_name = facility_name
        self.periods = periods

        self.df_raw = None
        self.visits_df = None
        self.net_revenue_df = None
        self.visits_forecast = None
        self.net_revenue_forecast = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        df = pd.read_excel(self.file_path)
        logger.info(f"Read {len(df)} rows from '{self.file_path}'.")

        required_cols = ["Facility", "Billed Charge", "Writeoff Adjustment"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing '{col}' in the data.")

        # Create Net Revenue
        df["Net Revenue"] = df["Billed Charge"] - df["Writeoff Adjustment"]
        self.df_raw = df

    def preprocess_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Monthly visits: filter to 2024, group by month, count visits.
        """
        if "Service Date" not in df.columns:
            raise ValueError("Missing 'Service Date' for visits.")

        df = df.rename(columns={"Service Date": "ds"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds"])

        # Filter to 2024
        df = df[df["ds"].dt.year == 2024]

        # Group monthly
        df = df.groupby(pd.Grouper(key="ds", freq="M")).size().reset_index(name="y")
        df = df[df["y"] >= 0]
        return df

    def preprocess_net_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Monthly Net Revenue: filter to 2024, group by month, sum net revenue.
        """
        required_cols = ["Claim Date", "Net Revenue"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing '{col}' for net revenue.")

        df = df.rename(columns={"Claim Date": "ds", "Net Revenue": "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds", "y"])

        # Filter to 2024
        df = df[df["ds"].dt.year == 2024]

        df = df[df["y"] > 0]  # optional, if negative net revenue doesn't make sense
        df = df.groupby(pd.Grouper(key="ds", freq="M"))["y"].sum().reset_index()
        return df

    def train_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("No 2024 data to train on!")

        model = Prophet()
        model.fit(df)
        # Forecast 5 years monthly
        future = model.make_future_dataframe(periods=self.periods, freq="M")
        forecast = model.predict(future)
        return forecast

    def process_data(self):
        """
        Loads data, filters to facility,
        preprocesses visits & net revenue for 2024, trains monthly,
        and forecasts 60 months beyond 2024.
        """
        self.load_data()
        facility_df = self.df_raw[self.df_raw["Facility"] == self.facility_name]
        if facility_df.empty:
            raise ValueError(f"No data for '{self.facility_name}' in the file.")

        self.visits_df = self.preprocess_visits(facility_df)
        self.net_revenue_df = self.preprocess_net_revenue(facility_df)

        self.visits_forecast = self.train_prophet(self.visits_df)
        self.net_revenue_forecast = self.train_prophet(self.net_revenue_df)


def calculate_forecast_metrics(actual, predicted) -> dict:
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse**0.5
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }


def create_prophet_forecast_figure(
    actual_df, forecast_df, title, y_title,
    actual_name="Actual", forecast_name="Forecast"
) -> go.Figure:
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
    cost_inflation_rate: float,
    net_revenue_inflation_rate: float,
    startup_costs: float = 0.0,
    monthly_grant_income: float = 0.0,
    net_revenue_factor: float = 1.0
) -> pd.DataFrame:
    """
    Break-even logic with monthly inflation; forecasting 5 years out.
    """
    df = df_forecast.copy().sort_values("ds").reset_index(drop=True)

    # Convert annual rates to monthly
    cost_monthly_inflation = (1 + cost_inflation_rate)**(1/12) - 1
    revenue_monthly_inflation = (1 + net_revenue_inflation_rate)**(1/12) - 1

    def inflated_net_revenue(row):
        idx = row.name
        factor = (1 + revenue_monthly_inflation) ** idx
        return row["net_revenue_yhat"] * factor

    df["net_revenue_inflated"] = df.apply(inflated_net_revenue, axis=1)
    df["net_revenue_scaled"] = df["net_revenue_inflated"] * net_revenue_factor

    df["monthly_net_revenue"] = df["net_revenue_scaled"] + monthly_grant_income
    df["variable_cost"] = df["visits_yhat"] * cost_per_visit

    def compute_fixed(row):
        idx = row.name
        factor = (1 + cost_monthly_inflation) ** idx
        overhead_i = monthly_base_overhead * factor
        n_phys = math.ceil(row["visits_yhat"] / physician_threshold)
        phys_i = n_phys * (monthly_physician_cost * factor)
        return overhead_i + phys_i

    df["fixed_cost"] = df.apply(compute_fixed, axis=1)
    df["monthly_profit"] = df["monthly_net_revenue"] - (df["variable_cost"] + df["fixed_cost"])

    cumul = []
    running = -startup_costs
    for p in df["monthly_profit"]:
        running += p
        cumul.append(running)
    df["cumulative_profit"] = cumul

    mask = df["cumulative_profit"] >= 0
    be_date = df.loc[mask.idxmax(), "ds"] if mask.any() else None
    df.attrs["break_even_date"] = be_date

    return df


# ---------------- MAIN APP / DASH ----------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load + forecast
preprocessor = None
try:
    # Train on 2024 monthly, forecast 5 years (60 months)
    preprocessor = ClinicDataPreprocessor(
        file_path="PaymentData2024.xlsx",
        facility_name="UPFH Family Clinic - West Jordan",
        periods=60
    )
    preprocessor.process_data()
    logger.info("Data loaded (2024 monthly), forecast out 5 years (60 months).")
except Exception as e:
    logger.error(f"Error: {e}")

# Merge forecasts
forecast_merged = pd.DataFrame(columns=["ds", "visits_yhat", "net_revenue_yhat"])
visits_metrics = {}
net_revenue_metrics = {}
visits_fig = go.Figure()
net_revenue_fig = go.Figure()

if (
    preprocessor
    and preprocessor.visits_forecast is not None
    and preprocessor.net_revenue_forecast is not None
    and not preprocessor.visits_forecast.empty
    and not preprocessor.net_revenue_forecast.empty
):
    v_fore = preprocessor.visits_forecast.rename(columns={"yhat": "visits_yhat"})
    nr_fore = preprocessor.net_revenue_forecast.rename(columns={"yhat": "net_revenue_yhat"})
    v_fore = v_fore[["ds", "visits_yhat"]].sort_values("ds")
    nr_fore = nr_fore[["ds", "net_revenue_yhat"]].sort_values("ds")

    forecast_merged = v_fore.merge(nr_fore, on="ds", how="inner").sort_values("ds")

    # In-sample metrics for the year 2024
    hist_len_v = len(preprocessor.visits_df)
    if hist_len_v <= len(v_fore):
        actual_v = preprocessor.visits_df["y"]
        pred_v = v_fore["visits_yhat"][:hist_len_v]
        visits_metrics = calculate_forecast_metrics(actual_v, pred_v)

    hist_len_nr = len(preprocessor.net_revenue_df)
    if hist_len_nr <= len(nr_fore):
        actual_nr = preprocessor.net_revenue_df["y"]
        pred_nr = nr_fore["net_revenue_yhat"][:hist_len_nr]
        net_revenue_metrics = calculate_forecast_metrics(actual_nr, pred_nr)

    # Figures
    visits_fig = create_prophet_forecast_figure(
        preprocessor.visits_df,
        preprocessor.visits_forecast,
        title="Visits Forecast (2024 Monthly +5 Years)",
        y_title="Visits"
    )
    net_revenue_fig = create_prophet_forecast_figure(
        preprocessor.net_revenue_df,
        preprocessor.net_revenue_forecast,
        title="Net Revenue Forecast (2024 Monthly +5 Years)",
        y_title="Net Revenue ($)"
    )


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
    html.H1("Monthly Model, Trained on 2024 Only, +5 Years Forecast", className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H3("Visits Forecast"),
            dcc.Graph(figure=visits_fig),
            metrics_card("Visits Metrics (2024 in-sample)", visits_metrics)
        ], width=6),
        dbc.Col([
            html.H3("Net Revenue Forecast"),
            dcc.Graph(figure=net_revenue_fig),
            metrics_card("Net Revenue Metrics (2024 in-sample)", net_revenue_metrics)
        ], width=6),
    ]),

    html.Hr(),
    html.H2("Break-Even Analysis", className="mt-4 mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Cost per Visit (Variable)"),
            dcc.Input(id="cost-per-visit", type="number", value=100, step=10, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Monthly Base Overhead (Fixed)"),
            dcc.Input(id="monthly-base-overhead", type="number", value=20000, step=1000, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Physician Threshold (Visits)"),
            dcc.Input(id="physician-threshold", type="number", value=500, step=50, style={"width": "100%"})
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Monthly Physician Cost"),
            dcc.Input(id="monthly-physician-cost", type="number", value=10000, step=1000, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Cost Inflation Rate (annual, e.g. 0.03)"),
            dcc.Input(id="cost-inflation", type="number", value=0.03, step=0.01, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Net Revenue Inflation Rate (annual, e.g. 0.02)"),
            dcc.Input(id="net-revenue-inflation", type="number", value=0.02, step=0.01, style={"width": "100%"})
        ], width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Start-up Costs"),
            dcc.Input(id="startup-costs", type="number", value=50000, step=5000, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Grant/Gov Monthly Income"),
            dcc.Input(id="monthly-grant-income", type="number", value=50000, step=100, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Net Revenue Factor (1.0 => no change)"),
            dcc.Input(id="net-revenue-factor", type="number", value=1.1, step="any", style={"width": "100%"})
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
            html.Div(id="break-even-output", style={"fontWeight": "bold","marginTop":"10px"})
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
        State("net-revenue-inflation","value"),
        State("startup-costs","value"),
        State("monthly-grant-income","value"),
        State("net-revenue-factor","value")
    ]
)
def update_break_even(
    n_clicks,
    cost_visit,
    base_overhead,
    threshold,
    physician_cost,
    cost_inflation,
    net_revenue_inflation,
    startup,
    grant_income,
    net_revenue_factor
):
    if not n_clicks or forecast_merged.empty:
        fig = go.Figure()
        return fig, "No forecast data or not updated yet."

    be_df = calculate_break_even_forecast(
        df_forecast=forecast_merged,
        cost_per_visit=cost_visit,
        monthly_base_overhead=base_overhead,
        physician_threshold=threshold,
        monthly_physician_cost=physician_cost,
        cost_inflation_rate=cost_inflation,
        net_revenue_inflation_rate=net_revenue_inflation,
        startup_costs=startup,
        monthly_grant_income=grant_income,
        net_revenue_factor=net_revenue_factor
    )

    be_date = be_df.attrs.get("break_even_date")
    if be_date is None:
        be_text = "No break-even within forecast horizon."
    else:
        be_text = f"Break-even in {be_date.strftime('%B %Y')}"

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
        y=[0, 0],
        mode="lines",
        line=dict(color="#FF6E6C", dash="dash", width=2),
        name="Zero Line"
    ))
    # Break-even date line
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

    # Additional lines on second axis
    fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["monthly_net_revenue"],
        mode="lines",
        line=dict(color="#ffd97d", width=3),
        name="Monthly Net Revenue",
        yaxis="y2"
    ))
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
        title="Break-Even Analysis (Monthly, 2024-Only +5 Years)",
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
