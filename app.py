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

# Global file path and pre-load the Excel file once.
FILE_PATH = "PaymentData2024.xlsx"
if os.path.exists(FILE_PATH):
    df_all = pd.read_excel(FILE_PATH)
    facility_options = [
        {"label": fac, "value": fac}
        for fac in sorted(df_all["Facility"].dropna().unique())
    ]
else:
    df_all = None
    facility_options = []


class ClinicDataPreprocessor:
    """
    Loads data from PaymentData2024.xlsx (or from a provided dataframe), calculates 'Net Revenue',
    and trains a Prophet model on *monthly* data *only in 2024*.
    Then forecasts 5 years (60 months) beyond 2024.
    """

    def __init__(
        self,
        file_path="PaymentData2024.xlsx",
        facility_name="",
        periods=60,  # 60 months forecast => 5 years
        data: pd.DataFrame = None
    ):
        """
        :param file_path: Path to your Excel file.
        :param facility_name: Which clinic to filter on.
        :param periods: How many *monthly* periods to forecast (60 => 5 years).
        :param data: Optional dataframe already loaded.
        """
        self.file_path = file_path
        self.facility_name = facility_name
        self.periods = periods
        self.df_raw = data  # Use provided data if available.
        self.visits_df = None
        self.net_revenue_df = None
        self.visits_forecast = None
        self.net_revenue_forecast = None

    def load_data(self):
        if self.df_raw is None:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            df = pd.read_excel(self.file_path)
            logger.info(f"Read {len(df)} rows from '{self.file_path}'.")
        else:
            df = self.df_raw.copy()
            logger.info(f"Using preloaded data with {len(df)} rows.")
        required_cols = ["Facility", "Billed Charge", "Writeoff Adjustment"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing '{col}' in the data.")
        # Create Net Revenue column.
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
        # Filter to 2024.
        df = df[df["ds"].dt.year == 2024]
        # Group monthly.
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
        # Filter to 2024.
        df = df[df["ds"].dt.year == 2024]
        df = df[df["y"] > 0]  # Optional: if negative net revenue doesn't make sense.
        df = df.groupby(pd.Grouper(key="ds", freq="M"))["y"].sum().reset_index()
        return df

    def train_prophet(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("No 2024 data to train on!")
        model = Prophet()
        model.fit(df)
        # Forecast 5 years monthly.
        future = model.make_future_dataframe(periods=self.periods, freq="M")
        forecast = model.predict(future)
        return forecast

    def process_data(self):
        """
        Loads data, filters to the chosen facility,
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
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def create_prophet_forecast_figure(actual_df, forecast_df, title, y_title,
                                   actual_name="Actual", forecast_name="Forecast") -> go.Figure:
    fig = go.Figure()
    # Plot actual data.
    fig.add_trace(go.Scatter(
        x=actual_df["ds"],
        y=actual_df["y"],
        mode="lines+markers",
        name=actual_name,
        line=dict(color="blue", width=2),
        marker=dict(size=5)
    ))
    # Plot the median forecast.
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["yhat"],
        mode="lines",
        name=forecast_name,
        line=dict(color="green", width=3)
    ))
    # Plot confidence intervals.
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
    fixed_cost_inflation_rate: float,  # applies only to fixed costs
    net_revenue_factor: float = 1.0,
    startup_costs: float = 0.0,
    monthly_grant_income: float = 0.0,
    confidence_interval: str = "middle"  # "lower", "middle", or "upper"
) -> pd.DataFrame:
    """
    Break-even logic applying inflation only to fixed costs (termed Fixed Cost Inflation)
    and using the forecasted net revenue as is (except for scaling).
    Allows selection of the confidence interval from the forecasts (for revenue and visits).
    Startup costs are spread evenly over the first 12 months and added to fixed costs.
    Forecasts 5 years out.
    """
    df = df_forecast.copy().sort_values("ds").reset_index(drop=True)
    # Choose the columns based on the confidence_interval option.
    ci = confidence_interval.lower()
    if ci == "lower":
        revenue_column = "net_revenue_yhat_lower"
        visits_column = "visits_yhat_lower"
    elif ci == "upper":
        revenue_column = "net_revenue_yhat_upper"
        visits_column = "visits_yhat_upper"
    else:
        revenue_column = "net_revenue_yhat"
        visits_column = "visits_yhat"
    # Convert annual fixed cost inflation rate to a monthly rate.
    fixed_cost_monthly_inflation = (1 + fixed_cost_inflation_rate) ** (1 / 12) - 1
    # Use the chosen revenue forecast and apply the revenue scaling factor.
    df["net_revenue_scaled"] = df[revenue_column] * net_revenue_factor
    # Monthly net revenue plus any additional grant income.
    df["monthly_net_revenue"] = df["net_revenue_scaled"] + monthly_grant_income
    # Variable costs are directly tied to the chosen visits forecast.
    df["variable_cost"] = df[visits_column] * cost_per_visit
    # Compute fixed costs with fixed cost inflation applied.
    # Also add a portion of startup costs evenly spread over the first 12 months.
    def compute_fixed(row, visits_col=visits_column):
        idx = row.name
        factor = (1 + fixed_cost_monthly_inflation) ** idx
        overhead_i = monthly_base_overhead * factor
        n_phys = math.ceil(row[visits_col] / physician_threshold)
        phys_i = n_phys * (monthly_physician_cost * factor)
        startup_allocation = startup_costs / 12 if idx < 12 else 0
        return overhead_i + phys_i + startup_allocation
    df["fixed_cost"] = df.apply(compute_fixed, axis=1)
    # Monthly profit is revenue minus variable and fixed costs.
    df["monthly_profit"] = df["monthly_net_revenue"] - (df["variable_cost"] + df["fixed_cost"])
    # Compute cumulative profit month by month.
    cumul = []
    running = 0
    for p in df["monthly_profit"]:
        running += p
        cumul.append(running)
    df["cumulative_profit"] = cumul
    # Identify the first month where cumulative profit is non-negative.
    mask = df["cumulative_profit"] >= 0
    be_date = df.loc[mask.idxmax(), "ds"] if mask.any() else None
    df.attrs["break_even_date"] = be_date
    return df


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


# ---------------- MAIN APP / DASH ----------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Use empty figures as placeholders.
empty_fig = go.Figure()

app.layout = dbc.Container([
    html.H1("Clinic Forecast Dashboard", className="mb-4"),
    # Facility selection dropdown.
    dbc.Row([
        dbc.Col([
            html.Label("Select Facility"),
            dcc.Dropdown(
                id="facility-selection",
                options=facility_options,
                value=facility_options[0]["value"] if facility_options else None,
                clearable=False
            )
        ], width=12)
    ], className="mb-3"),
    # Update buttons for Prophet models and Break-Even Analysis.
    dbc.Row([
        dbc.Col([
            dbc.Button("Update Prophet Models", id="update-prophet", color="primary", className="mr-2")
        ], width="auto"),
        dbc.Col([
            dbc.Button("Update Break-Even Analysis", id="update-breakeven", color="secondary")
        ], width="auto")
    ], className="mb-3"),
    # Forecast graphs and performance metrics.
    dbc.Row([
        dbc.Col([
            html.H3("Visits Forecast"),
            dcc.Graph(id="visits-graph", figure=empty_fig),
            html.Div(id="visits-metrics")
        ], width=6),
        dbc.Col([
            html.H3("Net Revenue Forecast"),
            dcc.Graph(id="net-revenue-graph", figure=empty_fig),
            html.Div(id="net-revenue-metrics")
        ], width=6)
    ]),
    html.Hr(),
    # Break-even analysis.
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
        ], width=4)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Monthly Physician Cost"),
            dcc.Input(id="monthly-physician-cost", type="number", value=10000, step=1000, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Fixed Cost Inflation Rate (annual, e.g. 0.03)"),
            dcc.Input(id="fixed-cost-inflation", type="number", value=0.03, step=0.01, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Net Revenue Factor (1.0 => no change)"),
            dcc.Input(id="net-revenue-factor", type="number", value=1.1, step="any", style={"width": "100%"})
        ], width=4)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Label("Start-up Costs (spread over first year)"),
            dcc.Input(id="startup-costs", type="number", value=50000, step=5000, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Grant/Gov Monthly Income"),
            dcc.Input(id="monthly-grant-income", type="number", value=50000, step=100, style={"width": "100%"})
        ], width=4),
        dbc.Col([
            html.Label("Forecast Confidence Interval"),
            dcc.Dropdown(
                id="confidence-interval",
                options=[
                    {"label": "Lower", "value": "lower"},
                    {"label": "Middle", "value": "middle"},
                    {"label": "Upper", "value": "upper"}
                ],
                value="middle",
                clearable=False
            )
        ], width=4)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="break-even-graph", figure=empty_fig)
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Break-Even Date:"),
            html.Div(id="break-even-output", style={"fontWeight": "bold", "marginTop": "10px"})
        ], width=12)
    ])
], fluid=True)


@app.callback(
    [Output("visits-graph", "figure"),
     Output("net-revenue-graph", "figure"),
     Output("visits-metrics", "children"),
     Output("net-revenue-metrics", "children"),
     Output("break-even-graph", "figure"),
     Output("break-even-output", "children")],
    [Input("update-prophet", "n_clicks"),
     Input("update-breakeven", "n_clicks")],
    [State("facility-selection", "value"),
     State("cost-per-visit", "value"),
     State("monthly-base-overhead", "value"),
     State("physician-threshold", "value"),
     State("monthly-physician-cost", "value"),
     State("fixed-cost-inflation", "value"),
     State("net-revenue-factor", "value"),
     State("startup-costs", "value"),
     State("monthly-grant-income", "value"),
     State("confidence-interval", "value")]
)
def update_all(n_clicks_prophet, n_clicks_breakeven, facility, cost_visit, base_overhead, threshold,
               physician_cost, fixed_cost_inflation, net_revenue_factor, startup, grant_income, confidence_interval):
    ctx = dash.callback_context
    if not ctx.triggered or not facility:
        empty = go.Figure()
        return empty, empty, "", "", empty, "No forecast data or not updated yet."
    try:
        # Process data for the selected facility using the preloaded dataframe.
        preprocessor = ClinicDataPreprocessor(
            file_path=FILE_PATH,
            facility_name=facility,
            periods=60,
            data=df_all
        )
        preprocessor.process_data()
    except Exception as e:
        logger.error(f"Error processing facility data: {e}")
        empty = go.Figure()
        return empty, empty, "", "", empty, f"Error: {e}"
    # Prepare merged forecast with confidence intervals.
    v_fore = preprocessor.visits_forecast.rename(columns={
        "yhat": "visits_yhat",
        "yhat_lower": "visits_yhat_lower",
        "yhat_upper": "visits_yhat_upper"
    })
    nr_fore = preprocessor.net_revenue_forecast.rename(columns={
        "yhat": "net_revenue_yhat",
        "yhat_lower": "net_revenue_yhat_lower",
        "yhat_upper": "net_revenue_yhat_upper"
    })
    v_fore = v_fore[["ds", "visits_yhat", "visits_yhat_lower", "visits_yhat_upper"]].sort_values("ds")
    nr_fore = nr_fore[["ds", "net_revenue_yhat", "net_revenue_yhat_lower", "net_revenue_yhat_upper"]].sort_values("ds")
    forecast_merged = v_fore.merge(nr_fore, on="ds", how="inner").sort_values("ds")
    # Create forecast figures.
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
    # Calculate in-sample metrics.
    visits_metrics = {}
    nr_metrics = {}
    hist_len_v = len(preprocessor.visits_df)
    if hist_len_v <= len(preprocessor.visits_forecast):
        actual_v = preprocessor.visits_df["y"]
        pred_v = preprocessor.visits_forecast["yhat"][:hist_len_v]
        visits_metrics = calculate_forecast_metrics(actual_v, pred_v)
    hist_len_nr = len(preprocessor.net_revenue_df)
    if hist_len_nr <= len(preprocessor.net_revenue_forecast):
        actual_nr = preprocessor.net_revenue_df["y"]
        pred_nr = preprocessor.net_revenue_forecast["yhat"][:hist_len_nr]
        nr_metrics = calculate_forecast_metrics(actual_nr, pred_nr)
    visits_metrics_card = metrics_card("Visits Metrics (2024 in-sample)", visits_metrics)
    nr_metrics_card = metrics_card("Net Revenue Metrics (2024 in-sample)", nr_metrics)
    # Calculate break-even analysis.
    be_df = calculate_break_even_forecast(
        df_forecast=forecast_merged,
        cost_per_visit=cost_visit,
        monthly_base_overhead=base_overhead,
        physician_threshold=threshold,
        monthly_physician_cost=physician_cost,
        fixed_cost_inflation_rate=fixed_cost_inflation,
        net_revenue_factor=net_revenue_factor,
        startup_costs=startup,
        monthly_grant_income=grant_income,
        confidence_interval=confidence_interval
    )
    be_date = be_df.attrs.get("break_even_date")
    if be_date is None:
        be_text = "No break-even within forecast horizon."
    else:
        be_text = f"Break-even in {be_date.strftime('%B %Y')}"
    # Build break-even graph.
    be_fig = go.Figure()
    be_fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["cumulative_profit"],
        mode="lines",
        line=dict(color="#4B8BBE", width=3),
        fill="tozeroy",
        fillcolor="rgba(75,139,190,0.2)",
        name="Cumulative Profit"
    ))
    x_min, x_max = be_df["ds"].min(), be_df["ds"].max()
    be_fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[0, 0],
        mode="lines",
        line=dict(color="#FF6E6C", dash="dash", width=2),
        name="Zero Line"
    ))
    if be_date:
        y_min = be_df["cumulative_profit"].min()
        y_max = be_df["cumulative_profit"].max()
        be_fig.add_trace(go.Scatter(
            x=[be_date, be_date],
            y=[y_min, y_max],
            mode="lines",
            line=dict(color="#72CCA7", dash="dot", width=3),
            name="Break-Even Date"
        ))
    be_fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["monthly_net_revenue"],
        mode="lines",
        line=dict(color="#ffd97d", width=3),
        name="Monthly Net Revenue",
        yaxis="y2"
    ))
    be_fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["fixed_cost"],
        mode="lines",
        line=dict(color="#f2b5d4", width=3),
        name="Fixed Cost",
        yaxis="y2"
    ))
    be_fig.add_trace(go.Scatter(
        x=be_df["ds"],
        y=be_df["variable_cost"],
        mode="lines",
        line=dict(color="#c49bbb", width=3),
        name="Variable Cost",
        yaxis="y2"
    ))
    be_fig.update_layout(
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
    return visits_fig, net_revenue_fig, visits_metrics_card, nr_metrics_card, be_fig, be_text


if __name__ == "__main__":
    app.run_server(debug=True)
