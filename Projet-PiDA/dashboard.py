# ============================================
# DASHBOARD
# ============================================

from dash import Dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet


# --------------------------------------------
# DATA LOADING & PREPROCESSING
# --------------------------------------------
df = pd.read_csv("wfp-hungermap-data-for-mdg.csv", skiprows=[1])
df["date"] = pd.to_datetime(df["date"])
df = df.dropna()

# Temporal aggregation
df_grouped = df.groupby("date").agg({
    "fcs people": "sum",
    "rcsi people": "sum",
    "health access people": "sum",
    "market access people": "sum",
    "population": "sum"
}).reset_index()

# Regional aggregation
df_regional = df.groupby("adminone").agg({
    "fcs people": "mean",
    "fcs prevalence": "mean",
    "rcsi prevalence": "mean",
    "population": "mean"
}).reset_index()

# Aggregation for treemap animation
def aggregate_by_date_region(df, date_col, region_col, value_col):
    """Aggregate data by date and region"""
    return df.groupby([date_col, region_col]).agg({
        value_col: "sum"
    }).reset_index()

grouped = aggregate_by_date_region(df, "date", "adminone", "fcs people")

# Calculate latest statistics
latest_date = df_grouped["date"].max()
prev_date = df_grouped["date"].unique()[-2] if len(df_grouped) > 1 else latest_date

latest_stats = df_grouped[df_grouped["date"] == latest_date].iloc[0]
prev_stats = df_grouped[df_grouped["date"] == prev_date].iloc[0]

# --------------------------------------------
# ENHANCED FIGURES
# --------------------------------------------

def create_area_chart(y_col, title, color):
    """Create beautiful area chart with gradient fill"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_grouped["date"],
        y=df_grouped[y_col],
        mode='lines',
        name=y_col,
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        template="plotly_white",
        height=400,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Number of People",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(248, 249, 250, 0.5)'
    )
    
    return fig

# Create individual charts
fig_fcs = create_area_chart(
    "fcs people",
    "Indicator 1 — Food Consumption Score (FCS) People Over Time",
    "#3b82f6"
)

fig_rcsi = create_area_chart(
    "rcsi people",
    "Indicator 2 — Reduced Coping Strategy Index (RCSI) People Over Time",
    "#8b5cf6"
)

fig_health = create_area_chart(
    "health access people",
    "Indicator 3 — Health Access Constraints Over Time",
    "#ec4899"
)

fig_market = create_area_chart(
    "market access people",
    "Indicator 4 — Market Access Constraints Over Time",
    "#f59e0b"
)

# Combined comparison chart
fig_combined = go.Figure()

indicators = [
    ("fcs people", "FCS People", "#3b82f6"),
    ("rcsi people", "RCSI People", "#8b5cf6"),
    ("health access people", "Health Access", "#ec4899"),
    ("market access people", "Market Access", "#f59e0b")
]

for col, name, color in indicators:
    fig_combined.add_trace(go.Scatter(
        x=df_grouped["date"],
        y=df_grouped[col],
        mode='lines',
        name=name,
        line=dict(color=color, width=2.5)
    ))

fig_combined.update_layout(
    title=dict(text="All Indicators Comparison", font=dict(size=22, color='#2c3e50')),
    template="plotly_white",
    height=500,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Inter, sans-serif"),
    plot_bgcolor='rgba(248, 249, 250, 0.5)'
)

# Regional bar chart
fig_regional = px.bar(
    df_regional.sort_values("fcs prevalence", ascending=False),
    x="adminone",
    y="fcs prevalence",
    title="FCS Prevalence by Region",
    color="fcs prevalence",
    color_continuous_scale="Reds",
    labels={"adminone": "Region", "fcs prevalence": "FCS Prevalence (%)"}
)

fig_regional.update_layout(
    template="plotly_white",
    height=450,
    font=dict(family="Inter, sans-serif"),
    xaxis_tickangle=-45,
    showlegend=False
)

# Scatter plot: FCS vs RCSI prevalence
fig_scatter = px.scatter(
    df_regional,
    x="fcs prevalence",
    y="rcsi prevalence",
    size="population",
    hover_name="adminone",
    title="Regional Analysis: FCS vs RCSI Prevalence",
    labels={
        "fcs prevalence": "FCS Prevalence",
        "rcsi prevalence": "RCSI Prevalence"
    },
    color="fcs prevalence",
    color_continuous_scale="Viridis"
)

fig_scatter.update_layout(
    template="plotly_white",
    height=450,
    font=dict(family="Inter, sans-serif")
)

# --------------------------------------------
# DYNAMIC TREEMAP WITH TIME SLIDER
# --------------------------------------------

def visualize_treemap_time_slider(df, date_col, region_col, value_col, title):
    """
    Create an animated treemap with a time slider.
    Input:
        df (DataFrame): Data source
        date_col (str): Column name for dates
        region_col (str): Column name for regions
        value_col (str): Column name for the indicator
        title (str): Chart title
    Output:
        Plotly Figure
    """
    # Convert dates to string format
    df = df.copy()
    df[date_col] = df[date_col].astype(str)
    
    # Unique dates for frames
    dates = sorted(df[date_col].unique())
    
    # Compute min and max for color scaling
    val_min = df[value_col].min()
    val_max = df[value_col].max()
    
    # Frames for animation
    frames = []
    for d in dates:
        df_d = df[df[date_col] == d].copy()
        
        # Calculate colour based on normalised value
        df_d['color_value'] = df_d[value_col]
        
        frames.append(
            go.Frame(
                data=[
                    go.Treemap(
                        labels=df_d[region_col],
                        parents=[""] * len(df_d),  # No parents to show flat structure
                        values=df_d[value_col],
                        text=[f"<b>{region}</b><br>{val:,.0f}" 
                              for region, val in zip(df_d[region_col], df_d[value_col])],
                        textposition="middle center",
                        textfont=dict(size=12, color="white", family="Arial"),
                        marker=dict(
                            colors=df_d['color_value'],
                            colorscale="RdYlGn_r",
                            cmin=val_min,
                            cmax=val_max,
                            line=dict(width=3, color="white"),
                            colorbar=dict(
                                title=dict(
                                    text="FCS people",
                                    side="right",
                                    font=dict(size=14)
                                ),
                                thickness=20,
                                len=0.7,
                                x=1.02,
                                tickformat=",",
                                tickfont=dict(size=11)
                            )
                        ),
                        hovertemplate="<b>%{label}</b><br>" +
                                      f"Value: %{{value:,.0f}}<br>" +
                                      "<extra></extra>",
                        texttemplate="%{text}",
                        pathbar=dict(visible=False)
                    )
                ],
                name=str(d)
            )
        )

    # Treemap initial
    df_init = df[df[date_col] == dates[0]].copy()
    df_init['color_value'] = df_init[value_col]

    fig = go.Figure(
        data=[
            go.Treemap(
                labels=df_init[region_col],
                parents=[""] * len(df_init),
                values=df_init[value_col],
                text=[f"<b>{region}</b><br>{val:,.0f}" 
                      for region, val in zip(df_init[region_col], df_init[value_col])],
                textposition="middle center",
                textfont=dict(size=12, color="white", family="Arial"),
                marker=dict(
                    colors=df_init['color_value'],
                    colorscale="RdYlGn_r",
                    cmin=val_min,
                    cmax=val_max,
                    line=dict(width=3, color="white"),
                    colorbar=dict(
                        title=dict(
                            text="FCS people",
                            side="right",
                            font=dict(size=14)
                        ),
                        thickness=20,
                        len=0.7,
                        x=1.02,
                        tickformat=",",
                        tickfont=dict(size=11)
                    )
                ),
                hovertemplate="<b>%{label}</b><br>" +
                              f"Value: %{{value:,.0f}}<br>" +
                              "<extra></extra>",
                texttemplate="%{text}",
                pathbar=dict(visible=False)
            )
        ],
        layout=go.Layout(
            title=dict(
                text=title,
                font=dict(size=22, color="#2c3e50", family="Arial"),
                x=0.5,
                xanchor="center",
                y=0.98,
                yanchor="top"
            ),
            font=dict(family="Arial", size=12),
            paper_bgcolor="#f8f9fa",
            plot_bgcolor="#f8f9fa",
            margin=dict(t=80, l=10, r=120, b=120),
            height=600,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    x=1.3,
                    y=-0.15,
                    xanchor="center",
                    yanchor="top",
                    direction="left",
                    buttons=[
                        dict(
                            label="▶️ Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 400, "easing": "cubic-in-out"}
                            }]
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ],
                    bgcolor="#3498db",
                    bordercolor="#2980b9",
                    borderwidth=2,
                    font=dict(color="white", size=13, family="Arial")
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="left",
                    x=0.02,
                    len=0.96,
                    currentvalue=dict(
                        prefix="📅 Date: ",
                        font=dict(size=15, color="#2c3e50", family="Arial"),
                        visible=True,
                        xanchor="left"
                    ),
                    transition=dict(duration=400, easing="cubic-in-out"),
                    pad=dict(b=10, t=40),
                    bgcolor="#ecf0f1",
                    bordercolor="#bdc3c7",
                    borderwidth=2,
                    tickcolor="#3498db",
                    ticklen=5,
                    tickwidth=2,
                    steps=[
                        dict(
                            method="animate",
                            args=[[str(d)], {
                                "frame": {"duration": 400, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 400, "easing": "cubic-in-out"}
                            }],
                            label=str(d)
                        )
                        for d in dates
                    ]
                )
            ]
        ),
        frames=frames
    )

    return fig

# Create dynamic treemap
fig_dynamic = visualize_treemap_time_slider(
    grouped,
    "date",
    "adminone",
    "fcs people",
    "Dynamic Treemap of FCS People by Region Over Time"
)

# --------------------------------------------
# ANIMATED MAP WITH REGION COORDINATES
# --------------------------------------------

# Madagascar region coordinates (approximate centroids)
region_coords = {
    'Alaotra Mangoro': {'lat': -17.9, 'lon': 48.5},
    'Amoron i Mania': {'lat': -20.5, 'lon': 47.0},
    'Amoron I Mania': {'lat': -20.5, 'lon': 47.0},  # Alternative spelling
    'Analamanga': {'lat': -18.9, 'lon': 47.5},
    'Analanjirofo': {'lat': -17.0, 'lon': 49.5},
    'Androy': {'lat': -25.0, 'lon': 46.0},
    'Anosy': {'lat': -24.5, 'lon': 47.0},
    'Atsimo Andrefana': {'lat': -22.5, 'lon': 44.5},
    'Atsimo Atsinanana': {'lat': -23.5, 'lon': 47.5},
    'Atsinanana': {'lat': -18.5, 'lon': 49.0},
    'Betsiboka': {'lat': -16.5, 'lon': 47.0},
    'Boeny': {'lat': -15.7, 'lon': 46.3},
    'Bongolava': {'lat': -18.5, 'lon': 46.5},
    'Diana': {'lat': -13.0, 'lon': 49.5},
    'Haute Matsiatra': {'lat': -21.5, 'lon': 47.0},
    'Ihorombe': {'lat': -22.5, 'lon': 46.5},
    'Itasy': {'lat': -18.8, 'lon': 46.8},
    'Melaky': {'lat': -18.5, 'lon': 45.0},
    'Menabe': {'lat': -20.0, 'lon': 45.0},
    'Sava': {'lat': -14.0, 'lon': 50.0},
    'Sofia': {'lat': -15.0, 'lon': 48.0},
    'Vakinankaratra': {'lat': -19.5, 'lon': 47.0},
    'Vatovavy Fitovinany': {'lat': -21.5, 'lon': 48.0}
}

# Prepare data for animated map
df_regions = df.copy()

# Add coordinates to dataframe
df_regions['lat'] = df_regions['adminone'].map(lambda x: region_coords.get(x, {}).get('lat'))
df_regions['lon'] = df_regions['adminone'].map(lambda x: region_coords.get(x, {}).get('lon'))

# Remove rows without coordinates
df_regions = df_regions.dropna(subset=['lat', 'lon'])

# Sort by date
df_regions = df_regions.sort_values('date')

# Convert date to string format for animation frame
df_regions['date_str'] = df_regions['date'].dt.strftime('%Y-%m-%d')

# Create the animated map
fig_map = px.scatter_mapbox(
    df_regions,
    lat="lat",
    lon="lon",
    color="fcs prevalence",
    size="population",
    animation_frame="date_str",
    hover_name="adminone",
    hover_data={
        "fcs prevalence": ":.2%",
        "population": ":,",
        "lat": False,
        "lon": False,
        "date_str": False
    },
    color_continuous_scale="RdYlGn_r",
    size_max=30,
    zoom=4.5,
    center={"lat": -19, "lon": 47},
    title="FCS Prevalence by Region Over Time",
    labels={
        "fcs prevalence": "FCS Prevalence",
        "population": "Population",
        "date_str": "Date"
    },
    range_color=[0, df_regions['fcs prevalence'].max()]
)

# Update map layout
fig_map.update_layout(
    mapbox_style="carto-positron",
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    height=700,
    coloraxis_colorbar={
        "title": "FCS<br>Prevalence",
        "tickformat": ".0%"
    }
)

# Improve animation settings
fig_map.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
fig_map.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

# --------------------------------------------
# PROPHET FORECASTING
# --------------------------------------------

def prepare_for_prophet(df, date_col, value_col):
    """
    Prepare data for Prophet forecasting algorithm
    Prophet requires columns named 'ds' (date) and 'y' (value)
    Input: df (dataframe), date_col (str), value_col (str)
    Output: dataframe with 'ds' and 'y' columns
    """
    df_prophet = df[[date_col, value_col]].copy()
    df_prophet.columns = ['ds', 'y']
    return df_prophet
def forecast_with_prophet(df_prophet, periods=30):
    """
    Forecast future values using Facebook Prophet algorithm
    Input: df_prophet (dataframe with 'ds' and 'y'), periods (int) - days to predict
    Output: model (Prophet object), prediction (dataframe)
    """
    # Create model with 95% confidence interval
    model = Prophet(interval_width=0.95)
    
    # Train model
    model.fit(df_prophet)
    
    # Create future dates
    future_dates = model.make_future_dataframe(periods=periods, freq='D')
    
    # Predict
    prediction = model.predict(future_dates)
    
    return model, prediction


def visualize_forecast(df_original, prediction, title="Forecast"):
    """
    Visualize original time series + predictions with confidence interval
    Input : dataframe and the prediction done
    Output : graph a the forecast visualized
    """
    fig = go.Figure()

    # Observed data
    fig.add_trace(go.Scatter(
        x=df_original['ds'], y=df_original['y'],
        mode='lines', name='Observed data',
        line=dict(color='blue', width=2)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=prediction['ds'], y=prediction['yhat'],
        mode='lines', name='Forecast',
        line=dict(color='red', dash='dash', width=2)
    ))

    # Upper bound (invisible)
    fig.add_trace(go.Scatter(
        x=prediction['ds'], y=prediction['yhat_upper'],
        mode='lines', line=dict(width=0), showlegend=False
    ))

    # Lower bound + fill
    fig.add_trace(go.Scatter(
        x=prediction['ds'], y=prediction['yhat_lower'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255,0,0,0.2)',
        name='Confidence interval (95%)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of people (FCS)",
        hovermode='x unified',
        height=500,
        template="plotly_white"
    )
    return fig


# Prepare and create forecast
try:
    df_prophet = prepare_for_prophet(df_grouped, date_col="date", value_col="fcs people")
    model, prediction = forecast_with_prophet(df_prophet, periods=30)
    
    if model is not None and prediction is not None:
        fig_forecast = visualize_forecast(
            df_prophet, 
            prediction, 
            title="FCS People Forecast - 30 Days Prediction"
        )
    else:
        fig_forecast = None
except Exception as e:
    print(f"Forecast generation failed: {e}")
    fig_forecast = None

# --------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------

def create_stat_card(title, value, change, icon, color):
    """Create a beautiful statistic card"""
    change_color = "success" if change < 0 else "danger"
    change_icon = "↓" if change < 0 else "↑"
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas fa-{icon} fa-2x", style={"color": color}),
            ], style={"position": "absolute", "right": "20px", "top": "20px", "opacity": "0.3"}),
            html.H6(title, className="text-muted mb-2", style={"fontSize": "0.9rem"}),
            html.H3(f"{value:,.0f}", className="mb-1", style={"fontWeight": "bold", "color": "#2c3e50"}),
            html.Small([
                html.Span(f"{change_icon} {abs(change):.1f}%", className=f"text-{change_color}"),
                html.Span(" from previous period", className="text-muted")
            ])
        ])
    ], className="shadow-sm mb-4", style={"borderLeft": f"4px solid {color}", "height": "100%"})

def create_info_box(title, content, color="#3b82f6"):
    """Create an information box"""
    return dbc.Alert([
        html.H5([
            html.I(className="fas fa-info-circle me-2"),
            title
        ], className="alert-heading"),
        html.P(content, className="mb-0")
    ], color="light", className="border-start border-4", style={"borderColor": color + "!important"})

# --------------------------------------------
# APP INITIALIZATION
# --------------------------------------------
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)
server = app.server

# --------------------------------------------
# CUSTOM CSS VIA MARKDOWN
# --------------------------------------------
css_injection = dcc.Markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif !important;
        background-color: #f8f9fa !important;
    }
    
    .sidebar {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important;
        min-height: 100vh !important;
        padding: 2rem 1rem !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1) !important;
    }
    
    .nav-link {
        color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    
    .nav-link:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        transform: translateX(5px);
    }
    
    .nav-link.active {
        background-color: #3b82f6 !important;
        color: white !important;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3) !important;
    }
    
    .card {
        border: none !important;
        border-radius: 12px !important;
        transition: transform 0.2s !important;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    .main-header {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        padding: 3rem 2rem !important;
        border-radius: 16px !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3) !important;
    }
    
    .content-section {
        background: white !important;
        padding: 2rem !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
        margin-bottom: 2rem !important;
    }
    </style>
    """,
    dangerously_allow_html=True
)

# --------------------------------------------
# LAYOUT COMPONENTS
# --------------------------------------------

# Sidebar
sidebar = html.Div([
    html.Div([
        html.H3("WFP Dashboard", style={"color": "white", "fontWeight": "bold", "marginBottom": "0.5rem"}),
        html.P("Madagascar 2024", style={"color": "rgba(255,255,255,0.6)", "fontSize": "0.9rem"})
    ], style={"marginBottom": "2rem"}),
    
    dbc.Nav([
        dbc.NavLink([
            html.I(className="fas fa-home me-2"),
            "Introduction"
        ], href="/", active="exact"),
        
        dbc.NavLink([
            html.I(className="fas fa-chart-line me-2"),
            "Overview"
        ], href="/overview", active="exact"),
        
        dbc.NavLink([
            html.I(className="fas fa-utensils me-2"),
            "FCS Indicator"
        ], href="/fcs", active="exact"),
        
        dbc.NavLink([
            html.I(className="fas fa-heartbeat me-2"),
            "RCSI Indicator"
        ], href="/rcsi", active="exact"),
        
        dbc.NavLink([
            html.I(className="fas fa-hospital me-2"),
            "Health Access"
        ], href="/health", active="exact"),
        
        dbc.NavLink([
            html.I(className="fas fa-store me-2"),
            "Market Access"
        ], href="/market", active="exact"),
        
        dbc.NavLink([
            html.I(className="fas fa-map me-2"),
            "Regional Analysis"
        ], href="/regional", active="exact"),
        
        dbc.NavLink([
            html.I(className="fas fa-chart-area me-2"),
            "Dynamic Treemap"
        ], href="/treemap", active="exact"),
        
        dbc.NavLink([
            html.I(className="fas fa-map-marked-alt me-2"),
            "Geographic Map"
        ], href="/map", active="exact"),
    ], vertical=True, pills=True),
    
    html.Hr(style={"borderColor": "rgba(255,255,255,0.2)", "margin": "2rem 0"}),
    
    html.Div([
        html.P("© 2024 WFP Team", style={"color": "rgba(255,255,255,0.5)", "fontSize": "0.75rem", "marginBottom": "0.25rem"}),
        html.P("Data: Jan-Mar 2024", style={"color": "rgba(255,255,255,0.5)", "fontSize": "0.75rem"})
    ])
], className="sidebar")

# Main content area
content = html.Div(id="page-content", style={"padding": "2rem", "minHeight": "100vh"})

# Main layout
app.layout = html.Div([
    css_injection,
    dcc.Location(id="url"),
    dbc.Row([
        dbc.Col(sidebar, width=2, style={"padding": 0}),
        dbc.Col(content, width=10)
    ], style={"margin": 0})
], style={"padding": 0})

# --------------------------------------------
# PAGE ROUTING
# --------------------------------------------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    
    # Calculate statistics
    fcs_change = ((latest_stats["fcs people"] - prev_stats["fcs people"]) / prev_stats["fcs people"]) * 100
    rcsi_change = ((latest_stats["rcsi people"] - prev_stats["rcsi people"]) / prev_stats["rcsi people"]) * 100
    health_change = ((latest_stats["health access people"] - prev_stats["health access people"]) / prev_stats["health access people"]) * 100
    market_change = ((latest_stats["market access people"] - prev_stats["market access people"]) / prev_stats["market access people"]) * 100
    
    # ========== INTRODUCTION PAGE ==========
    if pathname == "/" or pathname is None:
        return html.Div([
            # Hero section
            html.Div([
                html.H1("WFP HungerMap — Madagascar", 
                       style={"fontSize": "3rem", "fontWeight": "bold", "marginBottom": "1rem"}),
                html.H4("Food Security Monitoring Dashboard", 
                       style={"opacity": "0.9", "fontWeight": "normal"}),
                html.P(f"Latest data: {latest_date.strftime('%B %d, %Y')}", 
                      style={"marginTop": "1rem", "fontSize": "1.1rem"})
            ], className="main-header"),
            
            # Statistics cards
            dbc.Row([
                dbc.Col(create_stat_card(
                    "FCS People",
                    latest_stats["fcs people"],
                    fcs_change,
                    "utensils",
                    "#3b82f6"
                ), width=3),
                dbc.Col(create_stat_card(
                    "RCSI People",
                    latest_stats["rcsi people"],
                    rcsi_change,
                    "heartbeat",
                    "#8b5cf6"
                ), width=3),
                dbc.Col(create_stat_card(
                    "Health Access Limited",
                    latest_stats["health access people"],
                    health_change,
                    "hospital",
                    "#ec4899"
                ), width=3),
                dbc.Col(create_stat_card(
                    "Market Access Limited",
                    latest_stats["market access people"],
                    market_change,
                    "store",
                    "#f59e0b"
                ), width=3),
            ], className="mb-4"),
            
            # Project description
            html.Div([
                html.H3("About This Dashboard", className="mb-4", style={"color": "#2c3e50"}),
                html.P([
                    "This interactive dashboard provides comprehensive insights into ",
                    html.Strong("food security challenges in Madagascar"),
                    ", utilizing data from the World Food Programme's HungerMap initiative. "
                    "Our analysis combines temporal trends, spatial clustering, and predictive modeling "
                    "to support evidence-based decision-making for humanitarian interventions."
                ], className="lead"),
                
                dbc.Row([
                    dbc.Col([
                        html.H5([
                            html.I(className="fas fa-users text-primary me-2"),
                            "Team Members"
                        ]),
                        html.Ul([
                            html.Li("Marya LATIF"),
                            html.Li("Raphaël MARQUES ARAUJO"),
                            html.Li("Faty LO"),
                            html.Li("François-Louis Legland"),
                            html.Li("Philippine LHOMME")
                        ])
                    ], width=6),
                    dbc.Col([
                        html.H5([
                            html.I(className="fas fa-database text-primary me-2"),
                            "Dataset"
                        ]),
                        html.P("WFP HungerMap - Madagascar"),
                        html.P(["Period: ", html.Strong("January - March 2024")]),
                        html.P(["Source: ", html.A("WFP HungerMap", href="https://hungermap.wfp.org/", target="_blank")])
                    ], width=6)
                ], className="mt-4")
            ], className="content-section"),
            
            # Key indicators
            html.Div([
                html.H3("Key Indicators", className="mb-4", style={"color": "#2c3e50"}),
                dbc.Row([
                    dbc.Col([
                        create_info_box(
                            "Food Consumption Score (FCS)",
                            "Measures dietary diversity, food frequency, and nutritional value. "
                            "Tracks the number of people with insufficient food consumption.",
                            "#3b82f6"
                        )
                    ], width=6),
                    dbc.Col([
                        create_info_box(
                            "Reduced Coping Strategy Index (RCSI)",
                            "Monitors negative coping behaviors adopted by households facing food shortages. "
                            "Higher values indicate greater stress.",
                            "#8b5cf6"
                        )
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        create_info_box(
                            "Health Access Constraints",
                            "Measures the population facing barriers to accessing essential healthcare services. "
                            "Critical for understanding food security impacts.",
                            "#ec4899"
                        )
                    ], width=6),
                    dbc.Col([
                        create_info_box(
                            "Market Access Constraints",
                            "Tracks physical and economic barriers to accessing food markets. "
                            "Essential for supply chain interventions.",
                            "#f59e0b"
                        )
                    ], width=6)
                ])
            ], className="content-section")
        ])
    
    # ========== OVERVIEW PAGE ==========
    elif pathname == "/overview":
        return html.Div([
            html.Div([
                html.H1("Dashboard Overview", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
                html.P("Comprehensive view of all food security indicators")
            ], className="main-header"),
            
            html.Div([
                dcc.Graph(figure=fig_combined, config={'displayModeBar': False})
            ], className="content-section"),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Temporal Trends", className="mb-3"),
                        html.P(
                            "The combined view shows all four indicators evolving over time. "
                            "Notable patterns include seasonal variations and correlation between indicators, "
                            "suggesting common underlying drivers of food insecurity.",
                            className="text-muted"
                        )
                    ], className="content-section")
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.H4("Key Insights", className="mb-3"),
                        html.Ul([
                            html.Li("FCS and RCSI show strong correlation"),
                            html.Li("Health access constraints remain relatively stable"),
                            html.Li("Market access shows seasonal fluctuations"),
                            html.Li("Overall trend indicates persistent vulnerability")
                        ], className="text-muted")
                    ], className="content-section")
                ], width=6)
            ])
        ])
    
    # ========== FCS PAGE ==========
    elif pathname == "/fcs":
        return html.Div([
            html.Div([
                html.H1("Food Consumption Score (FCS)", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
                html.P("Indicator 1 — Measuring dietary diversity and food security")
            ], className="main-header"),
            
            dbc.Row([
                dbc.Col([
                    create_stat_card(
                        "Current FCS People",
                        latest_stats["fcs people"],
                        fcs_change,
                        "utensils",
                        "#3b82f6"
                    )
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H6("What is FCS?", className="text-muted mb-2"),
                        html.P(
                            "The Food Consumption Score is a composite indicator that measures "
                            "dietary diversity, food frequency, and relative nutritional importance "
                            "of different food groups. Higher scores indicate better food security.",
                            style={"fontSize": "0.9rem"}
                        )
                    ], className="content-section", style={"height": "100%"})
                ], width=8)
            ]),
            
            html.Div([
                html.H4("Historical Trend", className="mb-3"),
                dcc.Graph(figure=fig_fcs, config={'displayModeBar': False})
            ], className="content-section"),
            
            # Forecast section
            html.Div([
                html.H4([
                    html.I(className="fas fa-crystal-ball me-2 text-primary"),
                    "30-Day Forecast"
                ], className="mb-3"),
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    html.Strong("Predictive Analytics: "),
                    "This forecast uses Prophet, a time series forecasting model developed by Meta. "
                    "The prediction shows expected FCS people numbers for the next 30 days based on historical patterns."
                ], color="info", className="mb-3"),
                
                dcc.Graph(figure=fig_forecast, config={'displayModeBar': False}) if fig_forecast else html.Div([
                    dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Strong("Forecast unavailable: "),
                        "Prophet library not installed. Install with: ",
                        html.Code("pip install prophet")
                    ], color="warning")
                ])
            ], className="content-section"),
            
            html.Div([
                html.H4("Interpretation", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.H6([
                            html.I(className="fas fa-chart-line text-primary me-2"),
                            "Historical Trend"
                        ]),
                        html.P(
                            "The FCS trend shows the number of people with insufficient food consumption over time. "
                            "Peaks may correlate with seasonal factors, economic shocks, or climate events. "
                            "This indicator is crucial for targeting food assistance programs.",
                            className="text-muted"
                        )
                    ], width=6),
                    dbc.Col([
                        html.H6([
                            html.I(className="fas fa-chart-area text-success me-2"),
                            "Forecast Insights"
                        ]),
                        html.P(
                            "The forecast helps anticipate future trends and prepare interventions. "
                            "The confidence interval (shaded area) shows the range of likely outcomes. "
                            "Wider intervals indicate greater uncertainty in predictions.",
                            className="text-muted"
                        )
                    ], width=6)
                ])
            ], className="content-section")
        ])
    
    # ========== RCSI PAGE ==========
    elif pathname == "/rcsi":
        return html.Div([
            html.Div([
                html.H1("Reduced Coping Strategy Index (RCSI)", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
                html.P("Indicator 2 — Monitoring household coping behaviors")
            ], className="main-header"),
            
            dbc.Row([
                dbc.Col([
                    create_stat_card(
                        "Current RCSI People",
                        latest_stats["rcsi people"],
                        rcsi_change,
                        "heartbeat",
                        "#8b5cf6"
                    )
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H6("What is RCSI?", className="text-muted mb-2"),
                        html.P(
                            "The Reduced Coping Strategy Index tracks negative coping behaviors "
                            "households adopt when facing food shortages, such as reducing meal portions, "
                            "borrowing food, or limiting adult consumption to feed children.",
                            style={"fontSize": "0.9rem"}
                        )
                    ], className="content-section", style={"height": "100%"})
                ], width=8)
            ]),
            
            html.Div([
                dcc.Graph(figure=fig_rcsi, config={'displayModeBar': False})
            ], className="content-section"),
            
            html.Div([
                html.H4("Interpretation", className="mb-3"),
                html.P(
                    "High RCSI values indicate that households are employing stress-based coping strategies, "
                    "which can have long-term negative impacts on health and well-being. "
                    "This indicator helps identify populations under acute stress.",
                    className="text-muted"
                )
            ], className="content-section")
        ])
    
    # ========== HEALTH ACCESS PAGE ==========
    elif pathname == "/health":
        return html.Div([
            html.Div([
                html.H1("Health Access Constraints", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
                html.P("Indicator 3 — Monitoring healthcare accessibility")
            ], className="main-header"),
            
            dbc.Row([
                dbc.Col([
                    create_stat_card(
                        "People with Limited Access",
                        latest_stats["health access people"],
                        health_change,
                        "hospital",
                        "#ec4899"
                    )
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H6("Why Health Access Matters", className="text-muted mb-2"),
                        html.P(
                            "Limited access to healthcare services exacerbates food insecurity impacts. "
                            "Malnutrition and health issues are interconnected, making this indicator "
                            "essential for holistic intervention planning.",
                            style={"fontSize": "0.9rem"}
                        )
                    ], className="content-section", style={"height": "100%"})
                ], width=8)
            ]),
            
            html.Div([
                dcc.Graph(figure=fig_health, config={'displayModeBar': False})
            ], className="content-section"),
            
            html.Div([
                html.H4("Interpretation", className="mb-3"),
                html.P(
                    "Health access constraints can amplify the effects of food insecurity, "
                    "particularly for vulnerable groups like children and pregnant women. "
                    "This metric helps coordinate health and nutrition interventions.",
                    className="text-muted"
                )
            ], className="content-section")
        ])
    
    # ========== MARKET ACCESS PAGE ==========
    elif pathname == "/market":
        return html.Div([
            html.Div([
                html.H1("Market Access Constraints", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
                html.P("Indicator 4 — Measuring food market accessibility")
            ], className="main-header"),
            
            dbc.Row([
                dbc.Col([
                    create_stat_card(
                        "People with Limited Access",
                        latest_stats["market access people"],
                        market_change,
                        "store",
                        "#f59e0b"
                    )
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H6("Why Market Access Matters", className="text-muted mb-2"),
                        html.P(
                            "Physical and economic barriers to food markets directly impact household "
                            "food security. This indicator captures both geographic isolation and "
                            "purchasing power constraints.",
                            style={"fontSize": "0.9rem"}
                        )
                    ], className="content-section", style={"height": "100%"})
                ], width=8)
            ]),
            
            html.Div([
                dcc.Graph(figure=fig_market, config={'displayModeBar': False})
            ], className="content-section"),
            
            html.Div([
                html.H4("Interpretation", className="mb-3"),
                html.P(
                    "Market access constraints highlight structural barriers to food security. "
                    "Addressing these issues requires infrastructure development, economic support, "
                    "and market strengthening interventions.",
                    className="text-muted"
                )
            ], className="content-section")
        ])
    
    # ========== REGIONAL ANALYSIS PAGE ==========
    elif pathname == "/regional":
        return html.Div([
            html.Div([
                html.H1("Regional Analysis", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
                html.P("Spatial patterns and regional vulnerabilities")
            ], className="main-header"),
            
            html.Div([
                html.H4("FCS Prevalence by Region", className="mb-3"),
                dcc.Graph(figure=fig_regional, config={'displayModeBar': False})
            ], className="content-section"),
            
            html.Div([
                html.H4("Regional Correlation Analysis", className="mb-3"),
                dcc.Graph(figure=fig_scatter, config={'displayModeBar': False})
            ], className="content-section"),
            
            html.Div([
                html.H4("Key Findings", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.H6("High-Risk Regions", className="text-danger"),
                        html.P(
                            "Regions with elevated FCS prevalence require prioritized interventions. "
                            "The correlation between FCS and RCSI suggests common vulnerability drivers.",
                            className="text-muted"
                        )
                    ], width=6),
                    dbc.Col([
                        html.H6("Geographic Patterns", className="text-primary"),
                        html.P(
                            "Spatial clustering indicates that food insecurity is not uniformly distributed. "
                            "Regional targeting can improve intervention efficiency.",
                            className="text-muted"
                        )
                    ], width=6)
                ])
            ], className="content-section")
        ])
    
    # ========== DYNAMIC TREEMAP PAGE ==========
    elif pathname == "/treemap":
        return html.Div([
            html.Div([
                html.H1("Dynamic Treemap Analysis", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
                html.P("Interactive visualization of FCS People by region over time")
            ], className="main-header"),
            
            html.Div([
                html.H4("How to Use This Visualization", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-play-circle fa-2x text-primary mb-2"),
                            html.H6("Play Animation", className="mt-2"),
                            html.P("Click the Play button to watch how FCS people evolve across regions over time.", 
                                   className="text-muted small")
                        ], className="text-center")
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-sliders-h fa-2x text-success mb-2"),
                            html.H6("Slider Control", className="mt-2"),
                            html.P("Use the time slider to manually select specific dates and explore the data.", 
                                   className="text-muted small")
                        ], className="text-center")
                    ], width=4),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-mouse-pointer fa-2x text-info mb-2"),
                            html.H6("Hover Details", className="mt-2"),
                            html.P("Hover over regions to see exact values and detailed information.", 
                                   className="text-muted small")
                        ], className="text-center")
                    ], width=4)
                ])
            ], className="content-section"),
            
            html.Div([
                dcc.Graph(figure=fig_dynamic, config={'displayModeBar': True})
            ], className="content-section"),
            
            html.Div([
                html.H4("Key Insights", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-chart-line text-primary me-2"),
                                "Temporal Patterns"
                            ]),
                            html.P(
                                "The treemap animation reveals how food insecurity evolves across different regions. "
                                "Larger blocks indicate regions with higher numbers of people affected by insufficient food consumption.",
                                className="text-muted"
                            )
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-layer-group text-success me-2"),
                                "Regional Comparison"
                            ]),
                            html.P(
                                "The size of each block is proportional to the FCS people count. Colors range from green (lower values) "
                                "to red (higher values), making it easy to identify the most vulnerable regions at any given time.",
                                className="text-muted"
                            )
                        ])
                    ], width=6)
                ])
            ], className="content-section"),
            
            dbc.Alert([
                html.I(className="fas fa-lightbulb me-2"),
                html.Strong("Pro Tip: "),
                "Use the animation to identify regions that show sudden changes or persistent vulnerability over time. "
                "This can help prioritize intervention areas."
            ], color="info", className="mb-0")
        ])
    
    # ========== GEOGRAPHIC MAP PAGE ==========
    elif pathname == "/map":
        return html.Div([
            html.Div([
                html.H1("Geographic Map Visualization", style={"fontSize": "2.5rem", "fontWeight": "bold"}),
                html.P("Interactive map of FCS Prevalence across Madagascar")
            ], className="main-header"),
            
            html.Div([
                html.H4("Understanding the Map", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-circle fa-2x mb-2", style={"color": "#dc3545"}),
                            html.H6("Circle Size", className="mt-2"),
                            html.P("Represents the population size of each region. Larger circles = more people.", 
                                   className="text-muted small")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-palette fa-2x text-warning mb-2"),
                            html.H6("Color Scale", className="mt-2"),
                            html.P("Red indicates high FCS prevalence (more food insecurity), green indicates lower prevalence.", 
                                   className="text-muted small")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-play-circle fa-2x text-success mb-2"),
                            html.H6("Time Animation", className="mt-2"),
                            html.P("Use the play button to see how food security changes over time across regions.", 
                                   className="text-muted small")
                        ], className="text-center")
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-search-plus fa-2x text-info mb-2"),
                            html.H6("Interactive Zoom", className="mt-2"),
                            html.P("Click and drag to pan, scroll to zoom in/out on specific areas of Madagascar.", 
                                   className="text-muted small")
                        ], className="text-center")
                    ], width=3)
                ])
            ], className="content-section"),
            
            html.Div([
                dcc.Graph(figure=fig_map, config={'displayModeBar': True})
            ], className="content-section"),
            
            html.Div([
                html.H4("Geographic Insights", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-map-pin text-danger me-2"),
                                "Hotspot Identification"
                            ]),
                            html.P(
                                "The map clearly shows geographic patterns of food insecurity. Red zones indicate areas "
                                "requiring immediate attention and intervention. Watch how these hotspots shift over time "
                                "to understand seasonal or chronic vulnerability patterns.",
                                className="text-muted"
                            )
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-route text-primary me-2"),
                                "Spatial Clustering"
                            ]),
                            html.P(
                                "Food insecurity often clusters in neighboring regions due to shared environmental, economic, "
                                "or infrastructure challenges. The map helps identify these clusters for coordinated regional "
                                "intervention strategies.",
                                className="text-muted"
                            )
                        ])
                    ], width=6)
                ]),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-users text-success me-2"),
                                "Population Context"
                            ]),
                            html.P(
                                "Larger circles indicate higher population centers. This helps prioritize interventions "
                                "where more people are affected, balancing severity (color) with scale (size).",
                                className="text-muted"
                            )
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-clock text-warning me-2"),
                                "Temporal Evolution"
                            ]),
                            html.P(
                                "Use the time slider to track changes. Sudden color shifts may indicate crisis events, "
                                "while gradual changes suggest seasonal patterns or long-term trends that require different "
                                "intervention approaches.",
                                className="text-muted"
                            )
                        ])
                    ], width=6)
                ])
            ], className="content-section"),
            
            dbc.Alert([
                html.I(className="fas fa-compass me-2"),
                html.Strong("Navigation Tip: "),
                "Double-click on the map to reset the view. Use the controls at the top-right to pan, zoom, "
                "or return to the default view. The animation controls at the bottom let you play through time or "
                "select specific dates."
            ], color="primary", className="mb-0")
        ])
    
    # Default fallback
    return html.Div([
        html.H1("Page not found", className="text-center mt-5"),
        html.P("Please use the sidebar to navigate.", className="text-center text-muted")
    ])

# --------------------------------------------
# RUN APPLICATION
# --------------------------------------------
if __name__ == "__main__":
    # Pour ouvrir dans le navigateur (pas dans Jupyter)
    app.run(debug=True, host='127.0.0.1', port=8050)