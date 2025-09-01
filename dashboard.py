
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import load_all_data
from ai_engine import run_complete_analysis

# Load data and run AI analysis
print("🔄 Loading data and running AI analysis...")
retail_df, reviews_df, trends_df = load_all_data()
ai_engine, ai_results = run_complete_analysis(retail_df, reviews_df)

# Initialize Dash app with modern theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
server = app.server
app.title = "AI Market Trend Analysis Dashboard"

# Define colors and styling
COLORS = {
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#51cf66',
    'warning': '#ffd43b',
    'danger': '#ff6b6b',
    'dark': '#212529',
    'light': '#f8f9fa'
}

# Helper functions for KPI cards
def create_kpi_card(title, value, icon, color="primary"):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className=f"fas {icon} fa-2x mb-2"),
                    html.H4(value, className="card-title mb-0"),
                    html.P(title, className="card-text")
                ], className="text-center")
            ])
        ])
    ], color=color, outline=True, className="shadow-sm h-100")

def create_metric_cards():
    '''Create KPI metric cards'''
    total_revenue = f"${retail_df['TotalPrice'].sum():,.0f}"
    total_orders = f"{retail_df['InvoiceNo'].nunique():,}"
    total_customers = f"{retail_df['CustomerID'].nunique():,}"
    avg_order_value = f"${retail_df.groupby('InvoiceNo')['TotalPrice'].sum().mean():.2f}"

    # Sentiment metrics
    if 'sentiment' in ai_results:
        sentiment_dist = ai_results['sentiment']['distribution']
        positive_pct = f"{sentiment_dist.get('Positive', 0)*100:.1f}%"
    else:
        positive_pct = "N/A"

    return dbc.Row([
        dbc.Col([create_kpi_card("Total Revenue", total_revenue, "fa-dollar-sign", "success")], md=2),
        dbc.Col([create_kpi_card("Total Orders", total_orders, "fa-shopping-cart", "info")], md=2),
        dbc.Col([create_kpi_card("Customers", total_customers, "fa-users", "warning")], md=2),
        dbc.Col([create_kpi_card("Avg Order Value", avg_order_value, "fa-chart-line", "primary")], md=2),
        dbc.Col([create_kpi_card("Positive Sentiment", positive_pct, "fa-smile", "success")], md=2),
        dbc.Col([create_kpi_card("Active Products", f"{retail_df['Description'].nunique()}", "fa-box", "secondary")], md=2)
    ], className="mb-4")

# Sales & Forecasting Tab
def create_sales_tab():
    # Get top products for dropdown
    top_products = retail_df['Description'].value_counts().head(20).index.tolist()

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("📈 Sales Forecasting", className="mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Product:", className="fw-bold"),
                                dcc.Dropdown(
                                    id="product-dropdown",
                                    options=[{"label": prod, "value": prod} for prod in top_products],
                                    value=top_products[0],
                                    style={"color": "black"}
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Forecast Period:", className="fw-bold"),
                                dcc.Dropdown(
                                    id="forecast-period",
                                    options=[
                                        {"label": "30 Days", "value": 30},
                                        {"label": "60 Days", "value": 60},
                                        {"label": "90 Days", "value": 90}
                                    ],
                                    value=90,
                                    style={"color": "black"}
                                )
                            ], md=6)
                        ])
                    ])
                ], className="mb-4"),

                dbc.Card([
                    dbc.CardBody([
                        dcc.Loading([
                            dcc.Graph(id="sales-forecast-chart")
                        ])
                    ])
                ]),

                # Trending products section
                html.Div(id="trending-products", className="mt-4")

            ], md=12)
        ])
    ])

# Customer Segmentation Tab
def create_customer_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("👥 Customer Segmentation", className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("RFM Analysis"),
                            dbc.CardBody([
                                dcc.Graph(id="customer-segments-scatter")
                            ])
                        ])
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Segment Distribution"),
                            dbc.CardBody([
                                dcc.Graph(id="segment-distribution")
                            ])
                        ])
                    ], md=4)
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Customer Insights"),
                            dbc.CardBody([
                                html.Div(id="customer-insights")
                            ])
                        ], className="mt-4")
                    ], md=12)
                ])
            ])
        ])
    ])

# Sentiment Analysis Tab
def create_sentiment_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("💭 Sentiment Analysis", className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Overall Sentiment Distribution"),
                            dbc.CardBody([
                                dcc.Graph(id="sentiment-pie-chart")
                            ])
                        ])
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Sentiment Over Time"),
                            dbc.CardBody([
                                dcc.Graph(id="sentiment-timeline")
                            ])
                        ])
                    ], md=6)
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Sample Reviews"),
                            dbc.CardBody([
                                html.Div(id="reviews-table")
                            ])
                        ], className="mt-4")
                    ], md=12)
                ])
            ])
        ])
    ])

# Analytics & Insights Tab
def create_analytics_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("🔍 Advanced Analytics", className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Anomaly Detection"),
                            dbc.CardBody([
                                dcc.Graph(id="anomaly-detection-chart")
                            ])
                        ])
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Product Performance Prediction"),
                            dbc.CardBody([
                                dcc.Graph(id="product-performance-chart")
                            ])
                        ])
                    ], md=6)
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Market Trends (Google Trends)"),
                            dbc.CardBody([
                                dcc.Graph(id="google-trends-chart")
                            ])
                        ], className="mt-4")
                    ], md=12)
                ])
            ])
        ])
    ])

# Main layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🤖 AI-Powered Market Trend Analysis", 
                       className="display-4 text-center mb-0"),
                html.P("Advanced Analytics Dashboard for E-commerce Intelligence", 
                       className="lead text-center text-muted")
            ], className="text-center py-4")
        ])
    ]),

    # KPI Cards
    create_metric_cards(),

    # Main tabs
    dbc.Card([
        dbc.CardHeader([
            dbc.Tabs([
                dbc.Tab(label="📈 Sales & Forecasting", tab_id="sales"),
                dbc.Tab(label="👥 Customer Segments", tab_id="customers"),
                dbc.Tab(label="💭 Sentiment Analysis", tab_id="sentiment"),
                dbc.Tab(label="🔍 Advanced Analytics", tab_id="analytics")
            ], id="main-tabs", active_tab="sales")
        ]),
        dbc.CardBody([
            html.Div(id="tab-content")
        ])
    ])
], fluid=True, className="py-4")

# Callbacks for tab switching
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab")
)
def switch_tab(active_tab):
    if active_tab == "sales":
        return create_sales_tab()
    elif active_tab == "customers":
        return create_customer_tab()
    elif active_tab == "sentiment":
        return create_sentiment_tab()
    elif active_tab == "analytics":
        return create_analytics_tab()

# Sales forecasting callback
@app.callback(
    [Output("sales-forecast-chart", "figure"),
     Output("trending-products", "children")],
    [Input("product-dropdown", "value"),
     Input("forecast-period", "value")]
)
def update_sales_forecast(selected_product, forecast_period):
    # Generate forecast
    model, forecast = ai_engine.forecast_sales_prophet(retail_df, selected_product, forecast_period)

    if forecast is not None:
        # Create forecast chart
        fig = go.Figure()

        # Historical data
        historical_data = retail_df[retail_df['Description'].str.contains(selected_product, case=False, na=False)]
        daily_sales = historical_data.groupby('InvoiceDate')['Quantity'].sum().reset_index()

        fig.add_trace(go.Scatter(
            x=daily_sales['InvoiceDate'],
            y=daily_sales['Quantity'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color=COLORS['primary'])
        ))

        # Forecast
        forecast_data = forecast[forecast['ds'] > daily_sales['InvoiceDate'].max()]
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color=COLORS['success'], dash='dash')
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='Confidence Interval',
            fillcolor='rgba(0,100,80,0.2)'
        ))

        fig.update_layout(
            title=f"Sales Forecast: {selected_product}",
            xaxis_title="Date",
            yaxis_title="Quantity Sold",
            template="plotly_dark",
            height=500
        )
    else:
        # Fallback chart
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for forecasting", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", height=500)

    # Trending products
    if 'trends' in ai_results:
        trends_up = ai_results['trends']['up']
        trends_down = ai_results['trends']['down']

        trending_content = dbc.Card([
            dbc.CardHeader("📊 Product Trends"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("🔥 Trending Up", className="text-success"),
                        html.Ul([html.Li(f"{prod}: +{slope:.2f}") for prod, slope in list(trends_up.items())[:5]])
                    ], md=6),
                    dbc.Col([
                        html.H6("📉 Trending Down", className="text-danger"),
                        html.Ul([html.Li(f"{prod}: {slope:.2f}") for prod, slope in list(trends_down.items())[:5]])
                    ], md=6)
                ])
            ])
        ])
    else:
        trending_content = html.Div("Trend data not available")

    return fig, trending_content

# Customer segmentation callbacks
@app.callback(
    [Output("customer-segments-scatter", "figure"),
     Output("segment-distribution", "figure"),
     Output("customer-insights", "children")],
    Input("main-tabs", "active_tab")
)
def update_customer_analysis(active_tab):
    if active_tab != "customers":
        return {}, {}, ""

    if 'customer_segments' not in ai_results:
        return {}, {}, "Customer segmentation data not available"

    rfm_data = ai_results['customer_segments']

    # Scatter plot
    scatter_fig = px.scatter(
        rfm_data,
        x="Recency",
        y="Frequency", 
        color="Segment",
        size="Monetary",
        hover_data=["Monetary"],
        title="Customer Segments (RFM Analysis)",
        template="plotly_dark"
    )

    # Distribution chart
    segment_counts = rfm_data['Segment'].value_counts()
    dist_fig = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Segment Distribution",
        template="plotly_dark"
    )

    # Insights
    insights = []
    for segment in rfm_data['Segment'].unique():
        segment_data = rfm_data[rfm_data['Segment'] == segment]
        avg_monetary = segment_data['Monetary'].mean()
        count = len(segment_data)

        insights.append(
            dbc.Alert([
                html.H6(f"{segment} ({count} customers)", className="alert-heading"),
                html.P(f"Average spend: ${avg_monetary:,.2f}")
            ], color="info")
        )

    return scatter_fig, dist_fig, insights

# Sentiment analysis callbacks
@app.callback(
    [Output("sentiment-pie-chart", "figure"),
     Output("sentiment-timeline", "figure"),
     Output("reviews-table", "children")],
    Input("main-tabs", "active_tab")
)
def update_sentiment_analysis(active_tab):
    if active_tab != "sentiment" or 'sentiment' not in ai_results:
        return {}, {}, "Sentiment data not available"

    sentiment_data = ai_results['sentiment']['data']
    sentiment_dist = ai_results['sentiment']['distribution']

    # Pie chart
    pie_fig = px.pie(
        values=sentiment_dist.values,
        names=sentiment_dist.index,
        title="Overall Sentiment Distribution",
        color_discrete_map={
            'Positive': COLORS['success'],
            'Negative': COLORS['danger'],
            'Neutral': COLORS['warning']
        },
        template="plotly_dark"
    )

    # Timeline (simplified)
    timeline_data = sentiment_data.groupby([sentiment_data['review_date'].dt.date, 'sentiment']).size().unstack(fill_value=0)

    timeline_fig = go.Figure()
    for sentiment in timeline_data.columns:
        color = COLORS['success'] if sentiment == 'Positive' else COLORS['danger'] if sentiment == 'Negative' else COLORS['warning']
        timeline_fig.add_trace(go.Scatter(
            x=timeline_data.index,
            y=timeline_data[sentiment],
            name=sentiment,
            line=dict(color=color)
        ))

    timeline_fig.update_layout(
        title="Sentiment Trends Over Time",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Number of Reviews"
    )

    # Sample reviews table
    sample_reviews = sentiment_data.sample(min(10, len(sentiment_data)))
    table = dash_table.DataTable(
        data=sample_reviews[['review_body', 'sentiment', 'rating']].to_dict('records'),
        columns=[
            {"name": "Review", "id": "review_body"},
            {"name": "Sentiment", "id": "sentiment"},
            {"name": "Rating", "id": "rating"}
        ],
        style_cell={'textAlign': 'left', 'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        page_size=5
    )

    return pie_fig, timeline_fig, table

# Analytics callbacks
@app.callback(
    [Output("anomaly-detection-chart", "figure"),
     Output("product-performance-chart", "figure"),
     Output("google-trends-chart", "figure")],
    Input("main-tabs", "active_tab")
)
def update_analytics(active_tab):
    if active_tab != "analytics":
        return {}, {}, {}

    # Anomaly detection chart
    if 'anomalies' in ai_results:
        anomalies = ai_results['anomalies']

        anom_fig = px.scatter(
            x=anomalies.index,
            y=anomalies['anomaly_score'],
            color=anomalies['anomaly'].map({1: 'Normal', -1: 'Anomaly'}),
            title="Customer Anomaly Detection",
            template="plotly_dark"
        )
    else:
        anom_fig = go.Figure()
        anom_fig.add_annotation(text="Anomaly data not available", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        anom_fig.update_layout(template="plotly_dark")

    # Product performance prediction
    product_sales = retail_df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)

    perf_fig = px.bar(
        x=product_sales.values,
        y=product_sales.index,
        orientation='h',
        title="Top 10 Products by Revenue",
        template="plotly_dark"
    )

    # Google Trends
    trends_fig = go.Figure()
    for col in trends_df.columns[1:]:  # Skip date column
        trends_fig.add_trace(go.Scatter(
            x=trends_df['date'],
            y=trends_df[col],
            name=col,
            mode='lines'
        ))

    trends_fig.update_layout(
        title="Google Trends Data",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Search Interest"
    )

    return anom_fig, perf_fig, trends_fig

if __name__ == "__main__":
    print("\n🚀 Starting AI Market Trend Analysis Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    app.run_server(debug=True, host='127.0.0.1', port=8050)
