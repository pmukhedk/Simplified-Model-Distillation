import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Load data
df = pd.read_csv("withrouge_recall_calculations_ahilan_weights.csv")
df.columns = df.columns.str.strip()

# Metrics & Models
numeric_metrics = [col for col in df.select_dtypes(include='number').columns if col != 'dataset_size']
model_names = df['model_name'].unique()

sbc_components = ["SBC-SemanticScore", "SBC-CompletenessScore", "SBC-CosineScore"]

def normalize_scores(data, metric_list):
    norm_df = data.copy()
    for metric in metric_list:
        max_val = data[metric].max()
        min_val = data[metric].min()
        norm_df[metric] = (data[metric] - min_val) / (max_val - min_val + 1e-8)
    return norm_df

# App init
app = dash.Dash(__name__)
app.title = "Summarization Model Evaluation Dashboard"

# Layout
app.layout = html.Div([
    html.H1("📊 Summarization Metrics Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Models:"),
        dcc.Dropdown(
            options=[{"label": model, "value": model} for model in model_names],
            value=list(model_names),
            multi=True,
            id="model-dropdown"
        ),
    ], id="model-dropdown-container", style={"width": "45%", "display": "inline-block"}),

    html.Div([
        html.Label("Select Metrics:"),
        dcc.Dropdown(
            options=[{"label": metric, "value": metric} for metric in numeric_metrics],
            value=["ROUGE-1-F1"],
            multi=True,
            id="metric-dropdown"
        ),
    ], id="metric-dropdown-container", style={"width": "45%", "display": "inline-block", "marginLeft": "5%"}),

    dcc.Tabs(id="tabs", value="line", children=[
        dcc.Tab(label="📈 Line Chart (Metric vs Dataset Size)", value="line"),
        dcc.Tab(label="📊 Grouped Bar (SBC Subscores)", value="bar"),
        dcc.Tab(label="🕸 Radar Chart (Model Profile)", value="radar"),
        dcc.Tab(label="🔥 Correlation Heatmap", value="heatmap"),
        dcc.Tab(label="⚡️ Metric Pairplot", value="pairplot"),
    ]),

    html.Div(id="tab-content"),

    html.Div(id="footer", children="Visualize summarization performance across metrics.", style={"marginTop": "20px"})
])

# Control visibility of dropdowns per tab
@app.callback(
    Output("metric-dropdown-container", "style"),
    Output("model-dropdown-container", "style"),
    Input("tabs", "value")
)
def toggle_dropdowns(tab_value):
    metric_visible = tab_value in ["line", "pairplot"]
    model_visible = tab_value in ["line", "bar", "radar", "pairplot"]

    metric_style = {"width": "45%", "display": "inline-block", "marginLeft": "5%"} if metric_visible else {"display": "none"}
    model_style = {"width": "45%", "display": "inline-block"} if model_visible else {"display": "none"}
    return metric_style, model_style

# Graph rendering logic
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("metric-dropdown", "value"),
    Input("model-dropdown", "value")
)
def update_tab(tab, selected_metrics, selected_models):
    filtered_df = df[df['model_name'].isin(selected_models)] if selected_models else df
    latest_df = filtered_df.sort_values('dataset_size').groupby('model_name').tail(1)

    if tab == "line":
        fig = go.Figure()
        for metric in selected_metrics:
            for model in selected_models:
                model_df = filtered_df[filtered_df['model_name'] == model].sort_values('dataset_size')
                fig.add_trace(go.Scatter(
                    x=model_df['dataset_size'],
                    y=model_df[metric],
                    mode='lines+markers',
                    name=f"{metric} | {model}"
                ))
        fig.update_layout(
            title="Selected Metrics vs Dataset Size",
            xaxis_title="Dataset Size",
            yaxis_title="Metric Value",
            hovermode="closest"
        )
        return dcc.Graph(figure=fig)

    elif tab == "bar":
        fig = go.Figure()
        for metric in sbc_components:
            fig.add_trace(go.Bar(
                x=latest_df['model_name'],
                y=latest_df[metric],
                name=metric
            ))
        fig.update_layout(
            barmode='group',
            title="SBC Component Scores (Latest Dataset Size)",
            xaxis_title="Model Name",
            yaxis_title="Score"
        )
        return dcc.Graph(figure=fig)

    elif tab == "radar":
        radar_metrics = ["ROUGE-1-F1", "ROUGE-L-F1", "BERTScore-F1", "SBC-Score", "CosineSimilarity"]
        norm_df = normalize_scores(latest_df, radar_metrics)
        fig = go.Figure()
        for _, row in norm_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in radar_metrics],
                theta=radar_metrics,
                fill='toself',
                name=row['model_name']
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar Chart: Model Profile Across Metrics"
        )
        return dcc.Graph(figure=fig)

    elif tab == "heatmap":
        corr = df[numeric_metrics].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Correlation Heatmap of Metrics")
        return dcc.Graph(figure=fig)

    elif tab == "pairplot":
        scatter_df = filtered_df[selected_metrics + ['model_name']].dropna()
        fig = px.scatter_matrix(
            scatter_df,
            dimensions=selected_metrics,
            color='model_name',
            title="Pairwise Scatter Plot of Selected Metrics"
        )
        return dcc.Graph(figure=fig)

    return html.Div("Invalid Tab Selected.")

# Run app
if __name__ == '__main__':
    app.run(debug=True)
