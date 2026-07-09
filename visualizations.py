"""
visualizations.py
Lightweight Plotly-based visualizations for the dashboard.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#00D4FF",
    "secondary": "#7B2FBE",
    "success":   "#00E676",
    "warning":   "#FFD600",
    "danger":    "#FF3D71",
    "bg":        "#0A0E1A",
    "card":      "#111827",
    "text":      "#E2E8F0",
    "grid":      "#1E293B",
}

GAUGE_STEPS_HEALTH = [
    {"range": [0, 30],  "color": "#FF3D71"},
    {"range": [30, 50], "color": "#FF9800"},
    {"range": [50, 70], "color": "#FFD600"},
    {"range": [70, 85], "color": "#00E676"},
    {"range": [85, 100],"color": "#00D4FF"},
]

GAUGE_STEPS_RUL = [
    {"range": [0, 0.2],  "color": "#FF3D71"},
    {"range": [0.2, 0.4],"color": "#FF9800"},
    {"range": [0.4, 0.6],"color": "#FFD600"},
    {"range": [0.6, 0.8],"color": "#00E676"},
    {"range": [0.8, 1.0],"color": "#00D4FF"},
]

_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def _apply_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["primary"])),
        **_PLOT_LAYOUT
    )
    return fig


# ── VHI Gauge ────────────────────────────────────────────────────────────────

def plot_vhi_gauge(vhi_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=vhi_score,
        number={"suffix": "%", "font": {"size": 36, "color": COLORS["primary"]}},
        title={"text": "Vehicle Health Index", "font": {"size": 14, "color": COLORS["text"]}},
        delta={"reference": 70, "increasing": {"color": COLORS["success"]},
               "decreasing": {"color": COLORS["danger"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["text"],
                     "tickfont": {"size": 10}},
            "bar": {"color": COLORS["primary"], "thickness": 0.25},
            "bgcolor": COLORS["card"],
            "bordercolor": COLORS["grid"],
            "steps": GAUGE_STEPS_HEALTH,
            "threshold": {
                "line": {"color": COLORS["warning"], "width": 3},
                "thickness": 0.8,
                "value": 70
            }
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=COLORS["text"]),
                      margin=dict(l=10, r=10, t=30, b=10),
                      height=250)
    return fig


# ── RUL Gauge ────────────────────────────────────────────────────────────────

def plot_rul_gauge(rul_km: int, max_km: int = 10_000) -> go.Figure:
    rul_km = max(0, int(rul_km or 0))
    rul_km = min(rul_km, 10_000)
    max_km = 10_000
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul_km,
        number={"suffix": " km", "font": {"size": 28, "color": COLORS["secondary"]}},
        title={"text": "RUL Before Critical Risk", "font": {"size": 14, "color": COLORS["text"]}},
        gauge={
            "axis": {"range": [0, max_km],
                     "tickformat": ",",
                     "tickfont": {"size": 9}},
            "bar": {"color": COLORS["secondary"], "thickness": 0.25},
            "bgcolor": COLORS["card"],
            "bordercolor": COLORS["grid"],
            "steps": [
                {"range": [0, max_km * 0.2],  "color": "#FF3D71"},
                {"range": [max_km * 0.2, max_km * 0.4], "color": "#FF9800"},
                {"range": [max_km * 0.4, max_km * 0.6], "color": "#FFD600"},
                {"range": [max_km * 0.6, max_km * 0.8], "color": "#00E676"},
                {"range": [max_km * 0.8, max_km], "color": "#00D4FF"},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=COLORS["text"]),
                      margin=dict(l=10, r=10, t=30, b=10),
                      height=250)
    return fig


# ── Component Risk Bar Chart ──────────────────────────────────────────────────

def plot_component_risk(systems: dict) -> go.Figure:
    names  = list(systems.keys())
    scores = [systems[n]["score"] for n in names]
    colors = []
    for s in scores:
        if s >= 75:   colors.append(COLORS["success"])
        elif s >= 55: colors.append(COLORS["warning"])
        elif s >= 35: colors.append("#FF9800")
        else:         colors.append(COLORS["danger"])

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}%" for s in scores],
        textposition="inside",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=True,
                   gridcolor=COLORS["grid"], tickformat=".0f"),
        yaxis=dict(autorange="reversed"),
        height=280,
        **_PLOT_LAYOUT,
    )
    fig.update_layout(title="Subsystem Health Scores")
    return fig


# ── SHAP Waterfall Chart ──────────────────────────────────────────────────────

def plot_shap_waterfall(shap_values, feature_names, top_n=10) -> go.Figure:
    if shap_values is None or feature_names is None:
        return _empty_figure("SHAP values unavailable.")

    pairs = sorted(zip(feature_names, shap_values),
                   key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [p[0].replace("_", " ").title() for p in pairs]
    vals  = [p[1] for p in pairs]
    colors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.add_vline(x=0, line_color=COLORS["text"], line_width=1)
    fig.update_layout(
        title="SHAP Feature Impact (Local Explanation)",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis=dict(autorange="reversed"),
        height=380,
        **_PLOT_LAYOUT,
    )
    return fig


# ── Feature Importance Chart ──────────────────────────────────────────────────

def plot_feature_importance(feature_names, importances, top_n=15) -> go.Figure:
    pairs = sorted(zip(feature_names, importances),
                   key=lambda x: x[1], reverse=True)[:top_n]
    names = [p[0].replace("_", " ").title() for p in reversed(pairs)]
    vals  = [p[1] for p in reversed(pairs)]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="Global Feature Importance (Random Forest)",
        xaxis_title="Importance Score",
        height=420,
        **_PLOT_LAYOUT,
    )
    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list) -> go.Figure:
    labels = ["No Maintenance", "Needs Maintenance"]
    z = np.array(cm)
    text = [[str(v) for v in row] for row in cm]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        colorscale=[[0, COLORS["card"]], [1, COLORS["primary"]]],
        showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=300,
        **_PLOT_LAYOUT,
    )
    return fig


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob) -> go.Figure:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC (AUC = {roc_auc:.3f})",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color=COLORS["grid"], width=1, dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1], gridcolor=COLORS["grid"]),
        yaxis=dict(range=[0, 1.02], gridcolor=COLORS["grid"]),
        height=320,
        legend=dict(x=0.6, y=0.1),
        **_PLOT_LAYOUT,
    )
    return fig


# ── Metrics Cards Bar ─────────────────────────────────────────────────────────

def plot_metrics_bar(metrics: dict) -> go.Figure:
    keys   = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    values = [metrics.get(k, 0) for k in keys]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"],
              COLORS["warning"], "#FF6B6B"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Model Performance Metrics",
        yaxis=dict(range=[0, 110], gridcolor=COLORS["grid"]),
        xaxis=dict(gridcolor=COLORS["grid"]),
        height=320,
        **_PLOT_LAYOUT,
    )
    return fig


# ── Helper ────────────────────────────────────────────────────────────────────

def _empty_figure(msg="No data available.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=COLORS["text"])
    )
    fig.update_layout(height=200, **_PLOT_LAYOUT)
    return fig
