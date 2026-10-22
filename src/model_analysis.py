"""Gráficos de diagnóstico de clasificación, independientes de Streamlit."""
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.interpretation import TEAL, CORAL, styled


def roc_plot(actual, probabilities):
    false_positive, true_positive, _ = roc_curve(actual, probabilities)
    fig = go.Figure(go.Scatter(x=false_positive, y=true_positive, mode="lines",
                              name=f"ROC-AUC: {roc_auc_score(actual, probabilities):.4f}",
                              line_color=TEAL))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar", mode="lines",
                            line=dict(color="#a5afb8", dash="dash")))
    fig.update_layout(height=340, xaxis_range=[0, 1], yaxis_range=[0, 1])
    return styled(fig, "Tasa de falsas alertas", "Sensibilidad (recall)")


def precision_recall_plot(actual, probabilities):
    precision, recall, _ = precision_recall_curve(actual, probabilities)
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line_color=TEAL,
                              name=f"Precisión media: {average_precision_score(actual, probabilities):.4f}"))
    prevalence = float(np.mean(actual))
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], name="Proporción de abandonos",
                            mode="lines", line=dict(color="#a5afb8", dash="dash")))
    fig.update_layout(height=340, xaxis_range=[0, 1], yaxis_range=[0, 1])
    return styled(fig, "Sensibilidad (recall)", "Precisión")


def probability_plot(actual, probabilities):
    actual, probabilities = np.asarray(actual), np.asarray(probabilities)
    fig = go.Figure()
    for value, label, color in [(0, "Permaneció", TEAL), (1, "Abandonó", CORAL)]:
        fig.add_trace(go.Histogram(x=probabilities[actual == value], name=label,
                                  xbins=dict(start=0, end=1.001, size=.05), marker_color=color, opacity=.7))
    fig.add_vline(x=.5, line_dash="dash", line_color="#526775")
    fig.update_layout(barmode="overlay", height=340, xaxis_range=[0, 1])
    return styled(fig, "Probabilidad estimada de abandono · umbral 0,50", "Clientes")
