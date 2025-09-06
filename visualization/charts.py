"""
Módulo de visualizaciones y gráficos para el simulador PCA
Contiene funciones para crear gráficos Bootstrap y diagnósticos
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as sp_stats
from config.constants import ESCENARIOS_ECONOMICOS


def crear_grafico_bootstrap_diagnostics(resultados):
    """Crea diagnósticos específicos del análisis Bootstrap"""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Bootstrap Distribution vs Normal',
            'Bias-Corrected Estimates',
            'Confidence Intervals',
            'Bootstrap Convergence'
        ]
    )

    pca_values = np.array(resultados['pca_values'])
    n_bootstrap = len(pca_values)

    # 1. Distribución Bootstrap vs Normal
    fig.add_trace(
        go.Histogram(
            x=pca_values,
            nbinsx=50,
            name='Bootstrap Distribution',
            opacity=0.7,
            marker_color='blue',
            showlegend=True
        ),
        row=1, col=1
    )

    # Overlay normal distribution
    x_norm = np.linspace(pca_values.min(), pca_values.max(), 100)
    y_norm = sp_stats.norm.pdf(x_norm, np.mean(pca_values), np.std(pca_values))
    y_norm_scaled = y_norm * len(pca_values) * \
        (pca_values.max() - pca_values.min()) / 50

    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm_scaled,
            mode='lines',
            name='Normal Approximation',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # 2. Estimaciones corregidas por sesgo
    if 'bootstrap_stats' in resultados:
        stats = resultados['bootstrap_stats']
        original_mean = stats.get('original_mean', 0)
        bootstrap_mean = stats['pca_mean']
        bias_corrected = stats['bias_corrected_mean']

        categories = ['Original', 'Bootstrap', 'Bias-Corrected']
        values = [original_mean, bootstrap_mean, bias_corrected]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                name='Estimates Comparison',
                marker_color=['green', 'orange', 'red'],
                showlegend=True
            ),
            row=1, col=2
        )

    # 3. Intervalos de confianza
    if 'bootstrap_stats' in resultados:
        stats = resultados['bootstrap_stats']
        bootstrap_mean = stats['pca_mean']
        ci_lower = stats['pca_ci_lower']
        ci_upper = stats['pca_ci_upper']

        fig.add_trace(
            go.Scatter(
                x=['Bootstrap Mean', 'CI Lower', 'CI Upper'],
                y=[bootstrap_mean, ci_lower, ci_upper],
                mode='markers+lines',
                name='Confidence Interval',
                marker=dict(size=10, color=['blue', 'red', 'red']),
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )

    # 4. Convergencia Bootstrap
    cumulative_means = np.cumsum(pca_values) / \
        np.arange(1, len(pca_values) + 1)
    sample_indices = np.arange(1, len(pca_values) + 1)

    fig.add_trace(
        go.Scatter(
            x=sample_indices[::50],  # Muestrear cada 50 puntos
            y=cumulative_means[::50],
            mode='lines',
            name='Cumulative Mean',
            line=dict(color='green', width=2)
        ),
        row=2, col=2
    )

    if 'bootstrap_stats' in resultados:
        bootstrap_mean = resultados['bootstrap_stats']['pca_mean']
        fig.add_hline(
            y=bootstrap_mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Final Mean: {bootstrap_mean:.4f}",
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Bootstrap Analysis Diagnostics"
    )

    return fig


def crear_grafico_distribucion_variables(resultados):
    """Crea gráfico de distribución de variables cognitivas"""

    variables = ['DH', 'CS', 'AV', 'SQ']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{var} Distribution' for var in variables]
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    colors = ['blue', 'red', 'green', 'orange']

    for i, var in enumerate(variables):
        row, col = positions[i]
        data = resultados['variables_cognitivas'][var]

        fig.add_trace(
            go.Histogram(
                x=data,
                name=f'{var} Distribution',
                marker_color=colors[i],
                opacity=0.7,
                nbinsx=30
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=600,
        title_text="Cognitive Variables Bootstrap Distributions"
    )

    return fig


def crear_grafico_correlaciones_bootstrap(resultados):
    """Crea matriz de correlaciones bootstrap"""

    # Construir matriz de datos
    data_matrix = np.column_stack([
        resultados['pse_values'],
        resultados['pca_values'],
        resultados['variables_cognitivas']['DH'],
        resultados['variables_cognitivas']['CS'],
        resultados['variables_cognitivas']['AV'],
        resultados['variables_cognitivas']['SQ']
    ])

    variables = ['PSE', 'PCA', 'DH', 'CS', 'AV', 'SQ']
    corr_matrix = np.corrcoef(data_matrix.T)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=variables,
        y=variables,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Bootstrap Correlation Matrix",
        height=500
    )

    return fig


def crear_grafico_efectos_economicos(resultados_modelos):
    """Crea gráfico de efectos en modelos económicos"""

    modelos = list(resultados_modelos.keys())
    impactos = []

    for modelo in modelos:
        original = np.array(resultados_modelos[modelo]['original'])
        pca_enhanced = np.array(resultados_modelos[modelo]['con_pca'])
        impacto_pct = np.mean((pca_enhanced - original) / original * 100)
        impactos.append(impacto_pct)

    fig = go.Figure(data=[
        go.Bar(
            x=modelos,
            y=impactos,
            marker_color=['blue' if x > 0 else 'red' for x in impactos],
            text=[f'{x:+.1f}%' for x in impactos],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Economic Models PCA Impact (%)",
        xaxis_title="Economic Model",
        yaxis_title="Impact Percentage",
        height=400
    )

    return fig
