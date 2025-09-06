"""
Dashboard comparativo para análisis Bootstrap multi-escenario
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.constants import ESCENARIOS_ECONOMICOS


def crear_dashboard_bootstrap_comparativo(resultados_dict):
    """Dashboard comparativo específico para análisis Bootstrap"""

    escenarios = list(resultados_dict.keys())
    colores = ['#34495e', '#e74c3c', '#27ae60']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Bootstrap PCA Distributions',
            'Bootstrap Confidence Intervals',
            'Bias Correction Analysis',
            'Economic Models Impact (Bootstrap)'
        ]
    )

    # 1. Distribuciones Bootstrap PCA
    for i, escenario in enumerate(escenarios):
        resultados = resultados_dict[escenario]
        fig.add_trace(
            go.Violin(
                y=resultados['pca_values'],
                name=ESCENARIOS_ECONOMICOS[escenario]['nombre'],
                box_visible=True,
                meanline_visible=True,
                fillcolor=colores[i % len(colores)],
                opacity=0.6,
                line_color=colores[i % len(colores)]
            ),
            row=1, col=1
        )

    # 2. Intervalos de Confianza Bootstrap
    scenarios_names = []
    ci_lowers = []
    ci_uppers = []
    means = []

    for escenario in escenarios:
        if 'bootstrap_stats' in resultados_dict[escenario]:
            stats = resultados_dict[escenario]['bootstrap_stats']
            scenarios_names.append(ESCENARIOS_ECONOMICOS[escenario]['nombre'])
            ci_lowers.append(stats.get('pca_ci_lower', 0))
            ci_uppers.append(stats.get('pca_ci_upper', 0))
            means.append(stats.get('pca_mean', 0))

    if scenarios_names:
        fig.add_trace(
            go.Scatter(
                x=scenarios_names,
                y=means,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    arrayminus=[m - l for m, l in zip(means, ci_lowers)],
                    array=[u - m for u, m in zip(ci_uppers, means)]
                ),
                mode='markers',
                marker=dict(size=12, color=colores[:len(scenarios_names)]),
                name='95% Confidence Intervals'
            ),
            row=1, col=2
        )

    # 3. Análisis de Corrección de Sesgo
    if len(escenarios) > 0:
        bias_data = []
        for escenario in escenarios:
            if 'bootstrap_stats' in resultados_dict[escenario]:
                stats = resultados_dict[escenario]['bootstrap_stats']
                bootstrap_mean = stats.get('pca_mean', 0)
                bias_corrected = stats.get('bias_corrected_mean', 0)
                bias = bootstrap_mean - bias_corrected
                bias_data.append(bias)

        if bias_data:
            fig.add_trace(
                go.Bar(
                    x=[ESCENARIOS_ECONOMICOS[esc]['nombre']
                        for esc in escenarios],
                    y=bias_data,
                    name='Bootstrap Bias',
                    marker_color=colores[:len(escenarios)]
                ),
                row=2, col=1
            )

    # 4. Impacto en Modelos Económicos (Keynes como ejemplo)
    if 'keynes' in resultados_dict[escenarios[0]]['modelos_externos']:
        keynes_impacts = []
        for escenario in escenarios:
            resultados = resultados_dict[escenario]
            original = resultados['modelos_externos']['keynes']['original']
            pca_enhanced = resultados['modelos_externos']['keynes']['con_pca']

            original_mean = np.mean(original)
            pca_mean = np.mean(pca_enhanced)
            impact = ((pca_mean - original_mean) / original_mean) * \
                100 if original_mean != 0 else 0
            keynes_impacts.append(impact)

        fig.add_trace(
            go.Bar(
                x=[ESCENARIOS_ECONOMICOS[esc]['nombre'] for esc in escenarios],
                y=keynes_impacts,
                name='Keynes Model Impact %',
                marker_color=colores[:len(escenarios)],
                text=[f'{imp:+.1f}%' for imp in keynes_impacts],
                textposition='auto'
            ),
            row=2, col=2
        )

    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Bootstrap Comparative Analysis Dashboard"
    )

    return fig


def crear_dashboard_metricas_bootstrap(resultados_dict):
    """Dashboard de métricas bootstrap detalladas"""

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Bootstrap Means by Scenario',
            'Standard Errors Comparison',
            'Confidence Interval Widths',
            'Bias Correction Impact',
            'Convergence Analysis',
            'Statistical Power'
        ]
    )

    escenarios = list(resultados_dict.keys())
    colores = ['#34495e', '#e74c3c', '#27ae60']

    # Extraer métricas
    means = []
    std_errors = []
    ci_widths = []
    biases = []
    scenario_names = []

    for escenario in escenarios:
        if 'bootstrap_stats' in resultados_dict[escenario]:
            stats = resultados_dict[escenario]['bootstrap_stats']
            scenario_names.append(ESCENARIOS_ECONOMICOS[escenario]['nombre'])
            means.append(stats.get('pca_mean', 0))
            std_errors.append(stats.get('bootstrap_se', 0))
            ci_width = stats.get('pca_ci_upper', 0) - \
                stats.get('pca_ci_lower', 0)
            ci_widths.append(ci_width)
            bias = stats.get('pca_mean', 0) - \
                stats.get('bias_corrected_mean', 0)
            biases.append(abs(bias))

    # 1. Bootstrap Means
    fig.add_trace(
        go.Bar(x=scenario_names, y=means, name='Bootstrap Means',
               marker_color=colores[:len(scenario_names)]),
        row=1, col=1
    )

    # 2. Standard Errors
    fig.add_trace(
        go.Bar(x=scenario_names, y=std_errors, name='Standard Errors',
               marker_color=colores[:len(scenario_names)]),
        row=1, col=2
    )

    # 3. CI Widths
    fig.add_trace(
        go.Bar(x=scenario_names, y=ci_widths, name='CI Widths',
               marker_color=colores[:len(scenario_names)]),
        row=1, col=3
    )

    # 4. Bias Impact
    fig.add_trace(
        go.Bar(x=scenario_names, y=biases, name='Absolute Bias',
               marker_color=colores[:len(scenario_names)]),
        row=2, col=1
    )

    fig.update_layout(height=800, showlegend=False)
    return fig
