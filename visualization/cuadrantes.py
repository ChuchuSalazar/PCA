"""
Análisis de cuadrantes optimizado para PCA vs PSE
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from config.constants import ESCENARIOS_ECONOMICOS


@st.cache_data(ttl=300)
def preparar_datos_cuadrantes_cache(resultados):
    """Prepara datos para cuadrantes con cache"""

    # Usar muestra más pequeña para optimizar
    n_total = len(resultados['pca_values'])
    n_muestra = min(300, n_total)
    indices = np.random.choice(n_total, n_muestra, replace=False)

    pca = np.array(resultados['pca_values'])[indices]
    pse = np.array(resultados['pse_values'])[indices]
    dh = np.array(resultados['variables_cognitivas']['DH'])[indices]
    cs = np.array(resultados['variables_cognitivas']['CS'])[indices]
    sq = np.array(resultados['variables_cognitivas']['SQ'])[indices]

    # Crear DataFrame optimizado
    df = pd.DataFrame({
        'PCA': pca,
        'PSE': pse,
        'DH': dh,
        'CS': cs,
        'SQ': sq
    })

    # Determinar sesgo dominante
    sesgos_abs = df[['SQ', 'CS', 'DH']].abs()
    df['sesgo_dominante'] = sesgos_abs.idxmax(axis=1)
    df['intensidad_dominante'] = sesgos_abs.max(axis=1)

    # Determinar cuadrante
    df['cuadrante'] = df.apply(lambda row:
                               'I' if row['PCA'] > 0 and row['PSE'] > 0 else
                               'II' if row['PCA'] > 0 and row['PSE'] <= 0 else
                               'III' if row['PCA'] <= 0 and row['PSE'] <= 0 else
                               'IV', axis=1
                               )

    return df


def crear_analisis_cuadrantes_completo(resultados_dict):
    """Análisis de cuadrantes optimizado"""

    if not resultados_dict:
        st.warning("No hay datos bootstrap disponibles")
        return

    st.markdown("### 📍 Análisis de Cuadrantes: PCA vs PSE por Sesgo Dominante")

    # Selector de escenario simplificado
    if len(resultados_dict) > 1:
        escenario = st.selectbox(
            "Selecciona escenario:",
            options=list(resultados_dict.keys()),
            format_func=lambda x: ESCENARIOS_ECONOMICOS[x]['nombre'],
            key="select_cuadrantes_escenario"
        )
    else:
        escenario = list(resultados_dict.keys())[0]

    # Información del escenario
    config = ESCENARIOS_ECONOMICOS[escenario]
    st.markdown(f"""
    <div style="background: {config['color']}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {config['color']};">
        <h4 style="color: {config['color']}; margin: 0;">{config['nombre']}</h4>
        <p style="margin: 0.5rem 0 0 0;">{config['descripcion']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Preparar datos con cache
    try:
        resultados = resultados_dict[escenario]
        datos = preparar_datos_cuadrantes_cache(resultados)

        # Crear visualización optimizada
        crear_cuadrantes_optimizado(datos, escenario)

        # Estadísticas resumidas
        mostrar_estadisticas_cuadrantes_simple(datos)

    except Exception as e:
        st.error(f"Error en análisis de cuadrantes: {str(e)}")


def crear_cuadrantes_optimizado(datos, escenario):
    """Crea visualización de cuadrantes optimizada"""

    fig = go.Figure()

    # Colores y símbolos por sesgo
    config_sesgos = {
        'SQ': {'color': '#2ecc71', 'symbol': 'circle', 'name': 'Status Quo'},
        'CS': {'color': '#f39c12', 'symbol': 'square', 'name': 'Contagio Social'},
        'DH': {'color': '#e74c3c', 'symbol': 'triangle-up', 'name': 'Descuento Hiperbólico'}
    }

    for sesgo in ['SQ', 'CS', 'DH']:
        datos_sesgo = datos[datos['sesgo_dominante'] == sesgo]

        if len(datos_sesgo) > 0:
            config = config_sesgos[sesgo]

            fig.add_trace(go.Scatter(
                x=datos_sesgo['PCA'],
                y=datos_sesgo['PSE'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=datos_sesgo['intensidad_dominante'],
                    colorscale='Viridis',
                    symbol=config['symbol'],
                    line=dict(color='black', width=1),
                    showscale=sesgo == 'SQ'  # Solo mostrar escala una vez
                ),
                name=config['name'],
                hovertemplate=f'<b>{config["name"]}</b><br>PCA: %{{x:.3f}}<br>PSE: %{{y:.3f}}<br>Intensidad: %{{marker.color:.3f}}<extra></extra>'
            ))

    # Líneas de cuadrantes
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

    # Añadir etiquetas de cuadrantes con conteos
    cuadrantes = ['I', 'II', 'III', 'IV']
    posiciones = [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]

    for cuadrante, (x_pos, y_pos) in zip(cuadrantes, posiciones):
        count = len(datos[datos['cuadrante'] == cuadrante])
        fig.add_annotation(
            x=x_pos, y=y_pos,
            text=f"<b>{cuadrante}</b><br>n = {count}",
            showarrow=False,
            font=dict(size=12, color='darkblue'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='darkblue',
            borderwidth=1
        )

    fig.update_layout(
        title=f'Cuadrantes PCA vs PSE - {ESCENARIOS_ECONOMICOS[escenario]["nombre"]}',
        xaxis_title='PCA (Propensión Conductual al Ahorro PCA)',
        yaxis_title='PSE (Perfil Socioeconomico Esperado)',
        height=500,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)


def mostrar_estadisticas_cuadrantes_simple(datos):
    """Muestra estadísticas simples por cuadrante"""

    st.markdown("#### Distribución por Cuadrante y Sesgo")

    # Crear tabla cruzada simple
    tabla = pd.crosstab(datos['cuadrante'],
                        datos['sesgo_dominante'], margins=True)

    # Mostrar como DataFrame simple
    st.dataframe(tabla, use_container_width=True)

    # Gráfico de barras simple
    fig_bars = go.Figure()

    cuadrantes_counts = datos['cuadrante'].value_counts().sort_index()

    fig_bars.add_trace(go.Bar(
        x=cuadrantes_counts.index,
        y=cuadrantes_counts.values,
        marker_color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'],
        text=cuadrantes_counts.values,
        textposition='auto'
    ))

    fig_bars.update_layout(
        title='Distribución por Cuadrante',
        xaxis_title='Cuadrante',
        yaxis_title='Frecuencia',
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig_bars, use_container_width=True)
