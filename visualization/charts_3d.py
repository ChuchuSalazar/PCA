"""
Visualizaciones 3D optimizadas para análisis de dispersión PCA
Versión ligera sin bloqueos de interfaz
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from config.constants import ESCENARIOS_ECONOMICOS


@st.cache_data(ttl=300)  # Cache por 5 minutos
def procesar_datos_3d(resultados_dict, tipo_analisis):
    """Procesa datos para 3D con cache para evitar recálculos"""
    datos_procesados = {}

    for escenario, resultados in resultados_dict.items():
        # Usar muestreo para reducir carga computacional
        n_total = len(resultados["pca_values"])
        n_muestra = min(500, n_total)  # Máximo 500 puntos por escenario

        # Muestreo aleatorio
        indices = np.random.choice(n_total, n_muestra, replace=False)

        datos_procesados[escenario] = {
            "pca": np.array(resultados["pca_values"])[indices],
            "pse": np.array(resultados["pse_values"])[indices],
            "dh": np.array(resultados["variables_cognitivas"]["DH"])[indices],
            "cs": np.array(resultados["variables_cognitivas"]["CS"])[indices],
            "av": np.array(resultados["variables_cognitivas"]["AV"])[indices],
            "sq": np.array(resultados["variables_cognitivas"]["SQ"])[indices],
        }

    return datos_procesados


def configurar_interfaz_3d(resultados_dict):
    """Interfaz 3D con tabs horizontales - OPTIMIZADA"""

    if not resultados_dict:
        st.warning("No hay datos bootstrap disponibles")
        return

    st.markdown("### 📊 Análisis de Dispersión 3D - PCA Multi-dimensional")

    # TABS HORIZONTALES EN LUGAR DE SELECTBOX
    tab1, tab2, tab3 = st.tabs(
        [
            "🎯 PCA vs PSE vs Sesgos",
            "⚖️ PCA vs PSE vs Status Quo",
            "🧠 Espacio 3D de Sesgos",
        ]
    )

    with tab1:
        st.markdown("**PCA vs PSE vs Índice de Sesgos Combinados**")
        generar_grafico_3d_optimizado(resultados_dict, "pca_pse_sesgos")
        mostrar_metricas_pie_grafico(resultados_dict, "pca_pse_sesgos")
        mostrar_interpretacion_expandida("pca_pse_sesgos")

    with tab2:
        st.markdown("**PCA vs PSE vs Status Quo (Sesgo Dominante)**")
        generar_grafico_3d_optimizado(resultados_dict, "pca_pse_sq")
        mostrar_metricas_pie_grafico(resultados_dict, "pca_pse_sq")
        mostrar_interpretacion_expandida("pca_pse_sq")

    with tab3:
        st.markdown("**Espacio Tridimensional de Sesgos Cognitivos**")
        generar_grafico_3d_optimizado(resultados_dict, "sesgos_combinados")
        mostrar_metricas_pie_grafico(resultados_dict, "sesgos_combinados")
        mostrar_interpretacion_expandida("sesgos_combinados")


@st.cache_data(ttl=900)  # Cache más largo para optimizar
def procesar_datos_3d_super_optimizado(resultados_dict, tipo_analisis):
    """Procesamiento súper optimizado con menos puntos"""
    datos_procesados = {}

    for escenario, resultados in resultados_dict.items():
        n_total = len(resultados["pca_values"])
        n_muestra = min(200, n_total)  # REDUCIDO A 200 para más velocidad
        indices = np.random.choice(n_total, n_muestra, replace=False)

        datos_procesados[escenario] = {
            "pca": np.array(resultados["pca_values"])[indices],
            "pse": np.array(resultados["pse_values"])[indices],
            "dh": np.array(resultados["variables_cognitivas"]["DH"])[indices],
            "cs": np.array(resultados["variables_cognitivas"]["CS"])[indices],
            "sq": np.array(resultados["variables_cognitivas"]["SQ"])[indices],
        }

    return datos_procesados


def generar_grafico_3d_optimizado(resultados_dict, tipo_analisis):
    """Versión super optimizada con texto más grande"""

    try:
        datos = procesar_datos_3d_super_optimizado(resultados_dict, tipo_analisis)

        fig = go.Figure()
        colores = {"baseline": "#34495e", "crisis": "#e74c3c", "bonanza": "#27ae60"}

        for escenario, data in datos.items():
            config = ESCENARIOS_ECONOMICOS[escenario]

            if tipo_analisis == "pca_pse_sesgos":
                idx_sesgos = (
                    0.5 * np.abs(data["sq"])
                    + 0.3 * np.abs(data["dh"])
                    + 0.2 * np.abs(data["cs"])
                )
                x, y, z = data["pse"], idx_sesgos, data["pca"]
                labels = (
                    "PSE (Propensión Esperada)",
                    "Índice Sesgos Combinados",
                    "PCA (Propensión Conductual)",
                )

            elif tipo_analisis == "pca_pse_sq":
                x, y, z = data["pse"], data["sq"], data["pca"]
                labels = (
                    "PSE (Propensión Esperada)",
                    "Status Quo Bias",
                    "PCA (Propensión Conductual)",
                )

            else:  # sesgos_combinados
                x, y, z = data["dh"], data["cs"], data["sq"]
                labels = ("Descuento Hiperbólico", "Contagio Social", "Status Quo")

            # SÍMBOLOS MÁS GRANDES Y VISIBLES
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=8,  # AUMENTADO
                        color=(
                            colores[escenario]
                            if tipo_analisis != "sesgos_combinados"
                            else data["pca"]
                        ),
                        colorscale=(
                            "Viridis" if tipo_analisis == "sesgos_combinados" else None
                        ),
                        opacity=0.8,
                        line=dict(color="black", width=1),  # LÍNEA MÁS GRUESA
                    ),
                    name=config["nombre"],
                    hovertemplate=f'<b>{config["nombre"]}</b><br>{labels[0]}: %{{x:.3f}}<br>{labels[1]}: %{{y:.3f}}<br>{labels[2]}: %{{z:.3f}}<extra></extra>',
                )
            )

        fig.update_layout(
            title=dict(
                text=f'<b>{tipo_analisis.replace("_", " ").title()}</b>',
                font=dict(
                    size=22, family="Arial Black", color="#2c3e50"
                ),  # TÍTULO MÁS GRANDE
            ),
            height=700,
            scene=dict(
                bgcolor="rgba(248,250,252,0.1)",
                camera=dict(eye=dict(x=1.3, y=1.3, z=1.0)),
                xaxis=dict(
                    title=dict(
                        text=f"<b>{labels[0]}</b>", font=dict(size=18, color="#1a1a1a")
                    ),  # MÁS GRANDE Y OSCURO
                    tickfont=dict(size=14, color="#333333"),  # TICKS MÁS GRANDES
                ),
                yaxis=dict(
                    title=dict(
                        text=f"<b>{labels[1]}</b>", font=dict(size=18, color="#1a1a1a")
                    ),
                    tickfont=dict(size=14, color="#333333"),
                ),
                zaxis=dict(
                    title=dict(
                        text=f"<b>{labels[2]}</b>", font=dict(size=18, color="#1a1a1a")
                    ),
                    tickfont=dict(size=14, color="#333333"),
                ),
            ),
            font=dict(size=14),
            legend=dict(
                font=dict(size=16, color="#2c3e50"),  # LEYENDA MÁS GRANDE
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=2,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")


def mostrar_informacion_sesgos_escenario(escenario):
    """Información de sesgos por escenario - Versión simplificada"""

    if escenario not in ESCENARIOS_ECONOMICOS:
        return

    config = ESCENARIOS_ECONOMICOS[escenario]

    info_sesgos = {
        "baseline": "Los sesgos operan en niveles naturales sin amplificación externa.",
        "crisis": "Sesgos intensificados: DH (1.6x), CS (1.5x), AV (1.7x), SQ (1.3x).",
        "bonanza": "Sesgos moderados pero persistentes con diferentes motivaciones.",
    }

    st.markdown(
        f"""
    <div style="background: {config['color']}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {config['color']};">
        <h4 style="color: {config['color']}; margin: 0;">{config['nombre']}</h4>
        <p style="margin: 0.5rem 0 0 0;">{info_sesgos.get(escenario, config['descripcion'])}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
