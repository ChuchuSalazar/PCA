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
        n_total = len(resultados['pca_values'])
        n_muestra = min(500, n_total)  # Máximo 500 puntos por escenario

        # Muestreo aleatorio
        indices = np.random.choice(n_total, n_muestra, replace=False)

        datos_procesados[escenario] = {
            'pca': np.array(resultados['pca_values'])[indices],
            'pse': np.array(resultados['pse_values'])[indices],
            'dh': np.array(resultados['variables_cognitivas']['DH'])[indices],
            'cs': np.array(resultados['variables_cognitivas']['CS'])[indices],
            'av': np.array(resultados['variables_cognitivas']['AV'])[indices],
            'sq': np.array(resultados['variables_cognitivas']['SQ'])[indices]
        }

    return datos_procesados


def configurar_interfaz_3d(resultados_dict):
    """Interfaz 3D optimizada sin bloqueos ni reruns"""

    if not resultados_dict:
        st.warning("No hay datos bootstrap disponibles")
        return

    st.markdown("### 📊 Análisis de Dispersión 3D - PCA Multi-dimensional")

    # Usar session_state para persistir selección
    if 'tipo_3d_persistente' not in st.session_state:
        st.session_state.tipo_3d_persistente = 'pca_pse_sesgos'

    if 'grafico_3d_generado' not in st.session_state:
        st.session_state.grafico_3d_generado = False

    # Layout en columnas para mejor control
    col_selector, col_boton = st.columns([3, 1])

    with col_selector:
        # Selector sin auto-refresh
        nuevo_tipo = st.selectbox(
            "Tipo de análisis 3D:",
            options=['pca_pse_sesgos', 'pca_pse_sq', 'sesgos_combinados'],
            format_func=lambda x: {
                'pca_pse_sesgos': 'PCA vs PSE vs Sesgos Combinados',
                'pca_pse_sq': 'PCA vs PSE vs Status Quo',
                'sesgos_combinados': 'Espacio 3D de Sesgos'
            }[x],
            index=['pca_pse_sesgos', 'pca_pse_sq', 'sesgos_combinados'].index(
                st.session_state.tipo_3d_persistente),
            key="select_3d_sin_rerun"
        )

    with col_boton:
        st.markdown("<br>", unsafe_allow_html=True)  # Espaciado
        generar_grafico = st.button(
            "🔄 Generar Gráfico 3D", type="primary", use_container_width=True)

    # Solo regenerar si se presiona el botón o si cambió el tipo
    if generar_grafico or nuevo_tipo != st.session_state.tipo_3d_persistente:
        st.session_state.tipo_3d_persistente = nuevo_tipo

        # Información del escenario
        if len(resultados_dict) == 1:
            escenario = list(resultados_dict.keys())[0]
            config = ESCENARIOS_ECONOMICOS[escenario]
            st.info(
                f"**Escenario:** {config['nombre']} - {config['descripcion']}")

        # Generar gráfico con indicador de progreso
        # generar_grafico_3d_con_progreso(resultados_dict, st.session_state.tipo_3d_persistente)
        generar_grafico_3d_con_spinner(
            resultados_dict, st.session_state.tipo_3d_persistente)

        # Mostrar métricas
        mostrar_metricas_dispersion_3d(
            resultados_dict, st.session_state.tipo_3d_persistente)

        # Interpretación
        mostrar_interpretacion_3d(st.session_state.tipo_3d_persistente)

    elif st.session_state.grafico_3d_generado:
        st.info(
            "💡 Gráfico ya generado. Presiona 'Generar Gráfico 3D' para actualizar con nueva selección.")


def mostrar_metricas_dispersion_3d(resultados_dict, tipo_analisis):
    """Muestra métricas de dispersión 3D para mejor entendimiento"""


st.markdown("#### 📊 Métricas de Dispersión 3D")

try:
    datos = procesar_datos_3d(resultados_dict, tipo_analisis)

    metricas_por_escenario = []

    for escenario, data in datos.items():
        config = ESCENARIOS_ECONOMICOS[escenario]

        if tipo_analisis == 'pca_pse_sesgos':
            indice_sesgos = 0.5 * \
                np.abs(data['sq']) + 0.3 * \
                np.abs(data['dh']) + 0.2 * np.abs(data['cs'])

            # Calcular métricas de dispersión
            dispersion_pca = float(np.std(data['pca']))
            dispersion_pse = float(np.std(data['pse']))
            dispersion_sesgos = float(np.std(indice_sesgos))

            # Correlación PCA-PSE
            correlacion_pca_pse = float(
                np.corrcoef(data['pca'], data['pse'])[0, 1])

            # Correlación PCA-Sesgos
            correlacion_pca_sesgos = float(
                np.corrcoef(data['pca'], indice_sesgos)[0, 1])

            # Rango de valores
            rango_pca = float(
                np.max(data['pca']) - np.min(data['pca']))
            rango_sesgos = float(
                np.max(indice_sesgos) - np.min(indice_sesgos))

            metricas_por_escenario.append({
                'Escenario': config['nombre'],
                'N_Puntos': len(data['pca']),
                'PCA_Std': dispersion_pca,
                'PSE_Std': dispersion_pse,
                'Sesgos_Std': dispersion_sesgos,
                'Corr_PCA_PSE': correlacion_pca_pse,
                'Corr_PCA_Sesgos': correlacion_pca_sesgos,
                'Rango_PCA': rango_pca,
                'Rango_Sesgos': rango_sesgos
            })

    # Mostrar tabla de métricas
    if metricas_por_escenario:
        df_metricas = pd.DataFrame(metricas_por_escenario)

        # Formatear números
        numeric_cols = ['PCA_Std', 'PSE_Std', 'Sesgos_Std',
                        'Corr_PCA_PSE', 'Corr_PCA_Sesgos', 'Rango_PCA', 'Rango_Sesgos']
        for col in numeric_cols:
            df_metricas[col] = df_metricas[col].round(4)

        st.dataframe(
            df_metricas, use_container_width=True, hide_index=True)

        # Interpretación de métricas
        st.markdown("""
            **💡 Interpretación de Métricas:**
            - **Std**: Dispersión de la variable (mayor valor = mayor heterogeneidad)
            - **Corr_PCA_PSE**: Correlación entre propensión conductual y expectativas
            - **Corr_PCA_Sesgos**: Correlación entre PCA e intensidad de sesgos
            - **Rango**: Amplitud de valores (diferencia entre máximo y mínimo)
            """)

except Exception as e:
    st.error(f"Error calculando métricas: {str(e)}")
    st.session_state.grafico_3d_generado = True


def crear_grafico_3d_ligero(resultados_dict, tipo_analisis):
    """Crea gráfico 3D ligero y rápido con texto más grande"""

    try:
        # Procesar datos con cache
        datos = procesar_datos_3d(resultados_dict, tipo_analisis)

        fig = go.Figure()

        colores_escenarios = {
            'baseline': '#34495e',
            'crisis': '#e74c3c',
            'bonanza': '#27ae60'
        }

        for escenario, data in datos.items():
            color = colores_escenarios.get(escenario, '#666666')
            config = ESCENARIOS_ECONOMICOS[escenario]

            if tipo_analisis == 'pca_pse_sesgos':
                # Índice de sesgos simplificado
                indice_sesgos = 0.5 * \
                    np.abs(data['sq']) + 0.3 * \
                    np.abs(data['dh']) + 0.2 * np.abs(data['cs'])

                fig.add_trace(go.Scatter3d(
                    x=data['pse'],
                    y=indice_sesgos,
                    z=data['pca'],
                    mode='markers',
                    marker=dict(size=4, color=color, opacity=0.7),
                    name=config['nombre'],
                    hovertemplate=f'<b>{escenario}</b><br>PSE: %{{x:.3f}}<br>Sesgos: %{{y:.3f}}<br>PCA: %{{z:.3f}}<extra></extra>'
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title='PSE',
                        yaxis_title='Índice Sesgos',
                        zaxis_title='PCA'
                    )
                )

            elif tipo_analisis == 'pca_pse_sq':
                fig.add_trace(go.Scatter3d(
                    x=data['pse'],
                    y=data['sq'],
                    z=data['pca'],
                    mode='markers',
                    marker=dict(size=4, color=color, opacity=0.7),
                    name=config['nombre'],
                    hovertemplate=f'<b>{escenario}</b><br>PSE: %{{x:.3f}}<br>Status Quo: %{{y:.3f}}<br>PCA: %{{z:.3f}}<extra></extra>'
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title='PSE',
                        yaxis_title='Status Quo',
                        zaxis_title='PCA'
                    )
                )

            elif tipo_analisis == 'sesgos_combinados':
                fig.add_trace(go.Scatter3d(
                    x=data['dh'],
                    y=data['cs'],
                    z=data['sq'],
                    mode='markers',
                    marker=dict(
                        size=4, color=data['pca'], colorscale='Viridis', opacity=0.8),
                    name=config['nombre'],
                    hovertemplate=f'<b>{escenario}</b><br>DH: %{{x:.3f}}<br>CS: %{{y:.3f}}<br>SQ: %{{z:.3f}}<extra></extra>'
                ))

                fig.update_layout(
                    scene=dict(
                        xaxis_title='Descuento Hiperbólico',
                        yaxis_title='Contagio Social',
                        zaxis_title='Status Quo'
                    )
                )

        fig.update_layout(
            title=dict(
                text=f'<b>Análisis 3D: {tipo_analisis.replace("_", " ").title()}</b>',
                font=dict(size=20)  # Título más grande
            ),
            height=700,  # Aumentar altura
            scene=dict(
                bgcolor='rgba(240,240,240,0.1)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                xaxis=dict(
                    # Ejes más grandes
                    title=dict(text='<b>PSE</b>', font=dict(size=16)),
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title=dict(text='<b>Índice Sesgos</b>' if tipo_analisis == 'pca_pse_sesgos' else '<b>Status Quo</b>',
                               font=dict(size=16)),
                    tickfont=dict(size=12)
                ),
                zaxis=dict(
                    title=dict(text='<b>PCA</b>', font=dict(size=16)),
                    tickfont=dict(size=12)
                )
            ),
            font=dict(size=14),  # Fuente general más grande
            legend=dict(font=dict(size=14))  # Leyenda más grande
        )

        # Nota metodológica con fuente más grande
        fig.add_annotation(
            text="<b>Metodología:</b> Bootstrap resampling con muestra optimizada para visualización",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=-0.08,
            xanchor='center',
            font=dict(size=12, color="gray")  # Aumentado de 10 a 12
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error en visualización 3D: {str(e)}")


def generar_grafico_3d_con_progreso(resultados_dict, tipo_analisis):
    """Genera gráfico 3D con indicadores de progreso"""

    # Crear contenedores para progreso
    progress_container = st.empty()
    status_container = st.empty()
    chart_container = st.empty()

    try:
        with progress_container.container():
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔄 Procesando datos bootstrap...")
            progress_bar.progress(20)

            # Procesar datos
            datos = procesar_datos_3d(resultados_dict, tipo_analisis)

            status_text.text("📊 Calculando métricas de dispersión...")
            progress_bar.progress(40)

            # Crear figura base
            fig = go.Figure()

            status_text.text("🎨 Generando visualización 3D...")
            progress_bar.progress(60)

            # Añadir trazos según tipo de análisis
            colores_escenarios = {
                'baseline': '#34495e',
                'crisis': '#e74c3c',
                'bonanza': '#27ae60'
            }

            for escenario, data in datos.items():
                status_text.text(
                    f"📈 Procesando {ESCENARIOS_ECONOMICOS[escenario]['nombre']}...")

                color = colores_escenarios.get(escenario, '#666666')
                config = ESCENARIOS_ECONOMICOS[escenario]

                if tipo_analisis == 'pca_pse_sesgos':
                    indice_sesgos = 0.5 * \
                        np.abs(data['sq']) + 0.3 * \
                        np.abs(data['dh']) + 0.2 * np.abs(data['cs'])

                    fig.add_trace(go.Scatter3d(
                        x=data['pse'],
                        y=indice_sesgos,
                        z=data['pca'],
                        mode='markers',
                        marker=dict(size=5, color=color, opacity=0.7),
                        name=config['nombre'],
                        hovertemplate=f'<b>{escenario}</b><br>PSE: %{{x:.3f}}<br>Sesgos: %{{y:.3f}}<br>PCA: %{{z:.3f}}<extra></extra>'
                    ))

                    axes_titles = {
                        'xaxis_title': '<b>PSE (Propensión Esperada)</b>',
                        'yaxis_title': '<b>Índice Sesgos Combinados</b>',
                        'zaxis_title': '<b>PCA (Propensión Conductual)</b>'
                    }

                elif tipo_analisis == 'pca_pse_sq':
                    fig.add_trace(go.Scatter3d(
                        x=data['pse'],
                        y=data['sq'],
                        z=data['pca'],
                        mode='markers',
                        marker=dict(size=5, color=color, opacity=0.7),
                        name=config['nombre']
                    ))

                    axes_titles = {
                        'xaxis_title': '<b>PSE (Propensión Esperada)</b>',
                        'yaxis_title': '<b>Status Quo Bias</b>',
                        'zaxis_title': '<b>PCA (Propensión Conductual)</b>'
                    }

                elif tipo_analisis == 'sesgos_combinados':
                    fig.add_trace(go.Scatter3d(
                        x=data['dh'],
                        y=data['cs'],
                        z=data['sq'],
                        mode='markers',
                        marker=dict(
                            size=5, color=data['pca'], colorscale='Viridis', opacity=0.8),
                        name=config['nombre']
                    ))

                    axes_titles = {
                        'xaxis_title': '<b>Descuento Hiperbólico</b>',
                        'yaxis_title': '<b>Contagio Social</b>',
                        'zaxis_title': '<b>Status Quo</b>'
                    }

            progress_bar.progress(80)
            status_text.text("🎨 Aplicando estilos y configuraciones...")

            # Configurar layout mejorado
            fig.update_layout(
                title=dict(
                    text=f'<b>Análisis de Dispersión 3D: {tipo_analisis.replace("_", " ").title()}</b>',
                    font=dict(size=22, family="Arial Black")
                ),
                height=750,
                scene=dict(
                    bgcolor='rgba(240,245,250,0.1)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                    xaxis=dict(
                        title=dict(text=axes_titles['xaxis_title'], font=dict(
                            size=16, family="Arial")),
                        tickfont=dict(size=12),
                        backgroundcolor="rgba(230,230,250,0.3)",
                        gridcolor="white",
                        showbackground=True
                    ),
                    yaxis=dict(
                        title=dict(text=axes_titles['yaxis_title'], font=dict(
                            size=16, family="Arial")),
                        tickfont=dict(size=12),
                        backgroundcolor="rgba(230,250,230,0.3)",
                        gridcolor="white",
                        showbackground=True
                    ),
                    zaxis=dict(
                        title=dict(text=axes_titles['zaxis_title'], font=dict(
                            size=16, family="Arial")),
                        tickfont=dict(size=12),
                        backgroundcolor="rgba(250,230,230,0.3)",
                        gridcolor="white",
                        showbackground=True
                    )
                ),
                font=dict(size=14, family="Arial"),
                legend=dict(font=dict(size=14), orientation="v", x=1.02, y=1)
            )

            progress_bar.progress(100)
            status_text.text("✅ Visualización 3D completada!")

        # Limpiar indicadores de progreso y mostrar gráfico
        progress_container.empty()

        with chart_container.container():
            st.plotly_chart(fig, use_container_width=True,
                            key=f"3d_plot_{tipo_analisis}")

    except Exception as e:
        progress_container.empty()
        status_container.empty()
        st.error(f"Error generando visualización 3D: {str(e)}")


def generar_grafico_3d_con_spinner(resultados_dict, tipo_analisis):
    """Genera gráfico 3D con spinner sin mover la interfaz"""

    try:
        with st.spinner("🔄 Procesando datos y generando visualización 3D..."):
            # Procesar datos
            datos = procesar_datos_3d(resultados_dict, tipo_analisis)

            # Crear figura base
            fig = go.Figure()

            colores_escenarios = {
                'baseline': '#34495e',
                'crisis': '#e74c3c',
                'bonanza': '#27ae60'
            }

            for escenario, data in datos.items():
                color = colores_escenarios.get(escenario, '#666666')
                config = ESCENARIOS_ECONOMICOS[escenario]

                if tipo_analisis == 'pca_pse_sesgos':
                    indice_sesgos = 0.5 * np.abs(data['sq']) + \
                        0.3 * np.abs(data['dh']) + \
                        0.2 * np.abs(data['cs'])

                    fig.add_trace(go.Scatter3d(
                        x=data['pse'],
                        y=indice_sesgos,
                        z=data['pca'],
                        mode='markers',
                        marker=dict(size=5, color=color, opacity=0.7),
                        name=config['nombre'],
                        hovertemplate=f'<b>{escenario}</b><br>PSE: %{{x:.3f}}'
                        f'<br>Sesgos: %{{y:.3f}}'
                        f'<br>PCA: %{{z:.3f}}<extra></extra>'
                    ))

                    axes_titles = {
                        'xaxis_title': '<b>PSE (Propensión Esperada)</b>',
                        'yaxis_title': '<b>Índice Sesgos Combinados</b>',
                        'zaxis_title': '<b>PCA (Propensión Conductual)</b>'
                    }

                elif tipo_analisis == 'pca_pse_sq':
                    fig.add_trace(go.Scatter3d(
                        x=data['pse'],
                        y=data['sq'],
                        z=data['pca'],
                        mode='markers',
                        marker=dict(size=5, color=color, opacity=0.7),
                        name=config['nombre']
                    ))

                    axes_titles = {
                        'xaxis_title': '<b>PSE (Propensión Esperada)</b>',
                        'yaxis_title': '<b>Status Quo Bias</b>',
                        'zaxis_title': '<b>PCA (Propensión Conductual)</b>'
                    }

                elif tipo_analisis == 'sesgos_combinados':
                    fig.add_trace(go.Scatter3d(
                        x=data['dh'],
                        y=data['cs'],
                        z=data['sq'],
                        mode='markers',
                        marker=dict(
                            size=5, color=data['pca'], colorscale='Viridis', opacity=0.8
                        ),
                        name=config['nombre']
                    ))

                    axes_titles = {
                        'xaxis_title': '<b>Descuento Hiperbólico</b>',
                        'yaxis_title': '<b>Contagio Social</b>',
                        'zaxis_title': '<b>Status Quo</b>'
                    }

            # Configurar layout
            fig.update_layout(
                title=dict(
                    text=f'<b>Análisis de Dispersión 3D: {tipo_analisis.replace("_", " ").title()}</b>',
                    font=dict(size=22, family="Arial Black")
                ),
                height=750,
                scene=dict(
                    bgcolor='rgba(240,245,250,0.1)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                    xaxis=dict(title=dict(
                        text=axes_titles['xaxis_title'], font=dict(size=16))),
                    yaxis=dict(title=dict(
                        text=axes_titles['yaxis_title'], font=dict(size=16))),
                    zaxis=dict(title=dict(
                        text=axes_titles['zaxis_title'], font=dict(size=16)))
                ),
                font=dict(size=14),
                legend=dict(font=dict(size=14), orientation="v", x=1.02, y=1)
            )

        # Mostrar gráfico al final, sin mover layout
        st.plotly_chart(fig, use_container_width=True,
                        key=f"3d_plot_{tipo_analisis}")

    except Exception as e:
        st.error(f"Error generando visualización 3D: {str(e)}")


def mostrar_interpretacion_3d(tipo_analisis):
    """Muestra interpretación del análisis 3D"""

    interpretaciones = {
        'pca_pse_sesgos': """
        **Interpretación:** El gráfico muestra cómo la PCA real se relaciona con las expectativas (PSE) 
        y la intensidad de los sesgos cognitivos. Mayor dispersión indica heterogeneidad en comportamientos.
        """,
        'pca_pse_sq': """
        **Interpretación:** Enfoque en Status Quo como sesgo dominante. Los puntos rojos (crisis) 
        muestran mayor resistencia al cambio que impacta las decisiones de ahorro.
        """,
        'sesgos_combinados': """
        **Interpretación:** Espacio tridimensional de sesgos donde el color representa la intensidad 
        de PCA resultante. Permite identificar combinaciones de sesgos más influyentes.
        """
    }

    if tipo_analisis in interpretaciones:
        st.info(interpretaciones[tipo_analisis])


def mostrar_informacion_sesgos_escenario(escenario):
    """Información de sesgos por escenario - Versión simplificada"""

    if escenario not in ESCENARIOS_ECONOMICOS:
        return

    config = ESCENARIOS_ECONOMICOS[escenario]

    info_sesgos = {
        'baseline': "Los sesgos operan en niveles naturales sin amplificación externa.",
        'crisis': "Sesgos intensificados: DH (1.6x), CS (1.5x), AV (1.7x), SQ (1.3x).",
        'bonanza': "Sesgos moderados pero persistentes con diferentes motivaciones."
    }

    st.markdown(f"""
    <div style="background: {config['color']}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {config['color']};">
        <h4 style="color: {config['color']}; margin: 0;">{config['nombre']}</h4>
        <p style="margin: 0.5rem 0 0 0;">{info_sesgos.get(escenario, config['descripcion'])}</p>
    </div>
    """, unsafe_allow_html=True)
