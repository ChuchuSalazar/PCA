import streamlit as st

st.set_page_config(
    page_title="PCA Simulator v3.2 - Bootstrap Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# Al inicio del archivo main.py, con las otras importaciones
from tutorial_gamificado import mostrar_tutorial_principal
from plotly.subplots import make_subplots
from visualization.cuadrantes import crear_analisis_cuadrantes_completo
from config.constants import (
    MODELOS_COEFICIENTES,
    ESCENARIOS_ECONOMICOS,
    MODELOS_EXTERNOS,
)
from config.styles import apply_custom_styles, get_header_html
from data.data_loader import cargar_datos
from data.participantes_analysis import (
    cargar_datos_participantes,
    crear_analisis_participantes,
)
from models.bootstrap_analysis import ejecutar_bootstrap_avanzado
from visualization.dashboard import crear_dashboard_bootstrap_comparativo
from visualization.charts import crear_grafico_bootstrap_diagnostics
from visualization.cuadrantes import crear_analisis_cuadrantes_completo
from visualization.charts_3d import (
    configurar_interfaz_3d,
    mostrar_informacion_sesgos_escenario,
)
from utils.export_utils import crear_excel_bootstrap_completo
from utils.statistics import calcular_estadisticas_avanzadas

from datetime import datetime


# PCA Simulator v3.2 - Bootstrap Analysis - Enhanced Version COMPLETO
# Enhanced Behavioral Economics Analysis Tool

# Author: MSc. Jesús Fernando Salazar Rojas
# Doctorado en Economía, UCAB – 2025
# Methodology: PLS-SEM + Bootstrap Resampling
# Framework: DH • CS • AV • SQ Analysis


# Configuración de la página


# =========================
# Configurar carpeta raíz
# =========================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Importaciones de módulos locales


def main():
    """Función principal de la aplicación"""

    # Aplicar estilos CSS personalizados
    apply_custom_styles()

    # Mostrar header principal
    st.markdown(get_header_html(), unsafe_allow_html=True)

    # Inicializar session state
    initialize_session_state()

    # Cargar datos
    with st.spinner("Loading bootstrap database..."):
        try:
            scores_df, items_df = cargar_datos()
        except Exception as e:
            st.error("❌ Error cargando los datos desde cargar_datos()")
            st.exception(e)
            st.stop()

    # Sidebar de configuración - llamamos solo una vez
    grupo, escenario, n_bootstrap, analisis_comparativo = setup_sidebar()

    # Interfaz principal
    main_interface(
        scores_df, items_df, grupo, escenario, n_bootstrap, analisis_comparativo
    )

    # Footer
    display_footer()


def initialize_session_state():
    """Inicializa las variables de session state - OPTIMIZADO"""
    if "resultados_dict" not in st.session_state:
        st.session_state.resultados_dict = None
    if "simulation_completed" not in st.session_state:
        st.session_state.simulation_completed = False
    if "current_parameters" not in st.session_state:
        st.session_state.current_parameters = None
    if "show_model_images" not in st.session_state:
        st.session_state.show_model_images = False
    # NUEVOS PARA OPTIMIZACIÓN
    if "last_bootstrap_hash" not in st.session_state:
        st.session_state.last_bootstrap_hash = None
    if "bootstrap_cache" not in st.session_state:
        st.session_state.bootstrap_cache = {}
    if "modo_tutorial" not in st.session_state:
        st.session_state.modo_tutorial = False


def generar_hash_parametros(grupo, escenario, n_bootstrap, analisis_comparativo):
    """Genera hash único para parámetros de bootstrap"""
    import hashlib

    param_string = f"{grupo}_{escenario}_{n_bootstrap}_{analisis_comparativo}"
    return hashlib.md5(param_string.encode()).hexdigest()


def execute_bootstrap_analysis(grupo, escenario, n_bootstrap, analisis_comparativo):
    """Ejecuta el análisis Bootstrap - OPTIMIZADO SIN REGENERACIÓN INNECESARIA"""
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        current_params = {
            "grupo": grupo,
            "escenario": escenario,
            "n_bootstrap": n_bootstrap,
            "analisis_comparativo": analisis_comparativo,
        }

        # Generar hash de parámetros actuales
        current_hash = generar_hash_parametros(
            grupo, escenario, n_bootstrap, analisis_comparativo
        )

        escenario_info = ESCENARIOS_ECONOMICOS[escenario]
        button_text = f"**EXECUTE BOOTSTRAP ANALYSIS**" + (
            f" - {escenario_info['nombre']}"
            if not analisis_comparativo
            else " - MULTI-SCENARIO"
        )

        if st.button(
            button_text,
            type="primary",
            use_container_width=True,
            key="execute_bootstrap_optimized",
        ):

            # VERIFICAR SI YA EXISTE RESULTADO CON MISMOS PARÁMETROS
            if (
                st.session_state.last_bootstrap_hash == current_hash
                and st.session_state.simulation_completed
                and st.session_state.resultados_dict
            ):

                st.success(
                    "✅ **Usando resultados Bootstrap existentes** (parámetros idénticos)"
                )
                st.info(
                    "💡 Los parámetros no han cambiado. Reutilizando análisis previo para optimizar tiempo."
                )
                return True

            # EJECUTAR NUEVO ANÁLISIS SOLO SI PARÁMETROS CAMBIARON
            if analisis_comparativo:
                st.markdown("### Executing Multi-scenario Bootstrap Analysis...")
                progress_bar = st.progress(0)
                resultados_dict = {}

                for i, esc in enumerate(["baseline", "crisis", "bonanza"]):
                    with st.spinner(
                        f"Bootstrap resampling: {ESCENARIOS_ECONOMICOS[esc]['nombre']}..."
                    ):
                        resultados_dict[esc] = ejecutar_bootstrap_avanzado(
                            grupo, esc, n_bootstrap
                        )
                    progress_bar.progress((i + 1) / 3)

                st.session_state.resultados_dict = resultados_dict
                st.session_state.simulation_completed = True
                st.session_state.current_parameters = current_params
                st.session_state.last_bootstrap_hash = current_hash  # GUARDAR HASH

                st.success(
                    f"**Multi-scenario Bootstrap Completed:** {n_bootstrap:,} × 3 iterations"
                )

            else:
                with st.spinner(f"Bootstrap analysis: {escenario_info['nombre']}..."):
                    resultado = ejecutar_bootstrap_avanzado(
                        grupo, escenario, n_bootstrap
                    )

                st.session_state.resultados_dict = {escenario: resultado}
                st.session_state.simulation_completed = True
                st.session_state.current_parameters = current_params
                st.session_state.last_bootstrap_hash = current_hash  # GUARDAR HASH

                st.success(
                    f"**Single Bootstrap Analysis Completed:** {n_bootstrap:,} iterations"
                )

            return True

    return False


def setup_sidebar():
    """Configura el sidebar con controles y toggle de imágenes"""
    with st.sidebar:
        # Selección de grupo
        grupo = st.selectbox(
            "Select analysis group:",
            options=list(MODELOS_COEFICIENTES.keys()),
            format_func=lambda x: "Male Savers" if x == "Hah" else "Female Savers",
        )

        # Métricas asociadas al grupo
        display_model_metrics(grupo)
        # AÑADIR ESTAS LÍNEAS:

        st.markdown("---")
        st.markdown("### 🎮 Aprendizaje Interactivo")
        
        # Botón del tutorial
        if st.button("🎯 Iniciar Tutorial Gamificado", type="secondary", use_container_width=True):
            # Activar modo tutorial
            st.session_state.modo_tutorial = True
            st.rerun()
        
        # Mostrar estado del tutorial si está disponible
        if hasattr(st.session_state, 'puntos_totales') and st.session_state.puntos_totales > 0:
            st.success(f"Tutorial: {st.session_state.puntos_totales} puntos")



        # Toggle para mostrar modelos o logo
        if st.button("🔄 Toggle PLS-SEM Models / UCAB Logo", type="secondary"):
            st.session_state.show_model_images = not st.session_state.show_model_images

        estado_actual = (
            "Modelos PLS-SEM" if st.session_state.show_model_images else "Logo UCAB"
        )
        st.info(f"**Mostrando:** {estado_actual}")

        st.markdown("---")

        # Selección de escenario
        escenario = setup_scenario_selection()

        # Parámetros de bootstrap
        n_bootstrap, analisis_comparativo = setup_bootstrap_parameters()

        return grupo, escenario, n_bootstrap, analisis_comparativo


def display_model_metrics(grupo):
    """Muestra las métricas del modelo seleccionado"""
    model_stats = MODELOS_COEFICIENTES[grupo]
    st.markdown(
        f"""
    <div class="metric-card">
        <h4>PLS-SEM Model Metrics - {grupo}</h4>
        <p><strong>R²:</strong> {model_stats['r2']:.4f}</p>
        <p><strong>RMSE:</strong> {model_stats['rmse']:.4f}</p>
        <p><strong>Correlation:</strong> {model_stats['correlation']:.4f}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def setup_scenario_selection():
    """Configura la selección de escenario económico"""
    st.markdown("**Economic Scenario**")
    escenario = st.radio(
        "Select economic context:",
        options=["baseline", "crisis", "bonanza"],
        format_func=lambda x: ESCENARIOS_ECONOMICOS[x]["nombre"],
        index=0,
        key="radio_escenario",
    )

    # Mostrar información del escenario
    escenario_info = ESCENARIOS_ECONOMICOS[escenario]
    st.markdown(
        f"""
    <div class="scenario-card" style="background: linear-gradient(45deg, {escenario_info['color']}22, {escenario_info['color']}44);">
        <h4 style="color: {escenario_info['color']};">{escenario_info['nombre']}</h4>
        <p style="margin: 0; font-size: 0.9rem; color: #333;">{escenario_info['descripcion']}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    return escenario


def setup_bootstrap_parameters():
    """Configura los parámetros del análisis Bootstrap"""
    st.markdown("---")
    st.markdown("**Bootstrap Parameters**")

    n_bootstrap = st.number_input(
        "Number of Bootstrap iterations",
        min_value=1000,
        max_value=5000,
        value=3000,
        step=500,
        help="Bootstrap resampling iterations for statistical inference",
        key="number_input_bootstrap",
    )

    analisis_comparativo = st.checkbox(
        "**Multi-scenario Bootstrap Analysis**", value=True, key="checkbox_comparativo"
    )

    # Información metodológica
    with st.expander("ℹ️ Bootstrap Methodology Info"):
        st.markdown(
            """
        **Bootstrap Resampling Benefits:**
        - More robust statistical inference
        - Bias correction capabilities
        - Confidence interval estimation
        - Distribution-free approach
        - Better handling of small samples
        
        **vs Monte Carlo:**
        - Uses actual data distribution
        - Accounts for sampling variability
        - Provides bias-corrected estimates
        """
        )

    return n_bootstrap, analisis_comparativo


def main_interface(
    scores_df, items_df, grupo, escenario, n_bootstrap, analisis_comparativo
):
    """Interfaz principal con logo UCAB"""
    # AÑADIR ESTA VERIFICACIÓN AL INICIO:
    # Verificar si está activo el modo tutorial
    if getattr(st.session_state, 'modo_tutorial', False):
        mostrar_tutorial_principal()
        return  # No mostrar el resto de la interfaz


    # Mostrar logo UCAB o imágenes del modelo
    if st.session_state.show_model_images:
        st.markdown("---")
        display_model_images(grupo)
        st.markdown("---")
    else:
        # MOSTRAR LOGO UCAB CON DEGRADADO
        display_ucab_logo()
        st.markdown("---")

    # Mostrar información del modelo actual
    display_current_model_info(grupo, escenario, n_bootstrap)

    # Botón de ejecución del análisis
    if execute_bootstrap_analysis(grupo, escenario, n_bootstrap, analisis_comparativo):
        # Mostrar resultados
        display_results()


def display_ucab_logo():
    """Muestra logo UCAB con efecto degradado"""
    try:
        if os.path.exists("Logo_UCAB_1.png"):
            from PIL import Image

            logo = Image.open("Logo_UCAB_1.png")

            # Container con efecto degradado

            # Logo centrado
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(logo, use_container_width=True)

        else:
            # Fallback si no hay logo
            st.markdown(
                """
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                        padding: 3rem; border-radius: 15px; text-align: center; color: white;">
                <h2>Universidad Católica Andrés Bello</h2>
                <h3>Doctorado en Economía</h3>
                <p>Simulador PCA v3.2 - Análisis Conductual del Ahorro</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Error mostrando logo: {str(e)}")


def display_model_images(grupo):
    """Muestra las imágenes del modelo estructural según el grupo"""
    try:
        if grupo == "Hah":
            image_path = "hombres.JPG"
            title = "Structural Model - Male Savers (Hah)"
        else:
            image_path = "mujeres.JPG"
            title = "Structural Model - Female Savers (Mah)"

        if os.path.exists(image_path):
            from PIL import Image

            image = Image.open(image_path)
            st.markdown(
                f"""
            <div class="model-images">
                <h4 style="color: #2c3e50; text-align: center; margin-bottom: 1rem;">{title}</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.image(image, caption=title, use_container_width=True)
        else:
            st.info(
                "Model images not available. Please ensure structural model images are in the working directory."
            )
    except Exception as e:
        st.error(f"Error loading model image: {str(e)}")


def display_current_model_info(grupo, escenario, n_bootstrap):
    """Información del modelo - HTML CORREGIDO"""
    st.markdown("---")

    # Header corregido
    context_html = """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 600;">
            🎯 Configuración del Modelo PLS-SEM Activo
        </h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
            Bootstrap Resampling con Ajustes Contextuales de Escenario
        </p>
    </div>
    """

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # ECUACIÓN DENTRO DEL MISMO RECUADRO
        context_html = f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 2rem; border-radius: 12px; border-left: 5px solid #007bff;
                    box-shadow: 0 4px 15px rgba(0,123,255,0.1); margin-bottom: 1rem;">
            <h3 style="color: #495057; margin: 0 0 1rem 0; font-weight: 600;">
                📊 Ecuación Estructural Activa
            </h3>
            <div style="background: #2d3748; color: #e2e8f0; padding: 1rem; border-radius: 6px; 
                        font-family: 'Courier New', monospace; font-size: 14px; margin: 1rem 0;">
                {MODELOS_COEFICIENTES[grupo]['ecuacion']}
            </div>
            <div style="background: rgba(0,123,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong style="color: #007bff;">💡 Metodología:</strong> Bootstrap Resampling con factores de ajuste contextual por escenario económico
            </div>
        </div>
        """
        st.markdown(context_html, unsafe_allow_html=True)

    with col2:
        # Profile corregido
        grupo_stats = MODELOS_COEFICIENTES[grupo]["grupo_stats"]
        grupo_nombre = (
            "Ahorradores Masculinos" if grupo == "Hah" else "Ahorradoras Femeninas"
        )

        context_html = f"""
        <div style="background: linear-gradient(135deg, #6f42c1 0%, #563d7c 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;
                    box-shadow: 0 8px 20px rgba(111, 66, 193, 0.3); margin-bottom: 1rem;">
            <h3 style="margin: 0 0 1.5rem 0; font-weight: 600;">
                👥 Perfil de Análisis
            </h3>
            <div style="background: rgba(255,255,255,0.15); padding: 1.5rem; border-radius: 10px;">
                <h4 style="margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 500;">
                    {grupo_nombre}
                </h4>
                <div style="display: grid; gap: 0.8rem;">
                    <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 6px;">
                        <strong>PSE Media:</strong> {grupo_stats['PSE_mean']:.4f}
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 6px;">
                        <strong>PCA Media:</strong> {grupo_stats['PCA_mean']:.4f}
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 6px;">
                        <strong>DH Media:</strong> {grupo_stats['DH_mean']:.4f}
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(context_html, unsafe_allow_html=True)

    with col3:
        escenario_info = ESCENARIOS_ECONOMICOS[escenario]

        context_html = f"""
        <div style="background: linear-gradient(135deg, {escenario_info['color']} 0%, {escenario_info['color']}CC 100%); 
                    padding: 2rem; border-radius: 15px; color: white; text-align: center;
                    box-shadow: 0 8px 20px rgba(52, 73, 94, 0.3); margin-bottom: 1rem;">
            <h3 style="margin: 0 0 1.5rem 0; font-weight: 600;">
                🌍 Contexto Económico
            </h3>
            <div style="background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">
                    {escenario_info['nombre']}
                </div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    Bootstrap: {n_bootstrap:,} iteraciones
                </div>
            </div>
            </h4>
            <div style="background: rgba(255,255,255,0.15); padding: 1rem; border-radius: 8px;">
                <h4 style="margin: 0 0 1rem 0; font-size: 1.1rem;">🧠 Promedios Cognitivos</h4>
                <div style="display: grid; gap: 0.6rem; font-size: 0.9rem;">
                    <div style="background: rgba(255,255,255,0.1); padding: 0.6rem; border-radius: 4px;">
                        <strong>CS:</strong> {grupo_stats['CS_mean']:.4f}
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 0.6rem; border-radius: 4px;">
                        <strong>AV:</strong> {grupo_stats['AV_mean']:.4f}
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 0.6rem; border-radius: 4px;">
                        <strong>SQ:</strong> {grupo_stats['SQ_mean']:.4f}
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(context_html, unsafe_allow_html=True)
        # Métricas del modelo
        model_stats = MODELOS_COEFICIENTES[grupo]
        context_html = f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
                        box-shadow: 0 6px 18px rgba(40, 167, 69, 0.3);">
                <h4 style="margin: 0 0 1rem 0; font-weight: 600;">📈 Métricas del Modelo</h4>
                <div style="display: grid; gap: 0.5rem; font-size: 0.85rem;">
                    <div><strong>R²:</strong> {model_stats['r2']:.4f}</div>
                    <div><strong>RMSE:</strong> {model_stats['rmse']:.4f}</div>
                    <div><strong>Correlación:</strong> {model_stats['correlation']:.4f}</div>
                </div>
            </div>
            """
        st.markdown(context_html, unsafe_allow_html=True)


def execute_bootstrap_analysis(grupo, escenario, n_bootstrap, analisis_comparativo):
    """Ejecuta el análisis Bootstrap"""
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        current_params = {
            "grupo": grupo,
            "escenario": escenario,
            "n_bootstrap": n_bootstrap,
            "analisis_comparativo": analisis_comparativo,
        }

        escenario_info = ESCENARIOS_ECONOMICOS[escenario]
        button_text = f"**EXECUTE BOOTSTRAP ANALYSIS**" + (
            f" - {escenario_info['nombre']}"
            if not analisis_comparativo
            else " - MULTI-SCENARIO"
        )

        if st.button(
            button_text,
            type="primary",
            use_container_width=True,
            key="execute_bootstrap",
        ):
            if analisis_comparativo:
                st.markdown("### Executing Multi-scenario Bootstrap Analysis...")
                progress_bar = st.progress(0)
                resultados_dict = {}

                for i, esc in enumerate(["baseline", "crisis", "bonanza"]):
                    with st.spinner(
                        f"Bootstrap resampling: {ESCENARIOS_ECONOMICOS[esc]['nombre']}..."
                    ):
                        resultados_dict[esc] = ejecutar_bootstrap_avanzado(
                            grupo, esc, n_bootstrap
                        )
                    progress_bar.progress((i + 1) / 3)

                st.session_state.resultados_dict = resultados_dict
                st.session_state.simulation_completed = True
                st.session_state.current_parameters = current_params

                st.success(
                    f"**Multi-scenario Bootstrap Completed:** {n_bootstrap:,} × 3 iterations"
                )

            else:
                with st.spinner(f"Bootstrap analysis: {escenario_info['nombre']}..."):
                    resultado = ejecutar_bootstrap_avanzado(
                        grupo, escenario, n_bootstrap
                    )

                st.session_state.resultados_dict = {escenario: resultado}
                st.session_state.simulation_completed = True
                st.session_state.current_parameters = current_params

                st.success(
                    f"**Single Bootstrap Analysis Completed:** {n_bootstrap:,} iterations"
                )
        return True
    return False


def display_results():
    """Muestra los resultados del análisis Bootstrap"""
    if (
        not st.session_state.simulation_completed
        or not st.session_state.resultados_dict
    ):
        return

    # Sección de descarga
    display_download_section()

    # Análisis de resultados
    if len(st.session_state.resultados_dict) > 1:
        display_multi_scenario_analysis()
    else:
        display_single_scenario_analysis()


def display_download_section():
    """Muestra la sección de descarga de resultados"""
    st.markdown("---")
    st.markdown(
        """
    <div class="download-section">
        <h3 style='margin: 0 0 1rem 0;'>Bootstrap Results Download</h3>
        <p style='margin: 0; opacity: 0.9;'>Download comprehensive Excel with Bootstrap statistics, confidence intervals, and bias corrections</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_download1, col_download2, col_download3 = st.columns([1, 2, 1])

    with col_download2:
        if st.button(
            "Generate & Download Bootstrap Excel Report",
            type="secondary",
            use_container_width=True,
            key="download_excel",
        ):
            with st.spinner("Generating Bootstrap Excel report..."):
                excel_buffer, filename = crear_excel_bootstrap_completo(
                    st.session_state.resultados_dict,
                    st.session_state.current_parameters,
                )

                st.download_button(
                    label=f"Download {filename}",
                    data=excel_buffer,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="download_button",
                )

                st.success(
                    "Bootstrap Excel report generated! Contains all resampling data and statistical inference."
                )


def display_multi_scenario_analysis():
    """Muestra análisis multi-escenario - CORREGIDO"""

    # Verificar datos antes de crear tabs
    if not st.session_state.resultados_dict:
        st.error("No bootstrap results available.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Bootstrap Dashboard",
            "Confidence Intervals",
            "Bias Correction",
            "Economic Impact",
            "Bootstrap Diagnostics",
            "3D Analysis",
            "Quadrant Analysis",
            "Descriptive Analysis",
        ]
    )

    with tab1:
        st.markdown("### Bootstrap Comparative Analysis Dashboard")
        try:
            fig_dashboard = crear_dashboard_bootstrap_comparativo(
                st.session_state.resultados_dict
            )
            st.plotly_chart(fig_dashboard, use_container_width=True)

            # Métricas Bootstrap comparativas
            display_bootstrap_comparative_metrics()
        except Exception as e:
            st.error(f"Error creating dashboard: {str(e)}")

    with tab2:
        try:
            display_confidence_intervals_analysis()
        except Exception as e:
            st.error(f"Error in confidence intervals: {str(e)}")

    with tab3:
        try:
            display_bias_correction_analysis()
        except Exception as e:
            st.error(f"Error in bias correction: {str(e)}")

    with tab4:
        # Esta es la pestaña problemática - usar try/except
        try:
            display_economic_impact_analysis()
        except Exception as e:
            st.error(f"Error in economic analysis: {str(e)}")
            st.info("Try refreshing the page or re-running the bootstrap analysis.")

    with tab5:
        try:
            display_bootstrap_diagnostics_analysis()
        except Exception as e:
            st.error(f"Error in diagnostics: {str(e)}")

    # En las funciones display_multi_scenario_analysis() y display_single_scenario_analysis()
    # Modificar las pestañas problemáticas:

    with tab6:  # Análisis 3D
        try:
            with st.container():
                st.markdown(
                    "**Nota:** Visualizaciones optimizadas para mejor rendimiento"
                )
                configurar_interfaz_3d(st.session_state.resultados_dict)
        except Exception as e:
            st.error(f"Error en análisis 3D: {str(e)}")
            st.info("Intenta recargar la página si el problema persiste.")

    with tab7:  # Cuadrantes
        try:
            with st.container():
                st.markdown(
                    "**Nota:** Usando muestra optimizada para análisis de cuadrantes"
                )

                crear_analisis_cuadrantes_completo(st.session_state.resultados_dict)
        except Exception as e:
            st.error(f"Error en análisis de cuadrantes: {str(e)}")
            st.info("Intenta seleccionar un escenario diferente.")

    with tab8:  # Análisis descriptivo
        try:
            cargar_datos_participantes()
            crear_analisis_participantes()
        except Exception as e:
            st.error(f"Error en análisis descriptivo: {str(e)}")


def display_bootstrap_comparative_metrics():
    """Muestra métricas comparativas sin causar rerun"""

    st.markdown("### Bootstrap Statistics Summary")
    col1, col2, col3 = st.columns(3)

    for i, (esc, col) in enumerate(
        zip(["baseline", "crisis", "bonanza"], [col1, col2, col3])
    ):
        with col:
            if (
                esc in st.session_state.resultados_dict
                and "bootstrap_stats" in st.session_state.resultados_dict[esc]
            ):
                stats = st.session_state.resultados_dict[esc]["bootstrap_stats"]

                st.markdown(
                    f"""
                <div style="background: {ESCENARIOS_ECONOMICOS[esc]['color']}20; 
                            padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: {ESCENARIOS_ECONOMICOS[esc]['color']};">
                        {ESCENARIOS_ECONOMICOS[esc]['nombre']}
                    </h4>
                    <p><strong>Bootstrap Mean:</strong> {stats.get('pca_mean', 0):.4f}</p>
                    <p><strong>Bootstrap Std:</strong> {stats.get('pca_std', 0):.4f}</p>
                    <p><strong>95% CI:</strong> [{stats.get('pca_ci_lower', 0):.3f}, {stats.get('pca_ci_upper', 0):.3f}]</p>
                    <p><strong>Iterations:</strong> {stats.get('bootstrap_n', 0):,}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )


def display_confidence_intervals_analysis():
    """Muestra análisis de intervalos de confianza"""
    st.markdown("### Bootstrap Confidence Intervals Analysis")

    # import plotly.graph_objects as go

    # Gráfico de intervalos de confianza
    fig_ci = go.Figure()

    scenarios = ["baseline", "crisis", "bonanza"]
    colors = ["blue", "red", "green"]

    for i, esc in enumerate(scenarios):
        if (
            esc in st.session_state.resultados_dict
            and "bootstrap_stats" in st.session_state.resultados_dict[esc]
        ):
            stats = st.session_state.resultados_dict[esc]["bootstrap_stats"]
            mean_val = stats.get("pca_mean", 0)
            ci_lower = stats.get("pca_ci_lower", 0)
            ci_upper = stats.get("pca_ci_upper", 0)

            fig_ci.add_trace(
                go.Scatter(
                    x=[ESCENARIOS_ECONOMICOS[esc]["nombre"]],
                    y=[mean_val],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        arrayminus=[mean_val - ci_lower],
                        array=[ci_upper - mean_val],
                    ),
                    mode="markers",
                    marker=dict(size=15, color=colors[i]),
                    name=f"{esc.title()} CI",
                    showlegend=True,
                )
            )

    fig_ci.update_layout(
        title="95% Bootstrap Confidence Intervals by Scenario",
        yaxis_title="PCA Value",
        height=500,
    )

    st.plotly_chart(fig_ci, use_container_width=True)

    # Tabla de intervalos detallada
    st.markdown("#### Detailed Confidence Intervals")
    ci_data = []
    for esc in scenarios:
        if (
            esc in st.session_state.resultados_dict
            and "bootstrap_stats" in st.session_state.resultados_dict[esc]
        ):
            stats = st.session_state.resultados_dict[esc]["bootstrap_stats"]
            ci_data.append(
                {
                    "Scenario": ESCENARIOS_ECONOMICOS[esc]["nombre"],
                    "Bootstrap_Mean": stats.get("pca_mean", 0),
                    "CI_Lower_2.5%": stats.get("pca_ci_lower", 0),
                    "CI_Upper_97.5%": stats.get("pca_ci_upper", 0),
                    "CI_Width": stats.get("pca_ci_upper", 0)
                    - stats.get("pca_ci_lower", 0),
                }
            )

    if ci_data:
        ci_df = pd.DataFrame(ci_data)
        st.dataframe(ci_df.round(4), use_container_width=True)


def display_bias_correction_analysis():
    """Muestra análisis de corrección de sesgo"""
    st.markdown("### Bootstrap Bias Correction Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Bias Correction Comparison")
        bias_data = []
        scenarios = ["baseline", "crisis", "bonanza"]
        for esc in scenarios:
            if (
                esc in st.session_state.resultados_dict
                and "bootstrap_stats" in st.session_state.resultados_dict[esc]
            ):
                stats = st.session_state.resultados_dict[esc]["bootstrap_stats"]
                bootstrap_mean = stats.get("pca_mean", 0)
                bias_corrected = stats.get("bias_corrected_mean", 0)
                bias = bootstrap_mean - bias_corrected

                bias_data.append(
                    {
                        "Scenario": ESCENARIOS_ECONOMICOS[esc]["nombre"],
                        "Bootstrap_Mean": bootstrap_mean,
                        "Bias_Corrected_Mean": bias_corrected,
                        "Estimated_Bias": bias,
                        "Bias_Percentage": (
                            (bias / bootstrap_mean) * 100 if bootstrap_mean != 0 else 0
                        ),
                    }
                )

        if bias_data:
            bias_df = pd.DataFrame(bias_data)
            st.dataframe(bias_df.round(4), use_container_width=True)

    with col2:
        st.markdown("#### Bootstrap vs Bias-Corrected")
        if bias_data:
            import plotly.graph_objects as go

            fig_bias = go.Figure()

            scenarios_names = [d["Scenario"] for d in bias_data]
            bootstrap_means = [d["Bootstrap_Mean"] for d in bias_data]
            bias_corrected_means = [d["Bias_Corrected_Mean"] for d in bias_data]

            fig_bias.add_trace(
                go.Bar(
                    x=scenarios_names,
                    y=bootstrap_means,
                    name="Bootstrap Mean",
                    marker_color="lightblue",
                    opacity=0.7,
                )
            )

            fig_bias.add_trace(
                go.Bar(
                    x=scenarios_names,
                    y=bias_corrected_means,
                    name="Bias-Corrected Mean",
                    marker_color="darkblue",
                    opacity=0.7,
                )
            )

            fig_bias.update_layout(
                title="Bootstrap vs Bias-Corrected Estimates",
                yaxis_title="PCA Value",
                barmode="group",
                height=400,
            )

            st.plotly_chart(fig_bias, use_container_width=True)


def display_economic_impact_analysis():
    """Análisis de impacto económico - SOLUCIÓN DEFINITIVA SIN SALIDAS"""
    st.markdown("### Economic Models Bootstrap Impact Analysis")

    if not st.session_state.resultados_dict:
        st.warning("No hay datos de bootstrap disponibles")
        return

    # USAR CONTAINER FIJO SIN SELECTBOX QUE CAUSE PROBLEMAS
    with st.container():
        st.markdown("#### 📊 Análisis por Modelo Económico")

        # CREAR TABS EN LUGAR DE SELECTBOX
        modelos = list(MODELOS_EXTERNOS.keys())
        tabs = st.tabs([MODELOS_EXTERNOS[m]["nombre"] for m in modelos])

        for i, (modelo_key, tab) in enumerate(zip(modelos, tabs)):
            with tab:
                mostrar_analisis_modelo_individual(
                    modelo_key, MODELOS_EXTERNOS[modelo_key]
                )


def mostrar_analisis_modelo_individual(modelo_key, modelo_info):
    """Muestra análisis individual de un modelo sin causar redirects"""

    # Información del modelo
    st.markdown(
        f"""
    <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #007bff;">
        <h4 style="color: #495057; margin: 0 0 1rem 0;">{modelo_info['nombre']}</h4>
        <div style="background: white; padding: 1rem; border-radius: 6px;">
            <p style="margin: 0.5rem 0;"><strong>Fórmula Original:</strong> <code>{modelo_info['original']}</code></p>
            <p style="margin: 0.5rem 0 0 0;"><strong>Con Ajuste PCA:</strong> <code>{modelo_info['con_pca']}</code></p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # Gráfico del modelo específico
        crear_grafico_modelo_fijo(modelo_key, modelo_info)

    with col2:
        # Estadísticas del modelo
        crear_estadisticas_modelo_fijo(modelo_key)


def crear_grafico_modelo_fijo(modelo_key, modelo_info):
    """Crea gráfico fijo para un modelo específico"""

    fig = go.Figure()
    colores = {"baseline": "#34495e", "crisis": "#e74c3c", "bonanza": "#27ae60"}

    for esc in ["baseline", "crisis", "bonanza"]:
        if esc in st.session_state.resultados_dict:
            resultados = st.session_state.resultados_dict[esc]
            if modelo_key in resultados["modelos_externos"]:
                datos = resultados["modelos_externos"][modelo_key]["con_pca"]

                fig.add_trace(
                    go.Box(
                        y=datos,
                        name=ESCENARIOS_ECONOMICOS[esc]["nombre"],
                        marker_color=colores[esc],
                        opacity=0.7,
                        boxmean="sd",
                    )
                )

    fig.update_layout(
        title=f'Distribución Bootstrap - {modelo_info["nombre"]}',
        yaxis_title="Ahorro Proyectado",
        height=400,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"modelo_{modelo_key}")


def crear_estadisticas_modelo_fijo(modelo_key):
    """Crea estadísticas fijas para un modelo"""

    st.markdown("**Estadísticas de Impacto:**")

    estadisticas = []
    for esc in ["baseline", "crisis", "bonanza"]:
        if esc in st.session_state.resultados_dict:
            resultados = st.session_state.resultados_dict[esc]
            if modelo_key in resultados["modelos_externos"]:
                original = np.mean(
                    resultados["modelos_externos"][modelo_key]["original"]
                )
                pca = np.mean(resultados["modelos_externos"][modelo_key]["con_pca"])
                impacto = ((pca - original) / original) * 100

                # Calcular intervalo de confianza del impacto
                original_data = resultados["modelos_externos"][modelo_key]["original"]
                pca_data = resultados["modelos_externos"][modelo_key]["con_pca"]

                impact_values = [
                    (p - o) / o * 100 for p, o in zip(pca_data, original_data) if o != 0
                ]
                if impact_values:
                    ci_lower = np.percentile(impact_values, 2.5)
                    ci_upper = np.percentile(impact_values, 97.5)
                else:
                    ci_lower = ci_upper = 0

                estadisticas.append(
                    {
                        "Escenario": ESCENARIOS_ECONOMICOS[esc]["nombre"],
                        "Impacto_Medio_%": round(impacto, 2),
                        "CI_Inferior_%": round(ci_lower, 2),
                        "CI_Superior_%": round(ci_upper, 2),
                        "Direccion": "Positivo" if impacto > 0 else "Negativo",
                        "Magnitud": (
                            "Alto"
                            if abs(impacto) > 15
                            else "Moderado" if abs(impacto) > 5 else "Bajo"
                        ),
                    }
                )

    if estadisticas:
        df = pd.DataFrame(estadisticas)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Interpretación adicional
        impacto_promedio = np.mean([stat["Impacto_Medio_%"] for stat in estadisticas])
        if abs(impacto_promedio) > 10:
            st.success(f"**Impacto Significativo:** {impacto_promedio:+.1f}% promedio")
        elif abs(impacto_promedio) > 5:
            st.info(f"**Impacto Moderado:** {impacto_promedio:+.1f}% promedio")
        else:
            st.warning(f"**Impacto Bajo:** {impacto_promedio:+.1f}% promedio")

    else:
        st.info("No hay datos disponibles para este modelo.")


# FUNCIÓN ADICIONAL: Matriz comparativa de todos los modelos
def crear_matriz_comparativa_modelos():
    """Crea matriz comparativa de impacto de todos los modelos"""

    st.markdown("---")
    st.markdown("#### 📊 Matriz Comparativa de Impacto Económico")

    matriz_impactos = []

    for modelo_key, modelo_info in MODELOS_EXTERNOS.items():
        fila_modelo = {"Modelo": modelo_info["nombre"]}

        for esc in ["baseline", "crisis", "bonanza"]:
            if esc in st.session_state.resultados_dict:
                resultados = st.session_state.resultados_dict[esc]
                if modelo_key in resultados["modelos_externos"]:
                    original = np.mean(
                        resultados["modelos_externos"][modelo_key]["original"]
                    )
                    pca = np.mean(resultados["modelos_externos"][modelo_key]["con_pca"])
                    impacto = ((pca - original) / original) * 100
                    fila_modelo[f"{esc.title()}_%"] = round(impacto, 2)
                else:
                    fila_modelo[f"{esc.title()}_%"] = 0.0

        # Calcular impacto promedio
        impactos = [
            fila_modelo.get(f"{esc.title()}_%", 0)
            for esc in ["baseline", "crisis", "bonanza"]
        ]
        fila_modelo["Promedio_%"] = round(np.mean(impactos), 2)
        fila_modelo["Volatilidad"] = round(np.std(impactos), 2)

        matriz_impactos.append(fila_modelo)

    if matriz_impactos:
        df_matriz = pd.DataFrame(matriz_impactos)

        # Aplicar formato condicional
        def colorear_impacto(val):
            if isinstance(val, (int, float)):
                if val > 10:
                    return "background-color: #d4edda; color: #155724"  # Verde
                elif val < -10:
                    return "background-color: #f8d7da; color: #721c24"  # Rojo
                elif abs(val) > 5:
                    return "background-color: #fff3cd; color: #856404"  # Amarillo
            return ""

        # Aplicar estilo solo a columnas de porcentaje
        cols_porcentaje = [col for col in df_matriz.columns if "%" in col]
        styled_df = df_matriz.style.applymap(colorear_impacto, subset=cols_porcentaje)

        st.dataframe(styled_df, use_container_width=True)

        # Análisis de la matriz
        st.markdown("**💡 Análisis de la Matriz:**")

        # Encontrar modelo con mayor impacto promedio
        max_impacto_idx = df_matriz["Promedio_%"].abs().idxmax()
        modelo_max_impacto = df_matriz.loc[max_impacto_idx]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Modelo Más Sensible",
                modelo_max_impacto["Modelo"],
                f"{modelo_max_impacto['Promedio_%']:+.1f}%",
            )

        with col2:
            crisis_impactos = [row["Crisis_%"] for row in matriz_impactos]
            crisis_promedio = np.mean(crisis_impactos)
            st.metric(
                "Impacto Crisis Promedio",
                f"{crisis_promedio:+.1f}%",
                "Todos los modelos",
            )

        with col3:
            volatilidades = [row["Volatilidad"] for row in matriz_impactos]
            vol_promedio = np.mean(volatilidades)
            st.metric(
                "Volatilidad Promedio", f"{vol_promedio:.1f}%", "Entre escenarios"
            )


# COMPLETAR LA FUNCIÓN PRINCIPAL display_economic_impact_analysis()
def display_economic_impact_analysis():
    """Análisis de impacto económico - VERSIÓN COMPLETA"""
    st.markdown("### Economic Models Bootstrap Impact Analysis")

    if not st.session_state.resultados_dict:
        st.warning("No hay datos de bootstrap disponibles")
        return

    # USAR CONTAINER FIJO SIN SELECTBOX QUE CAUSE PROBLEMAS
    with st.container():
        st.markdown("#### 📊 Análisis por Modelo Económico")

        # CREAR TABS EN LUGAR DE SELECTBOX
        modelos = list(MODELOS_EXTERNOS.keys())
        tabs = st.tabs([MODELOS_EXTERNOS[m]["nombre"] for m in modelos])

        for i, (modelo_key, tab) in enumerate(zip(modelos, tabs)):
            with tab:
                mostrar_analisis_modelo_individual(
                    modelo_key, MODELOS_EXTERNOS[modelo_key]
                )

        # AÑADIR MATRIZ COMPARATIVA AL FINAL
        crear_matriz_comparativa_modelos()


# FUNCIÓN ADICIONAL: Resumen ejecutivo de modelos
def crear_resumen_ejecutivo_modelos():
    """Crea resumen ejecutivo del impacto de sesgos en modelos económicos"""

    st.markdown("#### 📋 Resumen Ejecutivo: Impacto de Sesgos Cognitivos")

    # Calcular estadísticas agregadas
    todos_impactos = []
    impactos_por_escenario = {"baseline": [], "crisis": [], "bonanza": []}

    for modelo_key in MODELOS_EXTERNOS.keys():
        for esc in ["baseline", "crisis", "bonanza"]:
            if esc in st.session_state.resultados_dict:
                resultados = st.session_state.resultados_dict[esc]
                if modelo_key in resultados["modelos_externos"]:
                    original = np.mean(
                        resultados["modelos_externos"][modelo_key]["original"]
                    )
                    pca = np.mean(resultados["modelos_externos"][modelo_key]["con_pca"])
                    impacto = ((pca - original) / original) * 100
                    todos_impactos.append(impacto)
                    impactos_por_escenario[esc].append(impacto)

    if todos_impactos:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            impacto_general = np.mean(todos_impactos)
            st.metric(
                "Impacto General", f"{impacto_general:+.1f}%", "Todos los modelos"
            )

        with col2:
            if impactos_por_escenario["crisis"]:
                impacto_crisis = np.mean(impactos_por_escenario["crisis"])
                st.metric(
                    "Impacto en Crisis",
                    f"{impacto_crisis:+.1f}%",
                    delta=f"vs General: {impacto_crisis - impacto_general:+.1f}%",
                )

        with col3:
            if impactos_por_escenario["bonanza"]:
                impacto_bonanza = np.mean(impactos_por_escenario["bonanza"])
                st.metric(
                    "Impacto en Bonanza",
                    f"{impacto_bonanza:+.1f}%",
                    delta=f"vs General: {impacto_bonanza - impacto_general:+.1f}%",
                )

        with col4:
            volatilidad_general = np.std(todos_impactos)
            st.metric("Volatilidad", f"{volatilidad_general:.1f}%", "Desv. estándar")

        # Conclusiones
        st.markdown("**🔍 Conclusiones Clave:**")

        if abs(impacto_general) > 15:
            conclusion_general = "Los sesgos cognitivos tienen un **impacto muy significativo** en las predicciones de los modelos económicos clásicos."
        elif abs(impacto_general) > 8:
            conclusion_general = "Los sesgos cognitivos muestran un **impacto considerable** en las predicciones económicas."
        else:
            conclusion_general = "Los sesgos cognitivos tienen un **impacto moderado** en los modelos económicos."

        st.info(conclusion_general)

        if impactos_por_escenario["crisis"] and impactos_por_escenario["bonanza"]:
            crisis_vs_bonanza = np.mean(impactos_por_escenario["crisis"]) - np.mean(
                impactos_por_escenario["bonanza"]
            )
            if abs(crisis_vs_bonanza) > 5:
                st.warning(
                    f"**Diferencial Crisis-Bonanza:** {crisis_vs_bonanza:+.1f}% - Los escenarios económicos **amplifican diferencialmente** el impacto de los sesgos."
                )


def mostrar_analisis_modelo_detallado(modelo_key, modelo_info):
    """Muestra análisis detallado de un modelo económico específico"""

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Bootstrap Distribution by Scenario")

        # Crear gráfico sin keys conflictivas
        fig_econ = go.Figure()

        colores_escenarios = {
            "baseline": "#34495e",
            "crisis": "#e74c3c",
            "bonanza": "#27ae60",
        }

        for esc in ["baseline", "crisis", "bonanza"]:
            if esc in st.session_state.resultados_dict:
                resultados = st.session_state.resultados_dict[esc]
                if modelo_key in resultados["modelos_externos"]:
                    datos_modelo = resultados["modelos_externos"][modelo_key]["con_pca"]

                    fig_econ.add_trace(
                        go.Box(
                            y=datos_modelo,
                            name=ESCENARIOS_ECONOMICOS[esc]["nombre"],
                            boxmean="sd",
                            marker_color=colores_escenarios[esc],
                            opacity=0.7,
                        )
                    )

        fig_econ.update_layout(
            title=f'Bootstrap Distribution - {modelo_info["nombre"]}',
            yaxis_title="Projected Saving",
            height=450,
            showlegend=True,
        )

        # Usar container para evitar conflictos
        with st.container():
            st.plotly_chart(fig_econ, use_container_width=True)

    with col2:
        st.markdown("#### Impact Statistics")

        # Calcular estadísticas sin usar keys problemáticas
        crear_tabla_estadisticas_modelo(modelo_key)

    # Análisis comparativo horizontal
    st.markdown("---")
    st.markdown("#### Comparative Impact Analysis")

    col_comp1, col_comp2 = st.columns(2)

    with col_comp1:
        mostrar_grafico_impacto_comparativo(modelo_key)

    with col_comp2:
        mostrar_metricas_impacto(modelo_key)


def crear_tabla_estadisticas_modelo(modelo_key):
    """Crea tabla de estadísticas para un modelo específico"""

    model_stats = []

    for esc in ["baseline", "crisis", "bonanza"]:
        if esc in st.session_state.resultados_dict:
            resultados = st.session_state.resultados_dict[esc]
            if modelo_key in resultados["modelos_externos"]:
                original_data = resultados["modelos_externos"][modelo_key]["original"]
                pca_data = resultados["modelos_externos"][modelo_key]["con_pca"]

                original_mean = float(np.mean(original_data))
                pca_mean = float(np.mean(pca_data))
                impact_pct = (
                    float(((pca_mean - original_mean) / original_mean) * 100)
                    if original_mean != 0
                    else 0.0
                )

                # Calcular intervalo de confianza del impacto
                impact_values = [
                    (p - o) / o * 100 for p, o in zip(pca_data, original_data) if o != 0
                ]
                if impact_values:
                    ci_lower = float(np.percentile(impact_values, 2.5))
                    ci_upper = float(np.percentile(impact_values, 97.5))
                    std_impact = float(np.std(impact_values))
                else:
                    ci_lower = ci_upper = std_impact = 0.0

                model_stats.append(
                    {
                        "Scenario": str(ESCENARIOS_ECONOMICOS[esc]["nombre"]),
                        "Impact_Mean_%": impact_pct,
                        "CI_Lower_%": ci_lower,
                        "CI_Upper_%": ci_upper,
                        "Bootstrap_Std_%": std_impact,
                        "Direction": str("Positive" if impact_pct > 0 else "Negative"),
                    }
                )

    if model_stats:
        model_df = pd.DataFrame(model_stats)
        # Formatear números para mejor visualización
        numeric_cols = ["Impact_Mean_%", "CI_Lower_%", "CI_Upper_%", "Bootstrap_Std_%"]
        for col in numeric_cols:
            model_df[col] = model_df[col].round(2)

        st.dataframe(model_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data available for this model.")


def mostrar_grafico_impacto_comparativo(modelo_key):
    """Muestra gráfico de impacto comparativo entre escenarios"""

    st.markdown("**Impact by Scenario**")

    escenarios = []
    impactos = []
    colores = []

    color_map = {"baseline": "#34495e", "crisis": "#e74c3c", "bonanza": "#27ae60"}

    for esc in ["baseline", "crisis", "bonanza"]:
        if esc in st.session_state.resultados_dict:
            resultados = st.session_state.resultados_dict[esc]
            if modelo_key in resultados["modelos_externos"]:
                original = np.mean(
                    resultados["modelos_externos"][modelo_key]["original"]
                )
                pca_enhanced = np.mean(
                    resultados["modelos_externos"][modelo_key]["con_pca"]
                )
                impact_pct = (
                    ((pca_enhanced - original) / original) * 100 if original != 0 else 0
                )

                escenarios.append(ESCENARIOS_ECONOMICOS[esc]["nombre"])
                impactos.append(impact_pct)
                colores.append(color_map[esc])

    if escenarios:
        fig_impact = go.Figure(
            data=[
                go.Bar(
                    x=escenarios,
                    y=impactos,
                    marker_color=colores,
                    text=[f"{imp:+.1f}%" for imp in impactos],
                    textposition="auto",
                    opacity=0.8,
                )
            ]
        )

        fig_impact.update_layout(
            title=f'PCA Impact on {MODELOS_EXTERNOS[modelo_key]["nombre"]}',
            yaxis_title="Impact Percentage (%)",
            height=300,
            showlegend=False,
        )

        # Añadir línea de referencia en 0
        fig_impact.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig_impact, use_container_width=True)


def mostrar_metricas_impacto(modelo_key):
    """Muestra métricas de impacto agregadas"""

    st.markdown("**Aggregate Metrics**")

    # Calcular métricas agregadas
    all_impacts = []
    all_original = []
    all_pca = []

    for esc in st.session_state.resultados_dict.values():
        if modelo_key in esc["modelos_externos"]:
            original_data = esc["modelos_externos"][modelo_key]["original"]
            pca_data = esc["modelos_externos"][modelo_key]["con_pca"]

            for orig, pca in zip(original_data, pca_data):
                if orig != 0:
                    impact_pct = ((pca - orig) / orig) * 100
                    all_impacts.append(impact_pct)
                    all_original.append(orig)
                    all_pca.append(pca)

    if all_impacts:
        avg_impact = np.mean(all_impacts)
        std_impact = np.std(all_impacts)
        max_impact = np.max(all_impacts)
        min_impact = np.min(all_impacts)

        # Mostrar métricas como tarjetas
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.metric("Average Impact", f"{avg_impact:+.2f}%")
            st.metric("Max Impact", f"{max_impact:+.2f}%")

        with col_m2:
            st.metric("Impact Std", f"{std_impact:.2f}%")
            st.metric("Min Impact", f"{min_impact:+.2f}%")

        # Interpretación del impacto
        if abs(avg_impact) > 15:
            interpretation = "🔴 High Impact"
        elif abs(avg_impact) > 5:
            interpretation = "🟡 Moderate Impact"
        else:
            interpretation = "🟢 Low Impact"

        st.markdown(f"**Interpretation:** {interpretation}")


def create_economic_impact_matrix():
    """Crear matriz de impacto económico separada para evitar conflictos de estado"""
    impact_matrix = []

    for escenario, resultados in st.session_state.resultados_dict.items():
        for modelo_key, modelo_info_matrix in MODELOS_EXTERNOS.items():
            if modelo_key in resultados["modelos_externos"]:
                original_mean = np.mean(
                    resultados["modelos_externos"][modelo_key]["original"]
                )
                pca_mean = np.mean(
                    resultados["modelos_externos"][modelo_key]["con_pca"]
                )
                impact_pct = ((pca_mean - original_mean) / original_mean) * 100

                impact_matrix.append(
                    {
                        "Scenario": str(escenario.title()),
                        "Economic_Model": str(modelo_info_matrix["nombre"]),
                        "Original_Mean_Saving": float(original_mean),
                        "PCA_Enhanced_Mean_Saving": float(pca_mean),
                        "Absolute_Impact": float(pca_mean - original_mean),
                        "Relative_Impact_%": float(impact_pct),
                        "Impact_Direction": str(
                            "Positive" if impact_pct > 0 else "Negative"
                        ),
                    }
                )

    if impact_matrix:
        impact_df = pd.DataFrame(impact_matrix)

        # Función de coloreo para la matriz
        def color_impact_safe(val):
            try:
                if isinstance(val, (int, float)) and not pd.isna(val):
                    if val > 10:
                        return "background-color: #27ae60; color: white"
                    elif val < -10:
                        return "background-color: #e74c3c; color: white"
                    elif abs(val) > 5:
                        return "background-color: #f39c12; color: white"
            except:
                pass
            return ""

        styled_df = impact_df.style.applymap(
            color_impact_safe, subset=["Relative_Impact_%"]
        )
        st.dataframe(styled_df, use_container_width=True)


def display_bootstrap_diagnostics_analysis():
    """Muestra análisis de diagnósticos Bootstrap - CORREGIDO"""
    st.markdown("### Bootstrap Diagnostics & Validation")

    # Usar container para mantener estado
    with st.container():
        # Seleccionar escenario para diagnósticos detallados con key único
        escenario_diag = st.selectbox(
            "Select scenario for detailed Bootstrap diagnostics:",
            options=list(st.session_state.resultados_dict.keys()),
            format_func=lambda x: ESCENARIOS_ECONOMICOS[x]["nombre"],
            key="selectbox_diagnostics_escenario",  # KEY ÚNICO
        )

        if escenario_diag in st.session_state.resultados_dict:
            resultados_diag = st.session_state.resultados_dict[escenario_diag]

            # Crear gráfico de diagnósticos Bootstrap
            fig_diagnostics = crear_grafico_bootstrap_diagnostics(resultados_diag)
            st.plotly_chart(
                fig_diagnostics,
                use_container_width=True,
                key=f"diagnostics_{escenario_diag}",
            )

            # Estadísticas de diagnóstico
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Bootstrap Quality Metrics")

                pca_values = np.array(resultados_diag["pca_values"])

                # Test de normalidad en muestra Bootstrap
                if len(pca_values) > 5000:
                    sample_for_test = np.random.choice(pca_values, 5000, replace=False)
                else:
                    sample_for_test = pca_values

                import scipy.stats as sp_stats

                shapiro_stat, shapiro_p = sp_stats.shapiro(sample_for_test)

                # Estadísticas de convergencia
                n_samples = len(pca_values)
                se_bootstrap = np.std(pca_values) / np.sqrt(n_samples)

                quality_metrics = pd.DataFrame(
                    [
                        ["Bootstrap Iterations", int(len(pca_values))],
                        ["Bootstrap SE", float(se_bootstrap)],
                        ["Shapiro-Wilk Stat", float(shapiro_stat)],
                        ["Shapiro p-value", float(shapiro_p)],
                        [
                            "Distribution",
                            str("Normal" if shapiro_p > 0.05 else "Non-normal"),
                        ],
                        ["Effective Sample Size", int(len(np.unique(pca_values)))],
                    ],
                    columns=["Metric", "Value"],
                )

                # Convertir tipos explícitamente para evitar error Arrow
                quality_metrics["Metric"] = quality_metrics["Metric"].astype(str)
                quality_metrics["Value"] = quality_metrics["Value"].astype(str)

                st.dataframe(quality_metrics, use_container_width=True)

            with col2:
                st.markdown("#### Convergence Analysis")

                # Análisis de convergencia Bootstrap
                window_size = min(100, len(pca_values) // 20)
                if window_size > 1:
                    moving_std = pd.Series(pca_values).rolling(window=window_size).std()
                    final_stability = np.std(
                        moving_std.dropna().tail(len(moving_std) // 4)
                    )

                    convergence_metrics = pd.DataFrame(
                        [
                            ["Final Mean", float(np.mean(pca_values[-500:]))],
                            ["Final Std", float(np.std(pca_values[-500:]))],
                            ["Moving Std Stability", float(final_stability)],
                            [
                                "Convergence Ratio",
                                float(final_stability / np.std(pca_values)),
                            ],
                            ["Bootstrap Error", float(se_bootstrap)],
                        ],
                        columns=["Metric", "Value"],
                    )

                    # Convertir tipos explícitamente
                    convergence_metrics["Metric"] = convergence_metrics[
                        "Metric"
                    ].astype(str)
                    convergence_metrics["Value"] = (
                        convergence_metrics["Value"].round(6).astype(str)
                    )

                    st.dataframe(convergence_metrics, use_container_width=True)


def display_single_scenario_analysis():
    """Muestra análisis de escenario único"""
    st.markdown("### Single Scenario Bootstrap Analysis Results")

    escenario_key = list(st.session_state.resultados_dict.keys())[0]
    resultados = st.session_state.resultados_dict[escenario_key]

    # Mostrar estadísticas Bootstrap básicas
    if "bootstrap_stats" in resultados:
        stats = resultados["bootstrap_stats"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Bootstrap Mean PCA", f"{stats.get('pca_mean', 0):.4f}")
            st.metric("Bootstrap Std", f"{stats.get('pca_std', 0):.4f}")

        with col2:
            st.metric("CI Lower (2.5%)", f"{stats.get('pca_ci_lower', 0):.4f}")
            st.metric("CI Upper (97.5%)", f"{stats.get('pca_ci_upper', 0):.4f}")

        with col3:
            st.metric(
                "Bias-Corrected Mean", f"{stats.get('bias_corrected_mean', 0):.4f}"
            )
            bootstrap_se = stats.get("pca_std", 0) / np.sqrt(
                stats.get("bootstrap_n", 1)
            )
            st.metric("Bootstrap SE", f"{bootstrap_se:.6f}")

        with col4:
            st.metric("Bootstrap Iterations", f"{stats.get('bootstrap_n', 0):,}")
            original_n = stats.get("original_n", 0)
            st.metric("Original Sample N", f"{original_n}")

    # Pestañas para análisis detallado de escenario único
    tab1, tab2, tab3 = st.tabs(
        [
            "Análisis Bootstrap Detallado",
            "📊 Visualización 3D",
            "📍 Análisis Cuadrantes",  # NUEVA PESTAÑA
        ]
    )

    with tab1:
        display_single_scenario_detailed_analysis(resultados)

    with tab2:
        # Análisis 3D para escenario único
        configurar_interfaz_3d(st.session_state.resultados_dict)

    with tab3:
        # Análisis de cuadrantes para escenario único
        crear_analisis_cuadrantes_completo(st.session_state.resultados_dict)


def display_single_scenario_detailed_analysis(resultados):
    """Análisis detallado para escenario único"""

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Gráficos para escenario único
    fig_single_bootstrap = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Bootstrap PCA Distribution",
            "Bootstrap Confidence Interval",
            "Economic Models Bootstrap Impact",
            "Bootstrap Convergence",
        ],
    )

    # 1. Distribución Bootstrap
    fig_single_bootstrap.add_trace(
        go.Histogram(
            x=resultados["pca_values"], nbinsx=40, name="Bootstrap PCA", opacity=0.7
        ),
        row=1,
        col=1,
    )

    # 2. Intervalo de confianza
    if "bootstrap_stats" in resultados:
        stats = resultados["bootstrap_stats"]
        mean_val = stats.get("pca_mean", 0)
        ci_lower = stats.get("pca_ci_lower", 0)
        ci_upper = stats.get("pca_ci_upper", 0)

        fig_single_bootstrap.add_trace(
            go.Scatter(
                x=["Bootstrap Estimate"],
                y=[mean_val],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    arrayminus=[mean_val - ci_lower],
                    array=[ci_upper - mean_val],
                ),
                mode="markers",
                marker=dict(size=15, color="blue"),
                name="95% CI",
            ),
            row=1,
            col=2,
        )

    # 3. Impacto en modelos económicos
    modelo_impactos = []
    modelo_nombres = []

    for modelo_key, modelo_info_single in MODELOS_EXTERNOS.items():
        if modelo_key in resultados["modelos_externos"]:
            original_mean = np.mean(
                resultados["modelos_externos"][modelo_key]["original"]
            )
            pca_mean = np.mean(resultados["modelos_externos"][modelo_key]["con_pca"])
            impact_pct = ((pca_mean - original_mean) / original_mean) * 100

            modelo_impactos.append(impact_pct)
            modelo_nombres.append(modelo_key)

    if modelo_impactos:
        fig_single_bootstrap.add_trace(
            go.Bar(
                x=modelo_nombres,
                y=modelo_impactos,
                name="Economic Impact %",
                marker_color="green",
            ),
            row=2,
            col=1,
        )

    # 4. Convergencia Bootstrap
    pca_values = np.array(resultados["pca_values"])
    cumulative_means = np.cumsum(pca_values) / np.arange(1, len(pca_values) + 1)
    sample_indices = np.arange(1, len(pca_values) + 1)

    # Submuestrear para claridad visual
    step = max(1, len(sample_indices) // 200)

    fig_single_bootstrap.add_trace(
        go.Scatter(
            x=sample_indices[::step],
            y=cumulative_means[::step],
            mode="lines",
            name="Cumulative Mean",
            line=dict(color="red", width=2),
        ),
        row=2,
        col=2,
    )

    fig_single_bootstrap.update_layout(
        height=800, showlegend=True, title_text=f"Bootstrap Analysis - Single Scenario"
    )

    st.plotly_chart(fig_single_bootstrap, use_container_width=True)

    # Matriz de impacto de todos los modelos económicos
    st.markdown("#### Economic Models Impact Matrix")

    impact_matrix = []
    for modelo_key, modelo_info_matrix in MODELOS_EXTERNOS.items():
        if modelo_key in resultados["modelos_externos"]:
            original_mean = np.mean(
                resultados["modelos_externos"][modelo_key]["original"]
            )
            pca_mean = np.mean(resultados["modelos_externos"][modelo_key]["con_pca"])
            impact_pct = ((pca_mean - original_mean) / original_mean) * 100

            impact_matrix.append(
                {
                    "Economic_Model": modelo_info_matrix["nombre"],
                    "Original_Mean_Saving": original_mean,
                    "PCA_Enhanced_Mean_Saving": pca_mean,
                    "Absolute_Impact": pca_mean - original_mean,
                    "Relative_Impact_%": impact_pct,
                    "Impact_Direction": "Positive" if impact_pct > 0 else "Negative",
                }
            )

    if impact_matrix:
        impact_df = pd.DataFrame(impact_matrix)

        # Colorear según impacto
        def color_impact(val):
            if isinstance(val, (int, float)):
                if val > 10:
                    return "background-color: #27ae60; color: white"
                elif val < -10:
                    return "background-color: #e74c3c; color: white"
                elif abs(val) > 5:
                    return "background-color: #f39c12; color: white"
            return ""

        styled_df = impact_df.style.applymap(color_impact, subset=["Relative_Impact_%"])
        st.dataframe(styled_df, use_container_width=True)


def display_footer():
    """Muestra el footer de la aplicación"""
    st.markdown(
        """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: #f9f9f9;
            padding: 5px 0;
            text-align: center;
            color: #666;
            font-size: 0.85em;
            border-top: 1px solid #ddd;
            line-height: 1.2;
        }
        .footer p {
            margin: 2px 0;
        }
    </style>
    <div class="footer">
        <p><strong>Simulador PCA - Tesis Doctoral</strong></p>
        <p>Desarrollado con PLS-SEM y Bootstrap en Python©</p>
        <p>Por MSc. Jesús F. Salazar Rojas</p>
        <p>Propensión Conductual al Ahorro (PCA) © 2025</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
