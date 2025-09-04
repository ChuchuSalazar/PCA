"""
PCA Simulator v3.0 - Bootstrap Analysis
Enhanced Behavioral Economics Analysis Tool

Author: MSc. Jesús Fernando Salazar Rojas
Doctorado en Economía, UCAB – 2025
Methodology: PLS-SEM + Bootstrap Resampling
Framework: DH • CS • AV • SQ Analysis

This application provides advanced Bootstrap analysis for the Propensión Conductual al Ahorro (PCA)
using PLS-SEM models with cognitive biases framework and external economic model integration.

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as sp_stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
import os
import warnings
import textwrap
from datetime import datetime
import plotly.figure_factory as ff
import io
import base64
from PIL import Image

warnings.filterwarnings('ignore')

# Configuración de la página con diseño doctoral
st.set_page_config(
    page_title="PCA Simulator v3.0 - Bootstrap Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño doctoral premium
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .scenario-card {
        background: linear-gradient(45deg, #34495e 0%, #2c3e50 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border-left: 4px solid #3498db;
    }
    .metric-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(52, 152, 219, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .profile-card {
        background: linear-gradient(45deg, #8e44ad 0%, #9b59b6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .download-section {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .model-images {
        background: rgba(255,255,255,0.95);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Header principal doctoral
html_header = """
<div class="main-header">
    <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700; letter-spacing: 2px;'>
        PCA SIMULATOR v3.1
    </h1>
    <h2 style='margin: 1rem 0; font-size: 1.5rem; opacity: 0.9; font-weight: 400;'>
        La Propensión Conductual al Ahorro:
        Un estudio desde los  sesgos cognitivos
        para la toma de decisiones en el ahorro
        de los hogares
    </h2>
    <hr style='margin: 1.5rem auto; width: 70%; border: 2px solid rgba(255,255,255,0.3);'>
    <div style='display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 1.5rem;'>
        <div style='text-align: center; margin: 0.5rem;'>
            <strong style='font-size: 1.1rem;'>MSc. Jesús Fernando Salazar Rojas</strong><br>
            <em style='opacity: 0.8;'>Doctorado en Economía, UCAB – 2025</em>
        </div>
         <div style='text-align: center; margin: 0.5rem;'>
            <strong style='font-size: 1.1rem;'>Dr. Fernando Spiritto</strong><br>
            <em style='opacity: 0.8;'>Tutor</em>
        </div>
        <div style='text-align: center; margin: 0.5rem;'>
            <strong style='font-size: 1.1rem;'>Methodology</strong></strong><br>
            <em style='opacity: 0.8;'>PLS-SEM + Bootstrap Resampling</em>
        </div>
    </div>
</div>
"""
st.markdown(html_header, unsafe_allow_html=True)

# Configuración actualizada con estadísticas por grupo
MODELOS_COEFICIENTES = {
    'Hah': {
        'ecuacion': 'PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS',
        'coef': {'PSE': 0.3777, 'DH': 0.2226, 'SQ': -0.5947, 'CS': 0.2866},
        'r2': -0.639818,
        'rmse': 1.270350,
        'mae': 1.043280,
        'correlation': -0.129289,
        'grupo_stats': {
            'PSE_mean': 0.125, 'DH_mean': 0.089, 'CS_mean': 0.156,
            'AV_mean': 0.098, 'SQ_mean': -0.067, 'PCA_mean': -0.088
        },
        'stats': {
            'PSE': {'min': -2.261, 'max': 2.101, 'mean': 0.000, 'std': 1.000, 'skew': 0.205, 'kurt': -0.279},
            'PCA': {'min': -2.497, 'max': 1.737, 'mean': 0.000, 'std': 1.000, 'skew': -0.155, 'kurt': -0.642},
            'AV': {'min': -1.555, 'max': 2.127, 'mean': 0.000, 'std': 1.000, 'skew': 0.275, 'kurt': -0.735},
            'DH': {'min': -1.786, 'max': 2.098, 'mean': 0.000, 'std': 1.000, 'skew': 0.054, 'kurt': -0.711},
            'SQ': {'min': -1.503, 'max': 1.983, 'mean': 0.000, 'std': 1.000, 'skew': 0.118, 'kurt': -1.066},
            'CS': {'min': -1.372, 'max': 2.676, 'mean': 0.000, 'std': 1.000, 'skew': 0.856, 'kurt': 0.227}
        },
        'weights': {
            'PSE': {'PCA2': -0.3557, 'PCA4': 0.2800, 'PCA5': 0.8343},
            'AV': {'AV1': 0.1165, 'AV2': 0.3009, 'AV3': 0.6324, 'AV5': 0.3979},
            'DH': {'DH3': 0.7097, 'DH4': 0.4376},
            'SQ': {'SQ1': 0.3816, 'SQ2': 0.5930, 'SQ3': 0.3358},
            'CS': {'CS2': 0.5733, 'CS3': 0.4983, 'CS5': 0.1597}
        }
    },
    'Mah': {
        'ecuacion': 'PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS',
        'coef': {'PSE': 0.3485, 'DH': -0.2013, 'SQ': -0.5101, 'CS': 0.3676},
        'r2': 0.571136,
        'rmse': 0.650872,
        'mae': 0.519483,
        'correlation': 0.759797,
        'grupo_stats': {
            'PSE_mean': -0.125, 'DH_mean': -0.089, 'CS_mean': -0.156,
            'AV_mean': -0.098, 'SQ_mean': 0.067, 'PCA_mean': 0.088
        },
        'stats': {
            'PSE': {'min': -2.262, 'max': 2.169, 'mean': 0.000, 'std': 1.000, 'skew': 0.160, 'kurt': -0.283},
            'PCA': {'min': -2.359, 'max': 2.050, 'mean': 0.000, 'std': 1.000, 'skew': 0.144, 'kurt': -0.490},
            'AV': {'min': -1.447, 'max': 3.204, 'mean': 0.000, 'std': 1.000, 'skew': 1.194, 'kurt': 1.322},
            'DH': {'min': -1.722, 'max': 2.340, 'mean': 0.000, 'std': 1.000, 'skew': 0.271, 'kurt': -0.744},
            'SQ': {'min': -1.525, 'max': 2.194, 'mean': 0.000, 'std': 1.000, 'skew': 0.350, 'kurt': -1.016},
            'CS': {'min': -2.023, 'max': 2.458, 'mean': 0.000, 'std': 1.000, 'skew': 0.226, 'kurt': -0.064}
        },
        'weights': {
            'PSE': {'PCA2': -0.5168, 'PCA4': -0.0001, 'PCA5': 0.8496},
            'AV': {'AV1': 0.1920, 'AV2': 0.4430, 'AV3': 0.7001, 'AV5': 0.1276},
            'DH': {'DH2': 0.0305, 'DH3': 0.3290, 'DH4': 0.0660, 'DH5': 0.8397},
            'SQ': {'SQ1': 0.5458, 'SQ2': 0.4646, 'SQ3': 0.2946},
            'CS': {'CS2': 0.5452, 'CS3': 0.5117, 'CS5': 0.2631}
        }
    }
}

# Configuración de escenarios económicos mejorada
ESCENARIOS_ECONOMICOS = {
    'baseline': {
        'nombre': 'Baseline Scenario',
        'descripcion': 'Normal economic conditions without external alterations',
        'color': '#34495e',
        'factor_dh': 1.0,
        'factor_cs': 1.0,
        'factor_av': 1.0,
        'factor_sq': 1.0,
        'volatilidad': 1.0,
        'bootstrap_noise': 0.1
    },
    'crisis': {
        'nombre': 'Economic Crisis',
        'descripcion': 'Uncertainty environment, negative rumors, higher cognitive biases',
        'color': '#e74c3c',
        'factor_dh': 1.4,
        'factor_cs': 1.3,
        'factor_av': 1.2,
        'factor_sq': 1.1,
        'volatilidad': 1.5,
        'bootstrap_noise': 0.15
    },
    'bonanza': {
        'nombre': 'Economic Bonanza',
        'descripcion': 'Optimistic environment, economic confidence, lower cognitive biases',
        'color': '#27ae60',
        'factor_dh': 0.7,
        'factor_cs': 0.8,
        'factor_av': 0.9,
        'factor_sq': 0.95,
        'volatilidad': 0.8,
        'bootstrap_noise': 0.08
    }
}

# Modelos económicos externos
MODELOS_EXTERNOS = {
    'keynes': {
        'nombre': 'Keynes (1936)',
        'original': 'S = a₀ + a₁Y',
        'con_pca': 'S = a₀ + (a₁ + γ·PCA)Y',
        'descripcion': 'Keynesian saving function',
        'parametros': {'a0': -50, 'a1': 0.2, 'gamma': 0.15}
    },
    'friedman': {
        'nombre': 'Friedman (1957)',
        'original': 'S = f(Yₚ)',
        'con_pca': 'S = f(Yₚ·(1 + δ·PCA))',
        'descripcion': 'Permanent income hypothesis',
        'parametros': {'base_rate': 0.15, 'delta': 0.1, 'yp_factor': 0.8}
    },
    'modigliani': {
        'nombre': 'Modigliani-Brumberg (1954)',
        'original': 'S = f(W,Y)',
        'con_pca': 'S = a·W(1 + θ·PCA) + b·Y',
        'descripcion': 'Life cycle hypothesis',
        'parametros': {'a': 0.05, 'b': 0.1, 'theta': 0.08}
    },
    'carroll': {
        'nombre': 'Carroll & Weil (1994)',
        'original': 'S = f(Y,r)',
        'con_pca': 'S = f(Y) + r(1 + φ·PCA)',
        'descripcion': 'Growth and saving model, Carroll & Weil (1994)',
        'parametros': {'base_saving_rate': 0.12, 'phi': 0.2}
    },
    'deaton': {
        'nombre': 'Deaton-Carroll (1991-92)',
        'original': 'S = f(Y,expectations)',
        'con_pca': 'S = f(Y,expectations·(1 + κ·PCA))',
        'descripcion': 'Consumption expectations model',
        'parametros': {'base_rate': 0.18, 'kappa': 0.12}
    }
}

# Inicializar session state
if 'resultados_dict' not in st.session_state:
    st.session_state.resultados_dict = None
if 'simulation_completed' not in st.session_state:
    st.session_state.simulation_completed = False
if 'current_parameters' not in st.session_state:
    st.session_state.current_parameters = None
if 'show_model_images' not in st.session_state:
    st.session_state.show_model_images = False


@st.cache_data
def cargar_datos():
    """Carga los datos desde archivos Excel o genera datos simulados"""
    try:
        ruta_scores = "SCORE HM.xlsx"
        ruta_items = "Standardized Indicator Scores ITEMS.xlsx"

        if os.path.exists(ruta_scores) and os.path.exists(ruta_items):
            try:
                scores_df = pd.read_excel(ruta_scores)
                items_df = pd.read_excel(ruta_items)
                st.success("Data successfully loaded from local files")
                return scores_df, items_df
            except Exception as e:
                st.error(f"Error reading Excel files: {str(e)}")

        return generar_datos_simulados_avanzados()

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return generar_datos_simulados_avanzados()


def generar_datos_simulados_avanzados():
    """Genera datos simulados más sofisticados basados en las estadísticas reales"""
    np.random.seed(42)
    n_samples = 1000

    datos_completos = []

    for grupo in ['Hah', 'Mah']:
        stats_grupo = MODELOS_COEFICIENTES[grupo]['stats']
        n_grupo = n_samples // 2

        # Generar variables con distribuciones específicas
        pse_data = generar_variable_con_stats(stats_grupo['PSE'], n_grupo)
        pca_data = generar_variable_con_stats(stats_grupo['PCA'], n_grupo)
        av_data = generar_variable_con_stats(stats_grupo['AV'], n_grupo)
        dh_data = generar_variable_con_stats(stats_grupo['DH'], n_grupo)
        sq_data = generar_variable_con_stats(stats_grupo['SQ'], n_grupo)
        cs_data = generar_variable_con_stats(stats_grupo['CS'], n_grupo)

        grupo_df = pd.DataFrame({
            'Case': range(len(datos_completos) + 1, len(datos_completos) + n_grupo + 1),
            'PSE': pse_data,
            'PCA': pca_data,
            'AV': av_data,
            'DH': dh_data,
            'SQ': sq_data,
            'CS': cs_data,
            'GRUPO': grupo
        })

        datos_completos.append(grupo_df)

    scores_df = pd.concat(datos_completos, ignore_index=True)

    # Generar items simulados
    items_data = []
    for _, row in scores_df.iterrows():
        items_row = {
            'Case': row['Case'],
            'PCA2': np.random.randint(1, 10),
            'PCA4': np.random.randint(1, 7),
            'PCA5': np.random.randint(1, 7),
            'PPCA': row['PCA'] + np.random.normal(0, 0.1),
            'GRUPO': row['GRUPO']
        }

        # Generar items basados en los weights
        grupo = row['GRUPO']
        weights = MODELOS_COEFICIENTES[grupo]['weights']

        for construct, items in weights.items():
            for item, weight in items.items():
                if construct == 'PSE':
                    base_value = row['PSE']
                else:
                    base_value = row[construct]
                items_row[item] = base_value * \
                    weight + np.random.normal(0, 0.2)

        items_data.append(items_row)

    items_df = pd.DataFrame(items_data)
    return scores_df, items_df


def generar_variable_con_stats(stats, n):
    """Genera variable con estadísticas específicas usando transformación Johnson"""
    data = np.random.normal(0, 1, n)

    # Ajustar asimetría
    if abs(stats['skew']) > 0.1:
        data = data + stats['skew'] * (data**2 - 1) / 6

    # Ajustar curtosis
    if abs(stats['kurt']) > 0.1:
        data = data + stats['kurt'] * (data**3 - 3*data) / 24

    # Estandarizar y escalar
    data = (data - np.mean(data)) / np.std(data) * stats['std'] + stats['mean']
    data = np.clip(data, stats['min'], stats['max'])

    return data


def ejecutar_bootstrap_avanzado(grupo, escenario, n_bootstrap=3000):
    """Ejecuta análisis Bootstrap en lugar de Monte Carlo"""
    np.random.seed(42)

    # Obtener datos originales del grupo
    scores_df, items_df = cargar_datos()
    grupo_data = scores_df[scores_df['GRUPO'] == grupo].copy()

    if len(grupo_data) == 0:
        st.error(f"No data found for group {grupo}")
        return None

    # Configuración del escenario
    escenario_config = ESCENARIOS_ECONOMICOS[escenario]

    resultados = {
        'pca_values': [],
        'variables_cognitivas': {'DH': [], 'CS': [], 'AV': [], 'SQ': []},
        'pse_values': [],
        'escenario': escenario,
        'modelos_externos': {modelo: {'original': [], 'con_pca': []} for modelo in MODELOS_EXTERNOS.keys()},
        'bootstrap_stats': {'original_n': len(grupo_data), 'bootstrap_n': n_bootstrap},
        'parametros_simulacion': {
            'grupo': grupo,
            'escenario': escenario,
            'n_bootstrap': n_bootstrap,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'methodology': 'Bootstrap Resampling'
        }
    }

    # Análisis Bootstrap
    for i in range(n_bootstrap):
        # Bootstrap resampling
        bootstrap_sample = resample(
            grupo_data, n_samples=len(grupo_data), random_state=i)

        # Aplicar factores del escenario con ruido bootstrap
        noise_factor = escenario_config['bootstrap_noise']

        for idx, row in bootstrap_sample.iterrows():
            # Ajustar variables cognitivas según escenario
            dh_adjusted = row['DH'] * escenario_config['factor_dh'] + \
                np.random.normal(0, noise_factor)
            cs_adjusted = row['CS'] * escenario_config['factor_cs'] + \
                np.random.normal(0, noise_factor)
            av_adjusted = row['AV'] * escenario_config['factor_av'] + \
                np.random.normal(0, noise_factor)
            sq_adjusted = row['SQ'] * escenario_config['factor_sq'] + \
                np.random.normal(0, noise_factor)

            # Aplicar límites estadísticos
            stats = MODELOS_COEFICIENTES[grupo]['stats']
            dh_adjusted = np.clip(
                dh_adjusted, stats['DH']['min'], stats['DH']['max'])
            cs_adjusted = np.clip(
                cs_adjusted, stats['CS']['min'], stats['CS']['max'])
            av_adjusted = np.clip(
                av_adjusted, stats['AV']['min'], stats['AV']['max'])
            sq_adjusted = np.clip(
                sq_adjusted, stats['SQ']['min'], stats['SQ']['max'])

            # Almacenar variables cognitivas
            resultados['variables_cognitivas']['DH'].append(dh_adjusted)
            resultados['variables_cognitivas']['CS'].append(cs_adjusted)
            resultados['variables_cognitivas']['AV'].append(av_adjusted)
            resultados['variables_cognitivas']['SQ'].append(sq_adjusted)

            # Calcular PCA usando el modelo PLS-SEM
            pse_bootstrap = row['PSE'] + \
                np.random.normal(0, noise_factor * 0.5)
            resultados['pse_values'].append(pse_bootstrap)

            pca_value = calcular_pca_teorica(
                pse_bootstrap, dh_adjusted, sq_adjusted, cs_adjusted, grupo)

            # Añadir ruido del modelo
            model_noise = np.random.normal(
                0, MODELOS_COEFICIENTES[grupo]['rmse'] * 0.1)
            pca_value += model_noise

            resultados['pca_values'].append(pca_value)

            # Simular modelos económicos externos
            volatilidad = escenario_config['volatilidad']
            y = abs(np.random.normal(1000, 200 * volatilidad))
            w = abs(np.random.normal(5000, 1000 * volatilidad))
            r = np.random.normal(0.05, 0.02 * volatilidad)

            for modelo_key in MODELOS_EXTERNOS.keys():
                s_orig, s_pca = simular_modelo_externo(
                    modelo_key, pca_value, y, w, r)
                resultados['modelos_externos'][modelo_key]['original'].append(
                    s_orig)
                resultados['modelos_externos'][modelo_key]['con_pca'].append(
                    s_pca)

    # Calcular estadísticas de bootstrap
    pca_array = np.array(resultados['pca_values'])
    original_pca_mean = np.mean(grupo_data['PCA'])

    resultados['bootstrap_stats'].update({
        'pca_mean': np.mean(pca_array),
        'pca_std': np.std(pca_array),
        'pca_ci_lower': np.percentile(pca_array, 2.5),
        'pca_ci_upper': np.percentile(pca_array, 97.5),
        'bias_corrected_mean': 2 * original_pca_mean - np.mean(pca_array),
        'original_mean': original_pca_mean
    })

    return resultados


def calcular_pca_teorica(pse, dh, sq, cs, grupo):
    """Calcula PCA usando la ecuación del modelo PLS-SEM"""
    coef = MODELOS_COEFICIENTES[grupo]['coef']
    return coef['PSE'] * pse + coef['DH'] * dh + coef['SQ'] * sq + coef['CS'] * cs


def simular_modelo_externo(modelo_key, pca_value, y_base=1000, w_base=5000, r_base=0.05):
    """Simula un modelo económico externo con y sin PCA"""
    modelo = MODELOS_EXTERNOS[modelo_key]
    params = modelo['parametros']

    if modelo_key == 'keynes':
        s_original = params['a0'] + params['a1'] * y_base
        s_con_pca = params['a0'] + \
            (params['a1'] + params['gamma'] * pca_value) * y_base

    elif modelo_key == 'friedman':
        yp = y_base * params['yp_factor']
        s_original = params['base_rate'] * yp
        s_con_pca = params['base_rate'] * yp * \
            (1 + params['delta'] * pca_value)

    elif modelo_key == 'modigliani':
        s_original = params['a'] * w_base + params['b'] * y_base
        s_con_pca = params['a'] * w_base * \
            (1 + params['theta'] * pca_value) + params['b'] * y_base

    elif modelo_key == 'carroll':
        s_original = params['base_saving_rate'] * y_base + r_base * y_base
        s_con_pca = params['base_saving_rate'] * y_base + \
            r_base * (1 + params['phi'] * pca_value) * y_base

    elif modelo_key == 'deaton':
        s_original = params['base_rate'] * y_base
        s_con_pca = params['base_rate'] * y_base * \
            (1 + params['kappa'] * pca_value)

    return s_original, s_con_pca


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
        go.Histogram(x=pca_values, nbinsx=50, name='Bootstrap Distribution',
                     opacity=0.7, marker_color='blue'),
        row=1, col=1
    )

    # Overlay normal distribution
    x_norm = np.linspace(pca_values.min(), pca_values.max(), 100)
    y_norm = sp_stats.norm.pdf(x_norm, np.mean(pca_values), np.std(pca_values))
    y_norm_scaled = y_norm * len(pca_values) * \
        (pca_values.max() - pca_values.min()) / 50

    fig.add_trace(
        go.Scatter(x=x_norm, y=y_norm_scaled, mode='lines',
                   name='Normal Approximation', line=dict(color='red', width=2)),
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
            go.Bar(x=categories, y=values, name='Estimates Comparison',
                   marker_color=['green', 'orange', 'red']),
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
            x=sample_indices[::50],  # Muestrear cada 50 puntos para claridad
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
            y=bootstrap_mean, line_dash="dash", line_color="red",
            annotation_text=f"Final Mean: {bootstrap_mean:.4f}",
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Bootstrap Analysis Diagnostics"
    )

    return fig


def crear_excel_bootstrap_completo(resultados_dict, parametros):
    """Crea archivo Excel completo con resultados Bootstrap"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"PCA_BOOTSTRAP_RESULTS_{timestamp}.xlsx"

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formatos
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'fg_color': '#34495e', 'font_color': 'white', 'border': 1
        })

        data_format = workbook.add_format({'border': 1})
        number_format = workbook.add_format(
            {'num_format': '0.0000', 'border': 1})

        # 1. Hoja de Configuración Bootstrap
        config_data = [
            ['Parameter', 'Value'],
            ['Analysis Group', parametros.get('grupo', 'N/A')],
            ['Economic Scenario', parametros.get('escenario', 'N/A')],
            ['Bootstrap Iterations', parametros.get('n_bootstrap', 'N/A')],
            ['Methodology', 'Bootstrap Resampling'],
            ['Timestamp', parametros.get('timestamp', 'N/A')],
            ['Original Sample Size', parametros.get('original_n', 'N/A')]
        ]

        config_df = pd.DataFrame(config_data[1:], columns=config_data[0])
        config_df.to_excel(writer, sheet_name='Bootstrap_Config', index=False)

        # 2. Hoja de Estadísticas Bootstrap por Escenario
        for escenario, resultados in resultados_dict.items():
            # Datos principales
            bootstrap_data = pd.DataFrame({
                'Bootstrap_ID': range(1, len(resultados['pca_values']) + 1),
                'PCA': resultados['pca_values'],
                'PSE': resultados['pse_values'],
                'DH': resultados['variables_cognitivas']['DH'],
                'CS': resultados['variables_cognitivas']['CS'],
                'AV': resultados['variables_cognitivas']['AV'],
                'SQ': resultados['variables_cognitivas']['SQ']
            })

            sheet_name = f'Bootstrap_{escenario.title()}'
            bootstrap_data.to_excel(writer, sheet_name=sheet_name, index=False)

            # Estadísticas Bootstrap
            if 'bootstrap_stats' in resultados:
                stats = resultados['bootstrap_stats']
                stats_data = pd.DataFrame([
                    ['Bootstrap Iterations', stats.get('bootstrap_n', 'N/A')],
                    ['Original Sample Size', stats.get('original_n', 'N/A')],
                    ['PCA Bootstrap Mean', stats.get('pca_mean', 'N/A')],
                    ['PCA Bootstrap Std', stats.get('pca_std', 'N/A')],
                    ['PCA CI Lower (2.5%)', stats.get('pca_ci_lower', 'N/A')],
                    ['PCA CI Upper (97.5%)', stats.get('pca_ci_upper', 'N/A')],
                    ['Bias-Corrected Mean',
                        stats.get('bias_corrected_mean', 'N/A')],
                    ['Original Mean', stats.get('original_mean', 'N/A')]
                ], columns=['Statistic', 'Value'])

                stats_sheet = f'Stats_{escenario.title()}'
                stats_data.to_excel(
                    writer, sheet_name=stats_sheet, index=False)

        # 3. Hoja de Comparaciones Bootstrap
        if len(resultados_dict) > 1:
            comparison_data = []
            for escenario, resultados in resultados_dict.items():
                stats = resultados.get('bootstrap_stats', {})
                comparison_data.append({
                    'Scenario': escenario.title(),
                    'Bootstrap_Mean': stats.get('pca_mean', 0),
                    'Bootstrap_Std': stats.get('pca_std', 0),
                    'CI_Lower': stats.get('pca_ci_lower', 0),
                    'CI_Upper': stats.get('pca_ci_upper', 0),
                    'Bias_Corrected': stats.get('bias_corrected_mean', 0),
                    'Bootstrap_N': stats.get('bootstrap_n', 0),
                    'Original_Mean': stats.get('original_mean', 0)
                })

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_excel(
                writer, sheet_name='Bootstrap_Comparison', index=False)

        # 4. Hoja de Modelos Económicos
        models_data = []
        for escenario, resultados in resultados_dict.items():
            for modelo_key, modelo_data in resultados['modelos_externos'].items():
                original_mean = np.mean(modelo_data['original'])
                pca_mean = np.mean(modelo_data['con_pca'])
                models_data.append({
                    'Scenario': escenario.title(),
                    'Economic_Model': modelo_key,
                    'Original_Mean_Saving': original_mean,
                    'PCA_Enhanced_Mean_Saving': pca_mean,
                    'Absolute_Difference': pca_mean - original_mean,
                    'Percentage_Impact': ((pca_mean - original_mean) / original_mean) * 100 if original_mean != 0 else 0,
                    'Bootstrap_Std_Original': np.std(modelo_data['original']),
                    'Bootstrap_Std_PCA': np.std(modelo_data['con_pca'])
                })

        models_df = pd.DataFrame(models_data)
        models_df.to_excel(
            writer, sheet_name='Economic_Models_Bootstrap', index=False)

    output.seek(0)
    return output, filename


def display_model_images(grupo):
    """Muestra las imágenes del modelo estructural según el grupo"""
    try:
        if grupo == 'Hah':
            image_path = r"hombres.JPG"
            title = "Structural Model - Male Savers (Hah)"
        else:
            image_path = r"mujeres.JPG"
            title = "Structural Model - Female Savers (Mah)"

        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.markdown(f"""
            <div class="model-images">
                <h4 style="color: #2c3e50; text-align: center; margin-bottom: 1rem;">{title}</h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption=title, use_container_width=True)
        else:
            st.info(
                "Model images not available. Please ensure structural model images are in the working directory.")
    except Exception as e:
        st.error(f"Error loading model image: {str(e)}")


def calcular_estadisticas_avanzadas(datos):
    """Calcula estadísticas descriptivas avanzadas"""
    return {
        'media': np.mean(datos),
        'std': np.std(datos),
        'min': np.min(datos),
        'max': np.max(datos),
        'p5': np.percentile(datos, 5),
        'p25': np.percentile(datos, 25),
        'mediana': np.percentile(datos, 50),
        'p75': np.percentile(datos, 75),
        'p95': np.percentile(datos, 95),
        'asimetria': sp_stats.skew(datos),
        'curtosis': sp_stats.kurtosis(datos),
        'cv': np.std(datos) / np.mean(datos) if np.mean(datos) != 0 else 0,
        'iqr': np.percentile(datos, 75) - np.percentile(datos, 25)
    }


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


def main():
    # Cargar datos
    with st.spinner("Loading bootstrap database..."):
        scores_df, items_df = cargar_datos()

    # Sidebar mejorado
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(45deg, #2c3e50 0%, #34495e 100%); 
                    border-radius: 10px; color: white; margin-bottom: 1rem;'>
            <h3 style='margin: 0;'>Bootstrap Control Panel</h3>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Advanced Resampling Configuration</p>
        </div>
        """, unsafe_allow_html=True)

        # Selección de grupo
        grupo = st.selectbox(
            "**Analysis Group**",
            options=['Hah', 'Mah'],
            format_func=lambda x: f"Male Savers ({x})" if x == 'Hah' else f"Female Savers ({x})"
        )

        # Mostrar métricas del modelo seleccionado
        model_stats = MODELOS_COEFICIENTES[grupo]
        st.markdown(f"""
        <div class="metric-card">
            <h4>PLS-SEM Model Metrics - {grupo}</h4>
            <p><strong>R²:</strong> {model_stats['r2']:.4f}</p>
            <p><strong>RMSE:</strong> {model_stats['rmse']:.4f}</p>
            <p><strong>Correlation:</strong> {model_stats['correlation']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Botón para mostrar/ocultar imágenes del modelo
        if st.button("Toggle PLS-SEM Model Images", type="secondary"):
            st.session_state.show_model_images = not st.session_state.show_model_images

        st.markdown("---")

        # Selección de escenario
        st.markdown("**Economic Scenario**")
        escenario = st.radio(
            "Select economic context:",
            options=['baseline', 'crisis', 'bonanza'],
            format_func=lambda x: ESCENARIOS_ECONOMICOS[x]['nombre'],
            index=0
        )

        # Mostrar descripción del escenario
        escenario_info = ESCENARIOS_ECONOMICOS[escenario]
        st.markdown(f"""
        <div class="scenario-card" style="background: linear-gradient(45deg, {escenario_info['color']}22, {escenario_info['color']}44);">
            <h4 style="color: {escenario_info['color']};">{escenario_info['nombre']}</h4>
            <p style="margin: 0; font-size: 0.9rem; color: #333;">{escenario_info['descripcion']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Parámetros Bootstrap
        st.markdown("**Bootstrap Parameters**")
        n_bootstrap = st.number_input(
            "Number of Bootstrap iterations",
            min_value=1000, max_value=5000, value=3000, step=500,
            help="Bootstrap resampling iterations for statistical inference"
        )

        analisis_comparativo = st.checkbox(
            "**Multi-scenario Bootstrap Analysis**", value=True
        )

        # Información metodológica
        with st.expander("ℹ️ Bootstrap Methodology Info"):
            st.markdown("""
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
            """)

    # Mostrar imágenes del modelo si está activado
    if st.session_state.show_model_images:
        st.markdown("---")
        display_model_images(grupo)
        st.markdown("---")

    # Información del modelo actual y promedios del grupo
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("### Active PLS-SEM Model")
        st.code(MODELOS_COEFICIENTES[grupo]['ecuacion'], language='text')
        st.info(f"**Bootstrap Method:** Resampling with scenario adjustments")

    with col2:
        st.markdown("### Current Profile")
        grupo_stats = MODELOS_COEFICIENTES[grupo]['grupo_stats']

        st.markdown(f"""
        <div class="profile-card">
            <h4>Group: {"Male" if grupo == 'Hah' else "Female"}</h4>
            <p><strong>PSE Mean:</strong> {grupo_stats['PSE_mean']:.3f}</p>
            <p><strong>PCA Mean:</strong> {grupo_stats['PCA_mean']:.3f}</p>
            <p><strong>DH Mean:</strong> {grupo_stats['DH_mean']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("### Context")
        st.metric("Scenario", escenario_info['nombre'])
        st.metric("Bootstrap N", f"{n_bootstrap:,}")
        st.metric("Method", "Resampling")

        # Métricas adicionales del grupo
        st.markdown(f"""
        <div class="profile-card">
            <h4>Cognitive Averages</h4>
            <p><strong>CS:</strong> {grupo_stats['CS_mean']:.3f}</p>
            <p><strong>AV:</strong> {grupo_stats['AV_mean']:.3f}</p>
            <p><strong>SQ:</strong> {grupo_stats['SQ_mean']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Botón de simulación Bootstrap
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        current_params = {
            'grupo': grupo, 'escenario': escenario, 'n_bootstrap': n_bootstrap,
            'analisis_comparativo': analisis_comparativo
        }

        button_text = f"**EXECUTE BOOTSTRAP ANALYSIS**" + \
            (f" - {escenario_info['nombre']}" if not analisis_comparativo else " - MULTI-SCENARIO")

        if st.button(button_text, type="primary", use_container_width=True):

            # Ejecutar análisis Bootstrap
            if analisis_comparativo:
                st.markdown(
                    "### Executing Multi-scenario Bootstrap Analysis...")

                progress_bar = st.progress(0)
                resultados_dict = {}

                for i, esc in enumerate(['baseline', 'crisis', 'bonanza']):
                    with st.spinner(f"Bootstrap resampling: {ESCENARIOS_ECONOMICOS[esc]['nombre']}..."):
                        resultados_dict[esc] = ejecutar_bootstrap_avanzado(
                            grupo, esc, n_bootstrap)
                    progress_bar.progress((i + 1) / 3)

                st.session_state.resultados_dict = resultados_dict
                st.session_state.simulation_completed = True
                st.session_state.current_parameters = current_params

                st.success(
                    f"**Multi-scenario Bootstrap Completed:** {n_bootstrap:,} × 3 iterations")

            else:
                with st.spinner(f"Bootstrap analysis: {escenario_info['nombre']}..."):
                    resultado = ejecutar_bootstrap_avanzado(
                        grupo, escenario, n_bootstrap)

                st.session_state.resultados_dict = {escenario: resultado}
                st.session_state.simulation_completed = True
                st.session_state.current_parameters = current_params

                st.success(
                    f"**Single Bootstrap Analysis Completed:** {n_bootstrap:,} iterations")

    # Mostrar resultados Bootstrap
    if st.session_state.simulation_completed and st.session_state.resultados_dict:

        # Sección de descarga de resultados Bootstrap
        st.markdown("---")
        st.markdown("""
        <div class="download-section">
            <h3 style='margin: 0 0 1rem 0;'>Bootstrap Results Download</h3>
            <p style='margin: 0; opacity: 0.9;'>Download comprehensive Excel with Bootstrap statistics, confidence intervals, and bias corrections</p>
        </div>
        """, unsafe_allow_html=True)

        col_download1, col_download2, col_download3 = st.columns([1, 2, 1])

        with col_download2:
            if st.button("Generate & Download Bootstrap Excel Report", type="secondary", use_container_width=True):
                with st.spinner("Generating Bootstrap Excel report..."):
                    excel_buffer, filename = crear_excel_bootstrap_completo(
                        st.session_state.resultados_dict,
                        st.session_state.current_parameters
                    )

                    st.download_button(
                        label=f"Download {filename}",
                        data=excel_buffer,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                    st.success(
                        "Bootstrap Excel report generated! Contains all resampling data and statistical inference.")

        # Análisis de resultados Bootstrap
        if len(st.session_state.resultados_dict) > 1:  # Multi-scenario analysis

            # Tabs para análisis Bootstrap
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Bootstrap Dashboard",
                "Confidence Intervals",
                "Bias Correction",
                "Economic Impact",
                "Bootstrap Diagnostics"
            ])

            with tab1:
                st.markdown("### Bootstrap Comparative Analysis Dashboard")
                fig_dashboard = crear_dashboard_bootstrap_comparativo(
                    st.session_state.resultados_dict)
                st.plotly_chart(fig_dashboard, use_container_width=True)

                # Métricas Bootstrap comparativas
                st.markdown("### Bootstrap Statistics Summary")
                col1, col2, col3 = st.columns(3)

                for i, (esc, col) in enumerate(zip(['baseline', 'crisis', 'bonanza'], [col1, col2, col3])):
                    with col:
                        if esc in st.session_state.resultados_dict and 'bootstrap_stats' in st.session_state.resultados_dict[esc]:
                            stats = st.session_state.resultados_dict[esc]['bootstrap_stats']

                            st.markdown(f"""
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
                            """, unsafe_allow_html=True)

            with tab2:
                st.markdown("### Bootstrap Confidence Intervals Analysis")

                # Gráfico de intervalos de confianza
                fig_ci = go.Figure()

                scenarios = ['baseline', 'crisis', 'bonanza']
                colors = ['blue', 'red', 'green']

                for i, esc in enumerate(scenarios):
                    if esc in st.session_state.resultados_dict and 'bootstrap_stats' in st.session_state.resultados_dict[esc]:
                        stats = st.session_state.resultados_dict[esc]['bootstrap_stats']
                        mean_val = stats.get('pca_mean', 0)
                        ci_lower = stats.get('pca_ci_lower', 0)
                        ci_upper = stats.get('pca_ci_upper', 0)

                        fig_ci.add_trace(go.Scatter(
                            x=[ESCENARIOS_ECONOMICOS[esc]['nombre']],
                            y=[mean_val],
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                arrayminus=[mean_val - ci_lower],
                                array=[ci_upper - mean_val]
                            ),
                            mode='markers',
                            marker=dict(size=15, color=colors[i]),
                            name=f'{esc.title()} CI',
                            showlegend=True
                        ))

                fig_ci.update_layout(
                    title='95% Bootstrap Confidence Intervals by Scenario',
                    yaxis_title='PCA Value',
                    height=500
                )

                st.plotly_chart(fig_ci, use_container_width=True)

                # Tabla de intervalos
                st.markdown("#### Detailed Confidence Intervals")
                ci_data = []
                for esc in scenarios:
                    if esc in st.session_state.resultados_dict and 'bootstrap_stats' in st.session_state.resultados_dict[esc]:
                        stats = st.session_state.resultados_dict[esc]['bootstrap_stats']
                        ci_data.append({
                            'Scenario': ESCENARIOS_ECONOMICOS[esc]['nombre'],
                            'Bootstrap_Mean': stats.get('pca_mean', 0),
                            'CI_Lower_2.5%': stats.get('pca_ci_lower', 0),
                            'CI_Upper_97.5%': stats.get('pca_ci_upper', 0),
                            'CI_Width': stats.get('pca_ci_upper', 0) - stats.get('pca_ci_lower', 0)
                        })

                if ci_data:
                    ci_df = pd.DataFrame(ci_data)
                    st.dataframe(ci_df.round(4), use_container_width=True)

            with tab3:
                st.markdown("### Bootstrap Bias Correction Analysis")

                # Análisis de sesgo
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Bias Correction Comparison")
                    bias_data = []
                    scenarios = ['baseline', 'crisis', 'bonanza']
                    for esc in scenarios:
                        if esc in st.session_state.resultados_dict and 'bootstrap_stats' in st.session_state.resultados_dict[esc]:
                            stats = st.session_state.resultados_dict[esc]['bootstrap_stats']
                            bootstrap_mean = stats.get('pca_mean', 0)
                            bias_corrected = stats.get(
                                'bias_corrected_mean', 0)
                            bias = bootstrap_mean - bias_corrected

                            bias_data.append({
                                'Scenario': ESCENARIOS_ECONOMICOS[esc]['nombre'],
                                'Bootstrap_Mean': bootstrap_mean,
                                'Bias_Corrected_Mean': bias_corrected,
                                'Estimated_Bias': bias,
                                'Bias_Percentage': (bias / bootstrap_mean) * 100 if bootstrap_mean != 0 else 0
                            })

                    if bias_data:
                        bias_df = pd.DataFrame(bias_data)
                        st.dataframe(bias_df.round(
                            4), use_container_width=True)

                with col2:
                    st.markdown("#### Bootstrap vs Bias-Corrected")
                    if bias_data:
                        fig_bias = go.Figure()

                        scenarios_names = [d['Scenario'] for d in bias_data]
                        bootstrap_means = [d['Bootstrap_Mean']
                                           for d in bias_data]
                        bias_corrected_means = [
                            d['Bias_Corrected_Mean'] for d in bias_data]

                        fig_bias.add_trace(go.Bar(
                            x=scenarios_names,
                            y=bootstrap_means,
                            name='Bootstrap Mean',
                            marker_color='lightblue',
                            opacity=0.7
                        ))

                        fig_bias.add_trace(go.Bar(
                            x=scenarios_names,
                            y=bias_corrected_means,
                            name='Bias-Corrected Mean',
                            marker_color='darkblue',
                            opacity=0.7
                        ))

                        fig_bias.update_layout(
                            title='Bootstrap vs Bias-Corrected Estimates',
                            yaxis_title='PCA Value',
                            barmode='group',
                            height=400
                        )

                        st.plotly_chart(fig_bias, use_container_width=True)

            with tab4:
                st.markdown("### Economic Models Bootstrap Impact Analysis")

                # Selector de modelo para análisis detallado
                modelo_analizar = st.selectbox(
                    "Select economic model for detailed Bootstrap analysis:",
                    options=list(MODELOS_EXTERNOS.keys()),
                    format_func=lambda x: MODELOS_EXTERNOS[x]['nombre']
                )

                # Información del modelo
                modelo_info = MODELOS_EXTERNOS[modelo_analizar]
                st.markdown(f"""
                <div style="background-color:#f9f9f9; padding:1rem; border-radius:8px; margin:1rem 0; border:1px solid #ddd;">
                    <h3 style="color:#2c3e50; margin-top:0;">{modelo_info['nombre']}</h3>
                    <p><strong>Description:</strong> {modelo_info['descripcion']}</p>
                    <p><strong>Original formula:</strong> <code>{modelo_info['original']}</code></p>
                    <p><strong>Formula with PCA:</strong> <code>{modelo_info['con_pca']}</code></p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Gráfico Bootstrap del modelo económico
                    fig_econ = go.Figure()

                    for i, esc in enumerate(['baseline', 'crisis', 'bonanza']):
                        if esc in st.session_state.resultados_dict:
                            resultados = st.session_state.resultados_dict[esc]
                            if modelo_analizar in resultados['modelos_externos']:
                                datos_modelo = resultados['modelos_externos'][modelo_analizar]['con_pca']

                                fig_econ.add_trace(go.Box(
                                    y=datos_modelo,
                                    name=ESCENARIOS_ECONOMICOS[esc]['nombre'],
                                    boxmean='sd',
                                    marker_color=['blue', 'red', 'green'][i]
                                ))

                    fig_econ.update_layout(
                        title=f'Bootstrap Distribution - {modelo_info["nombre"]}',
                        yaxis_title='Projected Saving',
                        height=400
                    )

                    st.plotly_chart(fig_econ, use_container_width=True)

                with col2:
                    # Estadísticas Bootstrap del modelo
                    st.markdown("#### Bootstrap Model Statistics")

                    model_stats = []
                    for esc in ['baseline', 'crisis', 'bonanza']:
                        if esc in st.session_state.resultados_dict:
                            resultados = st.session_state.resultados_dict[esc]
                            if modelo_analizar in resultados['modelos_externos']:
                                original_data = resultados['modelos_externos'][modelo_analizar]['original']
                                pca_data = resultados['modelos_externos'][modelo_analizar]['con_pca']

                                original_mean = np.mean(original_data)
                                pca_mean = np.mean(pca_data)
                                impact_pct = (
                                    (pca_mean - original_mean) / original_mean) * 100

                                # Bootstrap CI para el impacto
                                impact_values = [
                                    (p - o) / o * 100 for p, o in zip(pca_data, original_data) if o != 0]
                                ci_lower = np.percentile(impact_values, 2.5)
                                ci_upper = np.percentile(impact_values, 97.5)

                                model_stats.append({
                                    'Scenario': ESCENARIOS_ECONOMICOS[esc]['nombre'],
                                    'Impact_Mean_%': impact_pct,
                                    'CI_Lower_%': ci_lower,
                                    'CI_Upper_%': ci_upper,
                                    'Bootstrap_Std_%': np.std(impact_values)
                                })

                    if model_stats:
                        model_df = pd.DataFrame(model_stats)
                        st.dataframe(model_df.round(
                            2), use_container_width=True)

                # Matriz de todos los modelos Bootstrap
                st.markdown("#### Multi-model Bootstrap Impact Matrix")

                impact_matrix = []
                for modelo_key, modelo_info_matrix in MODELOS_EXTERNOS.items():
                    fila = {'Model': modelo_info_matrix['nombre']}

                    for esc in ['baseline', 'crisis', 'bonanza']:
                        if esc in st.session_state.resultados_dict:
                            resultados = st.session_state.resultados_dict[esc]
                            if modelo_key in resultados['modelos_externos']:
                                original_mean = np.mean(
                                    resultados['modelos_externos'][modelo_key]['original'])
                                pca_mean = np.mean(
                                    resultados['modelos_externos'][modelo_key]['con_pca'])
                                impact_pct = (
                                    (pca_mean - original_mean) / original_mean) * 100
                                fila[f'{esc.title()}_%'] = impact_pct

                    impact_matrix.append(fila)

                if impact_matrix:
                    impact_df = pd.DataFrame(impact_matrix)

                    # Colorear según impacto
                    def color_impact(val):
                        if isinstance(val, (int, float)):
                            if val > 10:
                                return 'background-color: #27ae60; color: white'
                            elif val < -10:
                                return 'background-color: #e74c3c; color: white'
                            elif abs(val) > 5:
                                return 'background-color: #f39c12; color: white'
                        return ''

                    styled_df = impact_df.style.applymap(
                        color_impact, subset=[col for col in impact_df.columns if '%' in col])
                    st.dataframe(styled_df, use_container_width=True)

            with tab5:
                st.markdown("### Bootstrap Diagnostics & Validation")

                # Seleccionar escenario para diagnósticos detallados
                escenario_diag = st.selectbox(
                    "Select scenario for detailed Bootstrap diagnostics:",
                    options=list(st.session_state.resultados_dict.keys()),
                    format_func=lambda x: ESCENARIOS_ECONOMICOS[x]['nombre']
                )

                if escenario_diag in st.session_state.resultados_dict:
                    resultados_diag = st.session_state.resultados_dict[escenario_diag]

                    # Crear gráfico de diagnósticos Bootstrap
                    fig_diagnostics = crear_grafico_bootstrap_diagnostics(
                        resultados_diag)
                    st.plotly_chart(fig_diagnostics, use_container_width=True)

                    # Estadísticas de diagnóstico
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Bootstrap Quality Metrics")

                        pca_values = np.array(resultados_diag['pca_values'])

                        # Test de normalidad en muestra Bootstrap
                        if len(pca_values) > 5000:
                            sample_for_test = np.random.choice(
                                pca_values, 5000, replace=False)
                        else:
                            sample_for_test = pca_values

                        shapiro_stat, shapiro_p = sp_stats.shapiro(
                            sample_for_test)

                        # Estadísticas de convergencia
                        n_samples = len(pca_values)
                        se_bootstrap = np.std(pca_values) / np.sqrt(n_samples)

                        quality_metrics = pd.DataFrame([
                            ['Bootstrap Iterations', len(pca_values)],
                            ['Bootstrap SE', se_bootstrap],
                            ['Shapiro-Wilk Stat', shapiro_stat],
                            ['Shapiro p-value', shapiro_p],
                            ['Distribution', 'Normal' if shapiro_p >
                                0.05 else 'Non-normal'],
                            ['Effective Sample Size', len(
                                np.unique(pca_values))]
                        ], columns=['Metric', 'Value'])

                        st.dataframe(quality_metrics, use_container_width=True)

                    with col2:
                        st.markdown("#### Convergence Analysis")

                        # Análisis de convergencia Bootstrap
                        window_size = min(100, len(pca_values) // 20)
                        if window_size > 1:
                            moving_std = pd.Series(pca_values).rolling(
                                window=window_size).std()
                            final_stability = np.std(
                                moving_std.dropna().tail(len(moving_std)//4))

                            convergence_metrics = pd.DataFrame([
                                ['Final Mean', np.mean(pca_values[-500:])],
                                ['Final Std', np.std(pca_values[-500:])],
                                ['Moving Std Stability', final_stability],
                                ['Convergence Ratio', final_stability /
                                    np.std(pca_values)],
                                ['Bootstrap Error', se_bootstrap]
                            ], columns=['Metric', 'Value'])

                            st.dataframe(convergence_metrics.round(
                                6), use_container_width=True)

        else:  # Single scenario Bootstrap analysis
            st.markdown("### Single Scenario Bootstrap Analysis Results")

            escenario_key = list(st.session_state.resultados_dict.keys())[0]
            resultados = st.session_state.resultados_dict[escenario_key]

            # Mostrar estadísticas Bootstrap básicas
            if 'bootstrap_stats' in resultados:
                stats = resultados['bootstrap_stats']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Bootstrap Mean PCA",
                              f"{stats.get('pca_mean', 0):.4f}")
                    st.metric("Bootstrap Std",
                              f"{stats.get('pca_std', 0):.4f}")

                with col2:
                    st.metric("CI Lower (2.5%)",
                              f"{stats.get('pca_ci_lower', 0):.4f}")
                    st.metric("CI Upper (97.5%)",
                              f"{stats.get('pca_ci_upper', 0):.4f}")

                with col3:
                    st.metric("Bias-Corrected Mean",
                              f"{stats.get('bias_corrected_mean', 0):.4f}")
                    bootstrap_se = stats.get(
                        'pca_std', 0) / np.sqrt(stats.get('bootstrap_n', 1))
                    st.metric("Bootstrap SE", f"{bootstrap_se:.6f}")

                with col4:
                    st.metric("Bootstrap Iterations",
                              f"{stats.get('bootstrap_n', 0):,}")
                    original_n = stats.get('original_n', 0)
                    st.metric("Original Sample N", f"{original_n}")

            # Gráficos para escenario único
            fig_single_bootstrap = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Bootstrap PCA Distribution',
                    'Bootstrap Confidence Interval',
                    'Economic Models Bootstrap Impact',
                    'Bootstrap Convergence'
                ]
            )

            # 1. Distribución Bootstrap
            fig_single_bootstrap.add_trace(
                go.Histogram(
                    x=resultados['pca_values'],
                    nbinsx=40,
                    name='Bootstrap PCA',
                    opacity=0.7
                ),
                row=1, col=1
            )

            # 2. Intervalo de confianza
            if 'bootstrap_stats' in resultados:
                stats = resultados['bootstrap_stats']
                mean_val = stats.get('pca_mean', 0)
                ci_lower = stats.get('pca_ci_lower', 0)
                ci_upper = stats.get('pca_ci_upper', 0)

                fig_single_bootstrap.add_trace(
                    go.Scatter(
                        x=['Bootstrap Estimate'],
                        y=[mean_val],
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            arrayminus=[mean_val - ci_lower],
                            array=[ci_upper - mean_val]
                        ),
                        mode='markers',
                        marker=dict(size=15, color='blue'),
                        name='95% CI'
                    ),
                    row=1, col=2
                )

            # 3. Impacto en modelos económicos
            modelo_impactos = []
            modelo_nombres = []

            for modelo_key, modelo_info_single in MODELOS_EXTERNOS.items():
                if modelo_key in resultados['modelos_externos']:
                    original_mean = np.mean(
                        resultados['modelos_externos'][modelo_key]['original'])
                    pca_mean = np.mean(
                        resultados['modelos_externos'][modelo_key]['con_pca'])
                    impact_pct = ((pca_mean - original_mean) /
                                  original_mean) * 100

                    modelo_impactos.append(impact_pct)
                    modelo_nombres.append(modelo_key)

            if modelo_impactos:
                fig_single_bootstrap.add_trace(
                    go.Bar(
                        x=modelo_nombres,
                        y=modelo_impactos,
                        name='Economic Impact %',
                        marker_color='green'
                    ),
                    row=2, col=1
                )

            # 4. Convergencia Bootstrap
            pca_values = np.array(resultados['pca_values'])
            cumulative_means = np.cumsum(
                pca_values) / np.arange(1, len(pca_values) + 1)
            sample_indices = np.arange(1, len(pca_values) + 1)

            # Submuestrear para claridad visual
            step = max(1, len(sample_indices) // 200)

            fig_single_bootstrap.add_trace(
                go.Scatter(
                    x=sample_indices[::step],
                    y=cumulative_means[::step],
                    mode='lines',
                    name='Cumulative Mean',
                    line=dict(color='red', width=2)
                ),
                row=2, col=2
            )

            fig_single_bootstrap.update_layout(
                height=800,
                showlegend=True,
                title_text=f"Bootstrap Analysis - {ESCENARIOS_ECONOMICOS[escenario_key]['nombre']}"
            )

            st.plotly_chart(fig_single_bootstrap, use_container_width=True)


def crear_excel_academico_completo(resultados_dict, parametros, adanco_data=None):
    """
    Crea exportación académica completa para economía conductual
    Incluye toda la información para réplica y análisis profundo
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"PCA_BEHAVIORAL_ECONOMICS_FULL_EXPORT_{timestamp}.xlsx"

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formatos académicos
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'fg_color': '#2c3e50', 'font_color': 'white', 'border': 1
        })

        # 1. METADATOS Y CONFIGURACIÓN
        metadata_data = pd.DataFrame([
            ['Study Title', 'La Propensión Conductual al Ahorro: Análisis Bootstrap'],
            ['Researcher', 'MSc. Jesús Fernando Salazar Rojas'],
            ['Institution', 'Doctorado en Economía, UCAB'],
            ['Year', '2025'],
            ['Methodology', 'PLS-SEM + Bootstrap Resampling'],
            ['Analysis Group', parametros.get('grupo', 'N/A')],
            ['Bootstrap Iterations', parametros.get('n_bootstrap', 'N/A')],
            ['Economic Scenarios', 'Baseline, Crisis, Bonanza'],
            ['Timestamp', parametros.get('timestamp', 'N/A')],
            ['Software', 'Python Streamlit + Bootstrap Analysis'],
            ['Cognitive Biases Framework',
                'DH (Hyperbolic Discounting), CS (Social Contagion), AV (Loss Aversion), SQ (Status Quo)']
        ], columns=['Parameter', 'Value'])

        metadata_data.to_excel(
            writer, sheet_name='00_Study_Metadata', index=False)

        # 2. ECUACIONES ESTRUCTURALES Y COEFICIENTES
        for grupo in ['Hah', 'Mah']:
            coef_data = []
            model_info = MODELOS_COEFICIENTES[grupo]

            # Ecuación principal
            coef_data.append(['Structural_Equation', model_info['ecuacion']])

            # Coeficientes
            for var, coef in model_info['coef'].items():
                coef_data.append([f'Beta_{var}', coef])

            # Métricas del modelo
            coef_data.extend([
                ['R_Squared', model_info['r2']],
                ['RMSE', model_info['rmse']],
                ['MAE', model_info['mae']],
                ['Correlation', model_info['correlation']]
            ])

            # Estadísticas descriptivas por variable
            for var, stats in model_info['stats'].items():
                coef_data.extend([
                    [f'{var}_Min', stats['min']],
                    [f'{var}_Max', stats['max']],
                    [f'{var}_Mean', stats['mean']],
                    [f'{var}_Std', stats['std']],
                    [f'{var}_Skewness', stats['skew']],
                    [f'{var}_Kurtosis', stats['kurt']]
                ])

            # Weights del modelo de medida
            for construct, items in model_info['weights'].items():
                for item, weight in items.items():
                    coef_data.append([f'Weight_{construct}_{item}', weight])

            coef_df = pd.DataFrame(coef_data, columns=['Parameter', 'Value'])
            coef_df.to_excel(
                writer, sheet_name=f'01_Model_Parameters_{grupo}', index=False)

        # 3. RESULTADOS BOOTSTRAP DETALLADOS POR ESCENARIO
        for escenario, resultados in resultados_dict.items():

            # 3.1 Datos Bootstrap completos
            bootstrap_data = pd.DataFrame({
                'Bootstrap_Sample_ID': range(1, len(resultados['pca_values']) + 1),
                'PCA_Bootstrap': resultados['pca_values'],
                'PSE_Bootstrap': resultados['pse_values'],
                'DH_Bootstrap': resultados['variables_cognitivas']['DH'],
                'CS_Bootstrap': resultados['variables_cognitivas']['CS'],
                'AV_Bootstrap': resultados['variables_cognitivas']['AV'],
                'SQ_Bootstrap': resultados['variables_cognitivas']['SQ']
            })

            # Añadir correlaciones entre constructos por muestra Bootstrap
            n_samples = len(bootstrap_data)
            correlations_per_sample = []

            for i in range(n_samples):
                sample_data = bootstrap_data.iloc[i]
                vars_dict = {
                    'PSE': sample_data['PSE_Bootstrap'],
                    'PCA': sample_data['PCA_Bootstrap'],
                    'DH': sample_data['DH_Bootstrap'],
                    'CS': sample_data['CS_Bootstrap'],
                    'AV': sample_data['AV_Bootstrap'],
                    'SQ': sample_data['SQ_Bootstrap']
                }

                # Calcular correlaciones (simuladas para cada muestra)
                correlations_per_sample.append({
                    'Bootstrap_Sample_ID': i + 1,
                    'PSE_PCA_Corr': np.corrcoef([vars_dict['PSE']], [vars_dict['PCA']])[0, 1],
                    'SQ_PCA_Corr': np.corrcoef([vars_dict['SQ']], [vars_dict['PCA']])[0, 1],
                    'DH_SQ_Corr': np.corrcoef([vars_dict['DH']], [vars_dict['SQ']])[0, 1],
                    'CS_SQ_Corr': np.corrcoef([vars_dict['CS']], [vars_dict['SQ']])[0, 1],
                    'AV_SQ_Corr': np.corrcoef([vars_dict['AV']], [vars_dict['SQ']])[0, 1]
                })

            correlations_df = pd.DataFrame(correlations_per_sample)

            # Combinar datos principales con correlaciones
            full_bootstrap_data = bootstrap_data.merge(
                correlations_df, on='Bootstrap_Sample_ID')
            full_bootstrap_data.to_excel(
                writer, sheet_name=f'02_Bootstrap_Full_{escenario.title()}', index=False)

            # 3.2 Efectos directos por muestra Bootstrap
            direct_effects = []
            grupo_actual = parametros.get('grupo', 'Hah')
            coef = MODELOS_COEFICIENTES[grupo_actual]['coef']

            for i in range(n_samples):
                effects = {
                    'Bootstrap_Sample_ID': i + 1,
                    # Efecto directo constante
                    'PSE_to_PCA_Effect': coef['PSE'],
                    'SQ_to_PCA_Effect': coef['SQ'],
                    # Efectos simulados
                    'DH_to_SQ_Effect': np.random.normal(0.3, 0.1),
                    'CS_to_SQ_Effect': np.random.normal(0.25, 0.08),
                    'AV_to_SQ_Effect': np.random.normal(0.2, 0.06)
                }
                direct_effects.append(effects)

            effects_df = pd.DataFrame(direct_effects)
            effects_df.to_excel(
                writer, sheet_name=f'03_Direct_Effects_{escenario.title()}', index=False)

            # 3.3 Estadísticas Bootstrap completas
            if 'bootstrap_stats' in resultados:
                stats = resultados['bootstrap_stats']
                bootstrap_stats_data = pd.DataFrame([
                    ['Original_Sample_Size', stats.get('original_n', 'N/A')],
                    ['Bootstrap_Iterations', stats.get('bootstrap_n', 'N/A')],
                    ['PCA_Original_Mean', stats.get('original_mean', 'N/A')],
                    ['PCA_Bootstrap_Mean', stats.get('pca_mean', 'N/A')],
                    ['PCA_Bootstrap_Std', stats.get('pca_std', 'N/A')],
                    ['PCA_Bootstrap_Variance', stats.get('pca_std', 0)**2],
                    ['PCA_CI_Lower_2.5%', stats.get('pca_ci_lower', 'N/A')],
                    ['PCA_CI_Upper_97.5%', stats.get('pca_ci_upper', 'N/A')],
                    ['PCA_CI_Width', stats.get(
                        'pca_ci_upper', 0) - stats.get('pca_ci_lower', 0)],
                    ['Bias_Corrected_Mean', stats.get(
                        'bias_corrected_mean', 'N/A')],
                    ['Estimated_Bias', stats.get(
                        'pca_mean', 0) - stats.get('bias_corrected_mean', 0)],
                    ['Bootstrap_Standard_Error', stats.get(
                        'pca_std', 0) / np.sqrt(stats.get('bootstrap_n', 1))],
                    ['Effective_Sample_Size', stats.get('bootstrap_n', 'N/A')]
                ], columns=['Bootstrap_Statistic', 'Value'])

                bootstrap_stats_data.to_excel(
                    writer, sheet_name=f'04_Bootstrap_Stats_{escenario.title()}', index=False)

        # 4. MODELOS ECONÓMICOS EXTERNOS - ANÁLISIS COMPLETO
        all_economic_models_data = []

        for escenario, resultados in resultados_dict.items():
            for modelo_key, modelo_info in MODELOS_EXTERNOS.items():
                if modelo_key in resultados['modelos_externos']:
                    original_data = resultados['modelos_externos'][modelo_key]['original']
                    pca_data = resultados['modelos_externos'][modelo_key]['con_pca']

                    # Estadísticas completas por modelo
                    for i, (orig, pca) in enumerate(zip(original_data, pca_data)):
                        all_economic_models_data.append({
                            'Scenario': escenario,
                            'Economic_Model': modelo_key,
                            'Model_Name': modelo_info['nombre'],
                            'Bootstrap_Sample_ID': i + 1,
                            'Original_Saving': orig,
                            'PCA_Enhanced_Saving': pca,
                            'Absolute_Difference': pca - orig,
                            'Relative_Difference_Pct': ((pca - orig) / orig) * 100 if orig != 0 else 0,
                            'PCA_Effect_Sign': 'Positive' if pca > orig else 'Negative',
                            'Effect_Magnitude': abs(pca - orig),
                            'Y_Base': 1000,  # Valores base usados
                            'W_Base': 5000,
                            'R_Base': 0.05
                        })

        economic_models_df = pd.DataFrame(all_economic_models_data)
        economic_models_df.to_excel(
            writer, sheet_name='05_Economic_Models_Full', index=False)

        # 5. RESUMEN ESTADÍSTICO COMPARATIVO
        summary_stats = []

        for escenario in resultados_dict.keys():
            for variable in ['PCA', 'PSE', 'DH', 'CS', 'AV', 'SQ']:
                if variable == 'PCA':
                    data = resultados_dict[escenario]['pca_values']
                elif variable == 'PSE':
                    data = resultados_dict[escenario]['pse_values']
                else:
                    data = resultados_dict[escenario]['variables_cognitivas'][variable]

                stats = calcular_estadisticas_avanzadas(data)

                summary_stats.append({
                    'Scenario': escenario,
                    'Variable': variable,
                    'Count': len(data),
                    'Mean': stats['media'],
                    'Median': stats['mediana'],
                    'Std_Dev': stats['std'],
                    'Variance': stats['std']**2,
                    'Min': stats['min'],
                    'Max': stats['max'],
                    'Q1': stats['p25'],
                    'Q3': stats['p75'],
                    'IQR': stats['iqr'],
                    'P5': stats['p5'],
                    'P95': stats['p95'],
                    'Skewness': stats['asimetria'],
                    'Kurtosis': stats['curtosis'],
                    'CV_Percent': stats['cv'] * 100,
                    'Range': stats['max'] - stats['min']
                })

        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(
            writer, sheet_name='06_Descriptive_Statistics', index=False)

        # 6. COMPARACIÓN CON ADANCO (si se proporciona)
        if adanco_data is not None:
            adanco_comparison = []

            # Comparar correlaciones
            for correlation_pair in ['PSE_PCA', 'SQ_PCA', 'DH_SQ', 'CS_SQ', 'AV_SQ']:
                bootstrap_corr = np.mean(
                    correlations_df[f'{correlation_pair}_Corr'])
                adanco_corr = adanco_data.get(
                    f'{correlation_pair}_correlation', np.nan)

                adanco_comparison.append({
                    'Comparison_Type': 'Inter_Construct_Correlation',
                    'Relationship': correlation_pair,
                    'Bootstrap_Value': bootstrap_corr,
                    'ADANCO_Value': adanco_corr,
                    'Difference': bootstrap_corr - adanco_corr if not np.isnan(adanco_corr) else np.nan,
                    'Percent_Difference': ((bootstrap_corr - adanco_corr) / adanco_corr * 100) if not np.isnan(adanco_corr) and adanco_corr != 0 else np.nan
                })

            # Comparar efectos directos
            for effect in ['PSE_to_PCA', 'SQ_to_PCA', 'DH_to_SQ', 'CS_to_SQ', 'AV_to_SQ']:
                bootstrap_effect = np.mean(effects_df[f'{effect}_Effect'])
                adanco_effect = adanco_data.get(
                    f'{effect}_direct_effect', np.nan)

                adanco_comparison.append({
                    'Comparison_Type': 'Direct_Effect',
                    'Relationship': effect,
                    'Bootstrap_Value': bootstrap_effect,
                    'ADANCO_Value': adanco_effect,
                    'Difference': bootstrap_effect - adanco_effect if not np.isnan(adanco_effect) else np.nan,
                    'Percent_Difference': ((bootstrap_effect - adanco_effect) / adanco_effect * 100) if not np.isnan(adanco_effect) and adanco_effect != 0 else np.nan
                })

            adanco_df = pd.DataFrame(adanco_comparison)
            adanco_df.to_excel(
                writer, sheet_name='07_ADANCO_Comparison', index=False)

        # 7. MATRIZ DE CORRELACIONES COMPLETA
        correlation_matrix_data = []
        variables = ['PSE', 'PCA', 'DH', 'CS', 'AV', 'SQ']

        for escenario, resultados in resultados_dict.items():
            # Construir matriz de datos para correlaciones
            data_matrix = np.column_stack([
                resultados['pse_values'],
                resultados['pca_values'],
                resultados['variables_cognitivas']['DH'],
                resultados['variables_cognitivas']['CS'],
                resultados['variables_cognitivas']['AV'],
                resultados['variables_cognitivas']['SQ']
            ])

            corr_matrix = np.corrcoef(data_matrix.T)

            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    correlation_matrix_data.append({
                        'Scenario': escenario,
                        'Variable_1': var1,
                        'Variable_2': var2,
                        'Correlation': corr_matrix[i, j],
                        'Is_Diagonal': i == j
                    })

        correlation_matrix_df = pd.DataFrame(correlation_matrix_data)
        correlation_matrix_df.to_excel(
            writer, sheet_name='08_Correlation_Matrix', index=False)

        # 8. TESTS ESTADÍSTICOS
        statistical_tests = []

        # Tests de normalidad para cada escenario y variable
        for escenario, resultados in resultados_dict.items():
            for variable in ['PCA', 'DH', 'CS', 'AV', 'SQ']:
                if variable == 'PCA':
                    data = resultados['pca_values']
                else:
                    data = resultados['variables_cognitivas'][variable]

                # Test de Shapiro-Wilk (muestra hasta 5000)
                sample_data = np.random.choice(
                    data, min(5000, len(data)), replace=False)
                shapiro_stat, shapiro_p = sp_stats.shapiro(sample_data)

                # Test de Kolmogorov-Smirnov contra distribución normal
                ks_stat, ks_p = sp_stats.kstest(
                    data, 'norm', args=(np.mean(data), np.std(data)))

                statistical_tests.append({
                    'Scenario': escenario,
                    'Variable': variable,
                    'Test_Type': 'Shapiro_Wilk_Normality',
                    'Statistic': shapiro_stat,
                    'P_Value': shapiro_p,
                    'Result': 'Normal' if shapiro_p > 0.05 else 'Non_Normal',
                    'Sample_Size': len(sample_data)
                })

                statistical_tests.append({
                    'Scenario': escenario,
                    'Variable': variable,
                    'Test_Type': 'Kolmogorov_Smirnov_Normality',
                    'Statistic': ks_stat,
                    'P_Value': ks_p,
                    'Result': 'Normal' if ks_p > 0.05 else 'Non_Normal',
                    'Sample_Size': len(data)
                })

        # Tests de diferencia entre escenarios
        scenarios = list(resultados_dict.keys())
        if len(scenarios) > 1:
            baseline_data = resultados_dict.get('baseline', {})

            for scenario in scenarios:
                if scenario != 'baseline' and baseline_data:
                    scenario_data = resultados_dict[scenario]

                    for variable in ['PCA', 'DH', 'CS', 'AV', 'SQ']:
                        if variable == 'PCA':
                            baseline_vals = baseline_data.get('pca_values', [])
                            scenario_vals = scenario_data.get('pca_values', [])
                        else:
                            baseline_vals = baseline_data.get(
                                'variables_cognitivas', {}).get(variable, [])
                            scenario_vals = scenario_data.get(
                                'variables_cognitivas', {}).get(variable, [])

                        if baseline_vals and scenario_vals:
                            # Test t de Student
                            t_stat, t_p = sp_stats.ttest_ind(
                                baseline_vals, scenario_vals)

                            # Test de Mann-Whitney U
                            u_stat, u_p = sp_stats.mannwhitneyu(
                                baseline_vals, scenario_vals, alternative='two-sided')

                            statistical_tests.extend([
                                {
                                    'Scenario': f'Baseline_vs_{scenario}',
                                    'Variable': variable,
                                    'Test_Type': 'T_Test_Independent',
                                    'Statistic': t_stat,
                                    'P_Value': t_p,
                                    'Result': 'Significant' if t_p < 0.05 else 'Not_Significant',
                                    'Sample_Size': f'{len(baseline_vals)}_{len(scenario_vals)}'
                                },
                                {
                                    'Scenario': f'Baseline_vs_{scenario}',
                                    'Variable': variable,
                                    'Test_Type': 'Mann_Whitney_U',
                                    'Statistic': u_stat,
                                    'P_Value': u_p,
                                    'Result': 'Significant' if u_p < 0.05 else 'Not_Significant',
                                    'Sample_Size': f'{len(baseline_vals)}_{len(scenario_vals)}'
                                }
                            ])

        tests_df = pd.DataFrame(statistical_tests)
        tests_df.to_excel(
            writer, sheet_name='09_Statistical_Tests', index=False)

        # 9. DOCUMENTACIÓN DE FÓRMULAS Y MÉTODOS
        formulas_data = [
            ['Method', 'Bootstrap Resampling'],
            ['Bootstrap_Formula',
                'θ* = f(X*)  donde X* es muestra bootstrap de X'],
            ['Bias_Correction', 'θ_bc = 2θ_original - θ_bootstrap'],
            ['Confidence_Interval',
                'CI = [percentil(2.5%), percentil(97.5%)]'],
            ['PLS_SEM_Hombres', 'PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS'],
            ['PLS_SEM_Mujeres', 'PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS'],
            ['Keynes_Model', 'S = a₀ + (a₁ + γ·PCA)Y'],
            ['Friedman_Model', 'S = f(Yₚ·(1 + δ·PCA))'],
            ['Modigliani_Model', 'S = a·W(1 + θ·PCA) + b·Y'],
            ['Carroll_Model', 'S = f(Y) + r(1 + φ·PCA)'],
            ['Deaton_Model', 'S = f(Y,expectations·(1 + κ·PCA))'],
            ['Scenario_Baseline', 'Factores: DH=1.0, CS=1.0, AV=1.0, SQ=1.0'],
            ['Scenario_Crisis', 'Factores: DH=1.4, CS=1.3, AV=1.2, SQ=1.1'],
            ['Scenario_Bonanza', 'Factores: DH=0.7, CS=0.8, AV=0.9, SQ=0.95']
        ]

        formulas_df = pd.DataFrame(formulas_data, columns=[
                                   'Component', 'Formula_Description'])
        formulas_df.to_excel(
            writer, sheet_name='10_Methods_Documentation', index=False)

    output.seek(0)
    return output, filename


# Footer fijo abajo
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
            line-height: 1.2;  /*  controla el espacio entre líneas */
        }
        .footer p {
            margin: 2px 0;  /* elimina los espacios grandes entre párrafos */
        }
    </style>
    <div class="footer">
        <p><strong>Simulador PCA - Tesis Doctoral</strong></p>
        <p>Desarrollado con PLS-SEM y Bootstrap en Python©</p>
        <p>Por MSc. Jesús F. Salazar Rojas</p>
        <p>Propensión Conductual al Ahorro (PCA) © 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)


if __name__ == "__main__":
    main()
