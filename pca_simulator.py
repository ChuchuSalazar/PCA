import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
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
    page_title="PCA Simulator v2.0 - Behavioral Economics Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño doctoral premium sin emojis
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
    .doctoral-section {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal doctoral
html_header = """
<div class="main-header">
    <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700; letter-spacing: 2px;'>
        PCA SIMULATOR v2.0
    </h1>
    <h2 style='margin: 1rem 0; font-size: 1.5rem; opacity: 0.9; font-weight: 400;'>
        La Propensión Conductual al Ahorro:
        Un estudio desde los sesgos cognitivos
        para la toma de decisiones en el ahorro
        de los hogares
    </h2>
    <hr style='margin: 1.5rem auto; width: 70%; border: 2px solid rgba(255,255,255,0.3);'>
    <div style='display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 1.5rem;'>
        <div style='text-align: center; margin: 0.5rem;'>
            <strong style='font-size: 1.1rem;'>MSc. Jesús Fernando Salazar Rojas</strong><br>
            <em style='opacity: 0.8;'>Doctorado en Economía, UCAB — 2025</em>
        </div>
        <div style='text-align: center; margin: 0.5rem;'>
            <strong style='font-size: 1.1rem;'>Methodology</strong><br>
            <em style='opacity: 0.8;'>PLS-SEM + Monte Carlo Simulation</em>
        </div>
        <div style='text-align: center; margin: 0.5rem;'>
            <strong style='font-size: 1.1rem;'>Cognitive Biases Framework</strong><br>
            <em style='opacity: 0.8;'>DH • CS • AV • SQ Analysis</em>
        </div>
    </div>
</div>
"""
st.markdown(html_header, unsafe_allow_html=True)

# Configuración actualizada con nuevos coeficientes y estadísticas descriptivas
MODELOS_COEFICIENTES = {
    'Hah': {
        'ecuacion': 'PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS',
        'coef': {'PSE': 0.3777, 'DH': 0.2226, 'SQ': -0.5947, 'CS': 0.2866},
        'r2': -0.639818,
        'rmse': 1.270350,
        'mae': 1.043280,
        'correlation': -0.129289,
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

# Configuración de escenarios económicos
ESCENARIOS_ECONOMICOS = {
    'baseline': {
        'nombre': 'Baseline Scenario',
        'descripcion': 'Normal economic conditions without external alterations',
        'color': '#34495e',
        'factor_dh': 1.0,
        'factor_cs': 1.0,
        'factor_av': 1.0,
        'factor_sq': 1.0,
        'volatilidad': 1.0
    },
    'crisis': {
        'nombre': 'Economic Crisis',
        'descripcion': 'Uncertainty environment, negative rumors, higher hyperbolic discounting',
        'color': '#e74c3c',
        'factor_dh': 1.4,
        'factor_cs': 1.3,
        'factor_av': 1.2,
        'factor_sq': 1.1,
        'volatilidad': 1.5
    },
    'bonanza': {
        'nombre': 'Economic Bonanza',
        'descripcion': 'Optimistic environment, economic confidence, lower hyperbolic discounting',
        'color': '#27ae60',
        'factor_dh': 0.7,
        'factor_cs': 0.8,
        'factor_av': 0.9,
        'factor_sq': 0.95,
        'volatilidad': 0.8
    }
}

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
        'descripcion': 'Growth and saving model',
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

# Etiquetas actualizadas
EDAD_LABELS = {
    1: 'Under 26', 2: '26-30', 3: '31-35', 4: '36-40', 5: '41-45',
    6: '46-50', 7: '51-55', 8: '56-60', 9: 'Over 60'
}

EDUCACION_LABELS = {
    1: 'Primary', 2: 'High School', 3: 'Technical Degree',
    4: 'University', 5: 'Postgraduate', 6: 'PhD'
}

INGRESO_LABELS = {
    1: '$3-100', 2: '$101-450', 3: '$451-1800',
    4: '$1801-2500', 5: '$2501-10000', 6: 'Over $10000'
}

# Valores por defecto (medias)
DEFAULT_VALUES = {
    'edad': 5,
    'educacion': 4,
    'ingresos': 3
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

        grupo = row['GRUPO']
        weights = MODELOS_COEFICIENTES[grupo]['weights']

        for item in ['AV1', 'AV2', 'AV3', 'AV5']:
            if item in weights['AV']:
                items_row[item] = row['AV'] * \
                    weights['AV'][item] + np.random.normal(0, 0.2)
            else:
                items_row[item] = np.random.normal(0, 1)

        for item in ['DH2', 'DH3', 'DH4', 'DH5']:
            if item in weights['DH']:
                items_row[item] = row['DH'] * \
                    weights['DH'][item] + np.random.normal(0, 0.2)
            else:
                items_row[item] = np.random.normal(0, 1)

        for item in ['SQ1', 'SQ2', 'SQ3']:
            items_row[item] = row['SQ'] * \
                weights['SQ'][item] + np.random.normal(0, 0.2)

        for item in ['CS2', 'CS3', 'CS5']:
            items_row[item] = row['CS'] * \
                weights['CS'][item] + np.random.normal(0, 0.2)

        items_data.append(items_row)

    items_df = pd.DataFrame(items_data)
    return scores_df, items_df


def generar_variable_con_stats(stats, n):
    """Genera variable con estadísticas específicas usando transformación Johnson"""
    data = np.random.normal(0, 1, n)

    if abs(stats['skew']) > 0.1:
        data = data + stats['skew'] * (data**2 - 1) / 6

    if abs(stats['kurt']) > 0.1:
        data = data + stats['kurt'] * (data**3 - 3*data) / 24

    data = (data - np.mean(data)) / np.std(data) * stats['std'] + stats['mean']
    data = np.clip(data, stats['min'], stats['max'])

    return data


def calcular_pse(pca2, pca4, pca5, grupo):
    """Calcula el Perfil Socioeconómico (PSE)"""
    weights = MODELOS_COEFICIENTES[grupo]['weights']['PSE']
    return weights['PCA2'] * pca2 + weights['PCA4'] * pca4 + weights['PCA5'] * pca5


def generar_variables_cognitivas_con_escenario(grupo, escenario, items_weights=None, n_samples=1):
    """Genera variables cognitivas ajustadas por escenario económico"""
    escenario_config = ESCENARIOS_ECONOMICOS[escenario]
    stats = MODELOS_COEFICIENTES[grupo]['stats']

    if items_weights is None:
        base_dh = np.random.normal(
            stats['DH']['mean'], stats['DH']['std'], n_samples)
        base_cs = np.random.normal(
            stats['CS']['mean'], stats['CS']['std'], n_samples)
        base_av = np.random.normal(
            stats['AV']['mean'], stats['AV']['std'], n_samples)
        base_sq = np.random.normal(
            stats['SQ']['mean'], stats['SQ']['std'], n_samples)
    else:
        base_dh = np.array([items_weights.get('DH', 0.0)] * n_samples)
        base_cs = np.array([items_weights.get('CS', 0.0)] * n_samples)
        base_av = np.array([items_weights.get('AV', 0.0)] * n_samples)
        base_sq = np.array([items_weights.get('SQ', 0.0)] * n_samples)

    dh_ajustado = base_dh * escenario_config['factor_dh'] + \
        np.random.normal(0, 0.1 * escenario_config['volatilidad'], n_samples)

    cs_ajustado = base_cs * escenario_config['factor_cs'] + \
        np.random.normal(0, 0.1 * escenario_config['volatilidad'], n_samples)

    av_ajustado = base_av * escenario_config['factor_av'] + \
        np.random.normal(0, 0.1 * escenario_config['volatilidad'], n_samples)

    sq_ajustado = base_sq * escenario_config['factor_sq'] + \
        np.random.normal(0, 0.1 * escenario_config['volatilidad'], n_samples)

    if grupo == 'Hah':
        sq_ajustado = sq_ajustado + 0.3 * av_ajustado

    dh_ajustado = np.clip(dh_ajustado, stats['DH']['min'], stats['DH']['max'])
    cs_ajustado = np.clip(cs_ajustado, stats['CS']['min'], stats['CS']['max'])
    av_ajustado = np.clip(av_ajustado, stats['AV']['min'], stats['AV']['max'])
    sq_ajustado = np.clip(sq_ajustado, stats['SQ']['min'], stats['SQ']['max'])

    return {
        'DH': dh_ajustado[0] if n_samples == 1 else dh_ajustado,
        'CS': cs_ajustado[0] if n_samples == 1 else cs_ajustado,
        'AV': av_ajustado[0] if n_samples == 1 else av_ajustado,
        'SQ': sq_ajustado[0] if n_samples == 1 else sq_ajustado
    }


def calcular_pca_teorica(pse, dh, sq, cs, grupo):
    """Calcula PCA usando la ecuación del modelo"""
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


def ejecutar_simulacion_monte_carlo_avanzada(pca2, pca4, pca5, grupo, escenario, n_simulaciones=5000):
    """Ejecuta simulación Monte Carlo con escenarios económicos"""
    np.random.seed(42)

    pse_base = calcular_pse(pca2, pca4, pca5, grupo)

    resultados = {
        'pca_values': [],
        'variables_cognitivas': {'DH': [], 'CS': [], 'AV': [], 'SQ': []},
        'pse_values': [],
        'escenario': escenario,
        'modelos_externos': {modelo: {'original': [], 'con_pca': []} for modelo in MODELOS_EXTERNOS.keys()},
        'parametros_simulacion': {
            'pca2': pca2, 'pca4': pca4, 'pca5': pca5, 'grupo': grupo,
            'escenario': escenario, 'n_simulaciones': n_simulaciones,
            'pse_base': pse_base, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    for i in range(n_simulaciones):
        vars_cognitivas = generar_variables_cognitivas_con_escenario(
            grupo, escenario)

        for var in ['DH', 'CS', 'AV', 'SQ']:
            resultados['variables_cognitivas'][var].append(
                vars_cognitivas[var])

        pse_actual = pse_base + np.random.normal(0, 0.1)
        resultados['pse_values'].append(pse_actual)

        pca_value = calcular_pca_teorica(
            pse_actual, vars_cognitivas['DH'], vars_cognitivas['SQ'], vars_cognitivas['CS'], grupo
        )

        error_residual = np.random.normal(
            0, MODELOS_COEFICIENTES[grupo]['rmse'] * 0.1)
        pca_value += error_residual

        resultados['pca_values'].append(pca_value)

        volatilidad = ESCENARIOS_ECONOMICOS[escenario]['volatilidad']
        y = abs(np.random.normal(1000, 200 * volatilidad))
        w = abs(np.random.normal(5000, 1000 * volatilidad))
        r = np.random.normal(0.05, 0.02 * volatilidad)

        for modelo_key in MODELOS_EXTERNOS.keys():
            s_orig, s_pca = simular_modelo_externo(
                modelo_key, pca_value, y, w, r)
            resultados['modelos_externos'][modelo_key]['original'].append(
                s_orig)
            resultados['modelos_externos'][modelo_key]['con_pca'].append(s_pca)

    return resultados


def crear_excel_completo(resultados_dict, parametros):
    """Crea archivo Excel completo con todos los resultados y cálculos"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"SIMULATOR2_RESULTADOS_{timestamp}.xlsx"

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formato para headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#34495e',
            'font_color': 'white',
            'border': 1
        })

        # Formato para datos
        data_format = workbook.add_format({'border': 1})
        number_format = workbook.add_format(
            {'num_format': '0.0000', 'border': 1})

        # 1. Hoja de Parámetros y Configuración
        parametros_df = pd.DataFrame([
            ['Grupo Analizado', parametros.get('grupo', 'N/A')],
            ['Edad (PCA2)', parametros.get('pca2', 'N/A')],
            ['Educación (PCA4)', parametros.get('pca4', 'N/A')],
            ['Ingresos (PCA5)', parametros.get('pca5', 'N/A')],
            ['Escenario', parametros.get('escenario', 'N/A')],
            ['N Simulaciones', parametros.get('n_simulaciones', 'N/A')],
            ['PSE Base', parametros.get('pse_base', 'N/A')],
            ['Timestamp', parametros.get('timestamp', 'N/A')]
        ], columns=['Parameter', 'Value'])

        parametros_df.to_excel(
            writer, sheet_name='Config_Parameters', index=False)
        worksheet = writer.sheets['Config_Parameters']
        for col_num, value in enumerate(parametros_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # 2. Hoja de Estadísticas del Modelo
        modelo_info = MODELOS_COEFICIENTES[parametros.get('grupo', 'Hah')]
        modelo_stats_df = pd.DataFrame([
            ['Ecuación', modelo_info['ecuacion']],
            ['R²', modelo_info['r2']],
            ['RMSE', modelo_info['rmse']],
            ['MAE', modelo_info['mae']],
            ['Correlación', modelo_info['correlation']]
        ], columns=['Metric', 'Value'])

        modelo_stats_df.to_excel(
            writer, sheet_name='Model_Statistics', index=False)
        worksheet = writer.sheets['Model_Statistics']
        for col_num, value in enumerate(modelo_stats_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # 3. Hojas de Resultados por Escenario
        for escenario, resultados in resultados_dict.items():
            # Datos principales del escenario
            datos_escenario = pd.DataFrame({
                'Simulation_ID': range(1, len(resultados['pca_values']) + 1),
                'PSE': resultados['pse_values'],
                'PCA': resultados['pca_values'],
                'DH': resultados['variables_cognitivas']['DH'],
                'CS': resultados['variables_cognitivas']['CS'],
                'AV': resultados['variables_cognitivas']['AV'],
                'SQ': resultados['variables_cognitivas']['SQ']
            })

            sheet_name = f'Data_{escenario.title()}'
            datos_escenario.to_excel(
                writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for col_num, value in enumerate(datos_escenario.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Aplicar formato numérico
            for row in range(1, len(datos_escenario) + 1):
                for col in range(1, len(datos_escenario.columns)):
                    worksheet.write(
                        row, col, datos_escenario.iloc[row-1, col], number_format)

        # 4. Hoja de Modelos Económicos
        modelos_data = []
        for escenario, resultados in resultados_dict.items():
            for modelo_key, modelo_data in resultados['modelos_externos'].items():
                for i, (orig, pca) in enumerate(zip(modelo_data['original'], modelo_data['con_pca'])):
                    modelos_data.append({
                        'Escenario': escenario,
                        'Modelo': modelo_key,
                        'Simulation_ID': i + 1,
                        'Original_Saving': orig,
                        'PCA_Enhanced_Saving': pca,
                        'Difference': pca - orig,
                        'Percentage_Change': ((pca - orig) / orig) * 100 if orig != 0 else 0
                    })

        modelos_df = pd.DataFrame(modelos_data)
        modelos_df.to_excel(writer, sheet_name='Economic_Models', index=False)
        worksheet = writer.sheets['Economic_Models']
        for col_num, value in enumerate(modelos_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # 5. Hoja de Estadísticas Descriptivas
        stats_data = []
        for escenario, resultados in resultados_dict.items():
            for var in ['PCA', 'PSE', 'DH', 'CS', 'AV', 'SQ']:
                if var == 'PCA':
                    data = resultados['pca_values']
                elif var == 'PSE':
                    data = resultados['pse_values']
                else:
                    data = resultados['variables_cognitivas'][var]

                stats_data.append({
                    'Escenario': escenario,
                    'Variable': var,
                    'Count': len(data),
                    'Mean': np.mean(data),
                    'Std': np.std(data),
                    'Min': np.min(data),
                    'Q25': np.percentile(data, 25),
                    'Median': np.percentile(data, 50),
                    'Q75': np.percentile(data, 75),
                    'Max': np.max(data),
                    'Skewness': stats.skew(data),
                    'Kurtosis': stats.kurtosis(data),
                    'CV': np.std(data) / np.mean(data) if np.mean(data) != 0 else 0
                })

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(
            writer, sheet_name='Descriptive_Statistics', index=False)
        worksheet = writer.sheets['Descriptive_Statistics']
        for col_num, value in enumerate(stats_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # 6. Hoja de Comparaciones entre Escenarios
        comparaciones_data = []
        baseline_data = resultados_dict.get('baseline', {})

        for escenario in ['crisis', 'bonanza']:
            if escenario in resultados_dict:
                escenario_data = resultados_dict[escenario]

                for var in ['PCA', 'DH', 'CS', 'AV', 'SQ']:
                    if var == 'PCA':
                        baseline_vals = baseline_data.get('pca_values', [])
                        escenario_vals = escenario_data.get('pca_values', [])
                    else:
                        baseline_vals = baseline_data.get(
                            'variables_cognitivas', {}).get(var, [])
                        escenario_vals = escenario_data.get(
                            'variables_cognitivas', {}).get(var, [])

                    if baseline_vals and escenario_vals:
                        baseline_mean = np.mean(baseline_vals)
                        escenario_mean = np.mean(escenario_vals)

                        # Test t
                        t_stat, p_value = stats.ttest_ind(
                            baseline_vals, escenario_vals)

                        comparaciones_data.append({
                            'Variable': var,
                            'Comparison': f'Baseline vs {escenario.title()}',
                            'Baseline_Mean': baseline_mean,
                            'Scenario_Mean': escenario_mean,
                            'Difference': escenario_mean - baseline_mean,
                            'Percent_Change': ((escenario_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0,
                            'T_Statistic': t_stat,
                            'P_Value': p_value,
                            'Significant': 'Yes' if p_value < 0.05 else 'No'
                        })

        comparaciones_df = pd.DataFrame(comparaciones_data)
        comparaciones_df.to_excel(
            writer, sheet_name='Scenario_Comparisons', index=False)
        worksheet = writer.sheets['Scenario_Comparisons']
        for col_num, value in enumerate(comparaciones_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # 7. Hoja de Correlaciones
        correlaciones_data = []
        for escenario, resultados in resultados_dict.items():
            data_corr = pd.DataFrame({
                'PSE': resultados['pse_values'],
                'PCA': resultados['pca_values'],
                'DH': resultados['variables_cognitivas']['DH'],
                'CS': resultados['variables_cognitivas']['CS'],
                'AV': resultados['variables_cognitivas']['AV'],
                'SQ': resultados['variables_cognitivas']['SQ']
            })

            corr_matrix = data_corr.corr()

            for i, var1 in enumerate(corr_matrix.columns):
                for j, var2 in enumerate(corr_matrix.columns):
                    if i <= j:  # Solo mitad superior de la matriz
                        correlaciones_data.append({
                            'Escenario': escenario,
                            'Variable_1': var1,
                            'Variable_2': var2,
                            'Correlation': corr_matrix.loc[var1, var2]
                        })

        correlaciones_df = pd.DataFrame(correlaciones_data)
        correlaciones_df.to_excel(
            writer, sheet_name='Correlations', index=False)
        worksheet = writer.sheets['Correlations']
        for col_num, value in enumerate(correlaciones_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # 8. Hoja de Resumen Ejecutivo
        resumen_data = []
        for escenario, resultados in resultados_dict.items():
            pca_mean = np.mean(resultados['pca_values'])
            pca_std = np.std(resultados['pca_values'])

            # Calcular impacto promedio en modelos económicos
            impactos = []
            for modelo_key in MODELOS_EXTERNOS.keys():
                orig_mean = np.mean(
                    resultados['modelos_externos'][modelo_key]['original'])
                pca_mean_model = np.mean(
                    resultados['modelos_externos'][modelo_key]['con_pca'])
                if orig_mean != 0:
                    impacto = ((pca_mean_model - orig_mean) / orig_mean) * 100
                    impactos.append(impacto)

            avg_impact = np.mean(impactos) if impactos else 0

            resumen_data.append({
                'Scenario': escenario.title(),
                'PCA_Mean': pca_mean,
                'PCA_StdDev': pca_std,
                'PCA_CV': (pca_std / pca_mean) * 100 if pca_mean != 0 else 0,
                'Avg_Economic_Impact_Percent': avg_impact,
                'N_Simulations': len(resultados['pca_values']),
                'DH_Mean': np.mean(resultados['variables_cognitivas']['DH']),
                'CS_Mean': np.mean(resultados['variables_cognitivas']['CS']),
                'AV_Mean': np.mean(resultados['variables_cognitivas']['AV']),
                'SQ_Mean': np.mean(resultados['variables_cognitivas']['SQ'])
            })

        resumen_df = pd.DataFrame(resumen_data)
        resumen_df.to_excel(
            writer, sheet_name='Executive_Summary', index=False)
        worksheet = writer.sheets['Executive_Summary']
        for col_num, value in enumerate(resumen_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

    output.seek(0)
    return output, filename


def crear_grafico_3d_escenarios(resultados_baseline, resultados_crisis, resultados_bonanza, grupo):
    """Crea visualización 3D de los tres escenarios"""
    fig = go.Figure()

    # Escenario Baseline
    fig.add_trace(go.Scatter3d(
        x=resultados_baseline['pse_values'][:500],
        y=resultados_baseline['variables_cognitivas']['CS'][:500],
        z=resultados_baseline['pca_values'][:500],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.6, colorscale='Blues'),
        name='Baseline',
        hovertemplate='<b>Baseline</b><br>PSE: %{x:.3f}<br>CS: %{y:.3f}<br>PCA: %{z:.3f}<extra></extra>'
    ))

    # Escenario Crisis
    fig.add_trace(go.Scatter3d(
        x=resultados_crisis['pse_values'][:500],
        y=resultados_crisis['variables_cognitivas']['CS'][:500],
        z=resultados_crisis['pca_values'][:500],
        mode='markers',
        marker=dict(size=3, color='red', opacity=0.6, colorscale='Reds'),
        name='Crisis',
        hovertemplate='<b>Crisis</b><br>PSE: %{x:.3f}<br>CS: %{y:.3f}<br>PCA: %{z:.3f}<extra></extra>'
    ))

    # Escenario Bonanza
    fig.add_trace(go.Scatter3d(
        x=resultados_bonanza['pse_values'][:500],
        y=resultados_bonanza['variables_cognitivas']['CS'][:500],
        z=resultados_bonanza['pca_values'][:500],
        mode='markers',
        marker=dict(size=3, color='green', opacity=0.6, colorscale='Greens'),
        name='Bonanza',
        hovertemplate='<b>Bonanza</b><br>PSE: %{x:.3f}<br>CS: %{y:.3f}<br>PCA: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'3D Visualization: PSE vs CS vs PCA - {grupo}',
        scene=dict(
            xaxis_title='PSE (Socioeconomic Profile)',
            yaxis_title='CS (Social Contagion)',
            zaxis_title='PCA (Saving Propensity)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=900, height=700, font=dict(size=12)
    )

    return fig


def crear_dashboard_comparativo(resultados_dict):
    """Crea dashboard comparativo de los tres escenarios"""
    escenarios = ['baseline', 'crisis', 'bonanza']
    colores = ['#34495e', '#e74c3c', '#27ae60']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'PCA Distribution by Scenario',
            'Average Cognitive Variables',
            'Impact on Economic Models',
            'PSE-PCA Correlations'
        ],
        specs=[[{"secondary_y": False}, {"type": "bar"}],
               [{"type": "bar"}, {"secondary_y": False}]]
    )

    # 1. Distribuciones PCA
    for i, escenario in enumerate(escenarios):
        resultados = resultados_dict[escenario]
        fig.add_trace(
            go.Violin(
                y=resultados['pca_values'],
                name=ESCENARIOS_ECONOMICOS[escenario]['nombre'],
                box_visible=True,
                meanline_visible=True,
                fillcolor=colores[i],
                opacity=0.6,
                line_color=colores[i]
            ),
            row=1, col=1
        )

    # 2. Variables Cognitivas Promedio
    vars_cognitivas = ['DH', 'CS', 'AV', 'SQ']
    for var in vars_cognitivas:
        valores = []
        for escenario in escenarios:
            valores.append(
                np.mean(resultados_dict[escenario]['variables_cognitivas'][var]))

        fig.add_trace(
            go.Bar(
                x=[ESCENARIOS_ECONOMICOS[esc]['nombre'] for esc in escenarios],
                y=valores,
                name=f'{var}',
                marker_color=px.colors.qualitative.Set3[vars_cognitivas.index(
                    var)]
            ),
            row=1, col=2
        )

    # 3. Impacto en Modelos Económicos
    modelo_keynes = 'keynes'
    impactos = []
    for escenario in escenarios:
        resultados = resultados_dict[escenario]
        original_mean = np.mean(
            resultados['modelos_externos'][modelo_keynes]['original'])
        pca_mean = np.mean(
            resultados['modelos_externos'][modelo_keynes]['con_pca'])
        impacto_pct = ((pca_mean - original_mean) / original_mean) * 100
        impactos.append(impacto_pct)

    fig.add_trace(
        go.Bar(
            x=[ESCENARIOS_ECONOMICOS[esc]['nombre'] for esc in escenarios],
            y=impactos,
            name='Impact % Keynes',
            marker_color=colores,
            text=[f'{imp:+.1f}%' for imp in impactos],
            textposition='auto'
        ),
        row=2, col=1
    )

    # 4. Correlaciones PSE-PCA
    for i, escenario in enumerate(escenarios):
        resultados = resultados_dict[escenario]
        fig.add_trace(
            go.Scatter(
                x=resultados['pse_values'][:200],
                y=resultados['pca_values'][:200],
                mode='markers',
                name=f'Correlation {ESCENARIOS_ECONOMICOS[escenario]["nombre"]}',
                marker=dict(color=colores[i], size=4, opacity=0.6)
            ),
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Comparative Dashboard of Economic Scenarios"
    )

    return fig


def crear_radar_chart_cognitivo(resultados_dict, grupo):
    """Crea gráfico radar para variables cognitivas"""
    categories = ['DH<br>(Hyperbolic<br>Discounting)',
                  'CS<br>(Social<br>Contagion)',
                  'AV<br>(Loss<br>Aversion)',
                  'SQ<br>(Status<br>Quo)']

    fig = go.Figure()

    for escenario, color in [('baseline', 'blue'), ('crisis', 'red'), ('bonanza', 'green')]:
        resultados = resultados_dict[escenario]
        valores = []

        for var in ['DH', 'CS', 'AV', 'SQ']:
            media = np.mean(resultados['variables_cognitivas'][var])
            stats = MODELOS_COEFICIENTES[grupo]['stats'][var]
            valor_norm = (media - stats['min']) / (stats['max'] - stats['min'])
            valores.append(valor_norm)

        fig.add_trace(go.Scatterpolar(
            r=valores + [valores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=ESCENARIOS_ECONOMICOS[escenario]['nombre'],
            line_color=color,
            fillcolor=color,
            opacity=0.3
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['Min', '25%', '50%', '75%', 'Max']
            )
        ),
        showlegend=True,
        title=f"Cognitive Biases Radar Chart - {grupo}",
        height=500
    )

    return fig


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
        'asimetria': stats.skew(datos),
        'curtosis': stats.kurtosis(datos),
        'cv': np.std(datos) / np.mean(datos) if np.mean(datos) != 0 else 0,
        'iqr': np.percentile(datos, 75) - np.percentile(datos, 25)
    }


def display_model_images(grupo):
    """Muestra las imágenes del modelo estructural según el grupo"""
    try:
        if grupo == 'Hah':
            image_path = r"hombres.JPG"
            title = "Structural Model - Male Savers (Hah)"
        else:
            image_path = r"C:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\01 TESIS DEFINITIVA\MODELO\modelos jpg\mujeres.JPG"
            title = "Structural Model - Female Savers (Mah)"

        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.markdown(f"""
            <div class="model-images">
                <h4 style="color: #2c3e50; text-align: center; margin-bottom: 1rem;">{title}</h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, caption=title, use_column_width=True)
        else:
            st.error(f"Model image not found at: {image_path}")
            st.info(
                "Please ensure the image files are located in the specified directory.")
    except Exception as e:
        st.error(f"Error loading model image: {str(e)}")


def main():
    # Cargar datos
    with st.spinner("Loading database..."):
        scores_df, items_df = cargar_datos()

    # Sidebar mejorado
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(45deg, #2c3e50 0%, #34495e 100%); 
                    border-radius: 10px; color: white; margin-bottom: 1rem;'>
            <h3 style='margin: 0;'>Control Panel</h3>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Advanced Configuration</p>
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
            <h4>Model Metrics - {grupo}</h4>
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

        # Variables socioeconómicas
        st.markdown("**Socioeconomic Variables**")

        pca2 = st.slider("**Age**", min_value=1, max_value=9,
                         value=DEFAULT_VALUES['edad'])
        st.caption(f"{EDAD_LABELS[pca2]}")

        pca4 = st.slider("**Education Level**", min_value=1,
                         max_value=6, value=DEFAULT_VALUES['educacion'])
        st.caption(f"{EDUCACION_LABELS[pca4]}")

        pca5 = st.slider("**Income Level**", min_value=1,
                         max_value=6, value=DEFAULT_VALUES['ingresos'])
        st.caption(f"{INGRESO_LABELS[pca5]}")

        st.markdown("---")

        # Parámetros de simulación
        st.markdown("**Simulation Parameters**")
        n_simulaciones = st.number_input(
            "Number of Monte Carlo simulations",
            min_value=1000, max_value=15000, value=5000, step=1000
        )

        analisis_comparativo = st.checkbox(
            "**Multi-scenario Comparative Analysis**", value=True
        )

    # Mostrar imágenes del modelo si está activado
    if st.session_state.show_model_images:
        st.markdown("---")
        display_model_images(grupo)
        st.markdown("---")

    # Información del modelo actual
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("### Active PLS-SEM Model")
        st.code(MODELOS_COEFICIENTES[grupo]['ecuacion'], language='text')
        pse_calculado = calcular_pse(pca2, pca4, pca5, grupo)
        st.info(f"**Calculated PSE:** {pse_calculado:.4f}")

    with col2:
        st.markdown("### Current Profile")
        st.metric("Group", "Male" if grupo == 'Hah' else "Female")
        st.metric("Age", EDAD_LABELS[pca2])
        st.metric("Education", EDUCACION_LABELS[pca4])

    with col3:
        st.markdown("### Context")
        st.metric("Income", INGRESO_LABELS[pca5])
        st.metric("Scenario", escenario_info['nombre'])
        st.metric("Simulations", f"{n_simulaciones:,}")

    # Botón de simulación
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        current_params = {
            'grupo': grupo, 'pca2': pca2, 'pca4': pca4, 'pca5': pca5,
            'escenario': escenario, 'n_simulaciones': n_simulaciones,
            'analisis_comparativo': analisis_comparativo
        }

        if st.button(
            f"**EXECUTE ADVANCED SIMULATION**" +
            (f" - {escenario_info['nombre']}" if not analisis_comparativo else " - MULTI-SCENARIO"),
            type="primary", use_container_width=True
        ):

            # Ejecutar simulación
            if analisis_comparativo:
                st.markdown("### Executing Multi-scenario Analysis...")

                progress_bar = st.progress(0)
                resultados_dict = {}

                for i, esc in enumerate(['baseline', 'crisis', 'bonanza']):
                    with st.spinner(f"Simulating scenario {ESCENARIOS_ECONOMICOS[esc]['nombre']}..."):
                        resultados_dict[esc] = ejecutar_simulacion_monte_carlo_avanzada(
                            pca2, pca4, pca5, grupo, esc, n_simulaciones
                        )
                    progress_bar.progress((i + 1) / 3)

                st.session_state.resultados_dict = resultados_dict
                st.session_state.simulation_completed = True
                st.session_state.current_parameters = current_params

                st.success(
                    f"**Multi-scenario Analysis Completed:** {n_simulaciones:,} × 3 simulations")

            else:
                with st.spinner(f"Simulating {escenario_info['nombre']} scenario..."):
                    resultado = ejecutar_simulacion_monte_carlo_avanzada(
                        pca2, pca4, pca5, grupo, escenario, n_simulaciones
                    )

                st.session_state.resultados_dict = {escenario: resultado}
                st.session_state.simulation_completed = True
                st.session_state.current_parameters = current_params

                st.success(
                    f"**Single Scenario Analysis Completed:** {n_simulaciones:,} simulations")

    # Mostrar resultados si la simulación está completada
    if st.session_state.simulation_completed and st.session_state.resultados_dict:

        # Sección de descarga de resultados
        st.markdown("---")
        st.markdown("""
        <div class="download-section">
            <h3 style='margin: 0 0 1rem 0;'>Complete Results Download</h3>
            <p style='margin: 0; opacity: 0.9;'>Download comprehensive Excel file with all calculations and data for audit trail</p>
        </div>
        """, unsafe_allow_html=True)

        col_download1, col_download2, col_download3 = st.columns([1, 2, 1])

        with col_download2:
            if st.button("Generate & Download Complete Results Excel", type="secondary", use_container_width=True):
                with st.spinner("Generating comprehensive Excel report..."):
                    excel_buffer, filename = crear_excel_completo(
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
                        "Excel file generated successfully! Contains all simulation data, statistics, and audit trail.")

        if len(st.session_state.resultados_dict) > 1:  # Multi-scenario analysis
            # Crear tabs mejoradas
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Comparative Dashboard",
                "3D Visualization",
                "Cognitive Analysis",
                "Economic Impact",
                "Advanced Statistics",
                "Validation & Diagnostics"
            ])

            with tab1:
                st.markdown("### Comparative Dashboard of Economic Scenarios")

                # Dashboard principal
                fig_dashboard = crear_dashboard_comparativo(
                    st.session_state.resultados_dict)
                st.plotly_chart(fig_dashboard, use_container_width=True)

                # Métricas comparativas
                st.markdown("### Comparative PCA Metrics")
                col1, col2, col3 = st.columns(3)

                for i, (esc, col) in enumerate(zip(['baseline', 'crisis', 'bonanza'], [col1, col2, col3])):
                    with col:
                        resultados = st.session_state.resultados_dict[esc]
                        media_pca = np.mean(resultados['pca_values'])
                        std_pca = np.std(resultados['pca_values'])

                        st.markdown(f"""
                        <div style="background: {ESCENARIOS_ECONOMICOS[esc]['color']}20; 
                                    padding: 1rem; border-radius: 8px; text-align: center;">
                            <h4 style="color: {ESCENARIOS_ECONOMICOS[esc]['color']};">
                                {ESCENARIOS_ECONOMICOS[esc]['nombre']}
                            </h4>
                            <p><strong>Mean PCA:</strong> {media_pca:.4f}</p>
                            <p><strong>Std PCA:</strong> {std_pca:.4f}</p>
                            <p><strong>CV:</strong> {(std_pca/media_pca)*100 if media_pca != 0 else 0:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

            with tab2:
                st.markdown("### Multi-scenario 3D Visualization")

                # Gráfico 3D
                fig_3d = crear_grafico_3d_escenarios(
                    st.session_state.resultados_dict['baseline'],
                    st.session_state.resultados_dict['crisis'],
                    st.session_state.resultados_dict['bonanza'],
                    grupo
                )
                st.plotly_chart(fig_3d, use_container_width=True)

                # Interpretación
                st.markdown("""
                **3D Visualization Interpretation:**
                - **X-axis (PSE):** Socioeconomic Profile, constant by design
                - **Y-axis (CS):** Social Contagion, varies significantly between scenarios
                - **Z-axis (PCA):** Behavioral Saving Propensity, dependent variable
                - **Clustering:** Observe how points group according to scenario
                """)

            with tab3:
                st.markdown("### Detailed Cognitive Biases Analysis")

                col1, col2 = st.columns([1, 1])

                with col1:
                    # Radar chart
                    fig_radar = crear_radar_chart_cognitivo(
                        st.session_state.resultados_dict, grupo)
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col2:
                    # Tabla de cambios relativos
                    st.markdown("#### Relative Changes vs Baseline")

                    datos_cambios = []
                    baseline_vals = st.session_state.resultados_dict['baseline']['variables_cognitivas']

                    for var in ['DH', 'CS', 'AV', 'SQ']:
                        baseline_mean = np.mean(baseline_vals[var])
                        crisis_mean = np.mean(
                            st.session_state.resultados_dict['crisis']['variables_cognitivas'][var])
                        bonanza_mean = np.mean(
                            st.session_state.resultados_dict['bonanza']['variables_cognitivas'][var])

                        cambio_crisis = (
                            (crisis_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
                        cambio_bonanza = (
                            (bonanza_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0

                        datos_cambios.append({
                            'Variable': var,
                            'Baseline': baseline_mean,
                            'Δ% Crisis': cambio_crisis,
                            'Δ% Bonanza': cambio_bonanza
                        })

                    df_cambios = pd.DataFrame(datos_cambios)
                    st.dataframe(df_cambios.round(3), use_container_width=True)

                    # Interpretaciones psicológicas
                    st.markdown("""
                    **Psychological Interpretation:**
                    
                    - **DH (Hyperbolic Discounting):** Crisis = higher impatience
                    - **CS (Social Contagion):** Crisis = greater social influence
                    - **AV (Loss Aversion):** Crisis = higher risk aversion
                    - **SQ (Status Quo):** Crisis = greater resistance to change
                    """)

            with tab4:
                st.markdown("### Economic Models Impact Analysis")

                # Keep the selected model in session state to prevent reset
                if 'selected_model' not in st.session_state:
                    st.session_state.selected_model = 'keynes'

                # Análisis detallado por modelo económico
                modelo_analizar = st.selectbox(
                    "Select economic model for detailed analysis:",
                    options=list(MODELOS_EXTERNOS.keys()),
                    format_func=lambda x: MODELOS_EXTERNOS[x]['nombre'],
                    key="model_selector",
                    index=list(MODELOS_EXTERNOS.keys()).index(
                        st.session_state.selected_model)
                )

                # Update session state
                st.session_state.selected_model = modelo_analizar

                # Calcular impactos para todos los escenarios
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Gráfico de violín comparativo
                    fig_violin = go.Figure()

                    colores_violin = ['blue', 'red', 'green']
                    for i, esc in enumerate(['baseline', 'crisis', 'bonanza']):
                        resultados = st.session_state.resultados_dict[esc]
                        datos_modelo = resultados['modelos_externos'][modelo_analizar]['con_pca']

                        fig_violin.add_trace(go.Violin(
                            y=datos_modelo,
                            name=ESCENARIOS_ECONOMICOS[esc]['nombre'],
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor=colores_violin[i],
                            line_color=colores_violin[i],
                            opacity=0.6
                        ))

                    fig_violin.update_layout(
                        title=f'Saving Distribution - {MODELOS_EXTERNOS[modelo_analizar]["nombre"]}',
                        yaxis_title='Projected Saving',
                        height=400
                    )

                    st.plotly_chart(fig_violin, use_container_width=True)

                with col2:
                    # Tabla de estadísticas del modelo
                    st.markdown("#### Model Statistics")

                    stats_modelo = []
                    for esc in ['baseline', 'crisis', 'bonanza']:
                        resultados = st.session_state.resultados_dict[esc]
                        datos = resultados['modelos_externos'][modelo_analizar]['con_pca']
                        stats = calcular_estadisticas_avanzadas(datos)

                        stats_modelo.append({
                            'Scenario': ESCENARIOS_ECONOMICOS[esc]['nombre'],
                            'Mean': stats['media'],
                            'Median': stats['mediana'],
                            'Std': stats['std'],
                            'CV%': stats['cv'] * 100
                        })

                    df_stats_modelo = pd.DataFrame(stats_modelo)
                    st.dataframe(df_stats_modelo.round(2),
                                 use_container_width=True)

                # Matriz de impactos todos los modelos
                st.markdown("#### Multi-model Impact Matrix")

                matriz_impactos = []
                for modelo_key, modelo_info in MODELOS_EXTERNOS.items():
                    fila = {'Model': modelo_info['nombre']}

                    for esc in ['baseline', 'crisis', 'bonanza']:
                        resultados = st.session_state.resultados_dict[esc]
                        original_mean = np.mean(
                            resultados['modelos_externos'][modelo_key]['original'])
                        pca_mean = np.mean(
                            resultados['modelos_externos'][modelo_key]['con_pca'])
                        impacto_pct = (
                            (pca_mean - original_mean) / original_mean) * 100
                        fila[f'{esc.title()}'] = impacto_pct

                    matriz_impactos.append(fila)

                df_impactos = pd.DataFrame(matriz_impactos)

                # Colorear la tabla según impactos
                def color_impact(val):
                    if isinstance(val, (int, float)):
                        if val > 5:
                            return 'background-color: #27ae60; color: white'
                        elif val < -5:
                            return 'background-color: #e74c3c; color: white'
                        else:
                            return 'background-color: #f39c12; color: white'
                    return ''

                styled_df = df_impactos.style.applymap(
                    color_impact, subset=['Baseline', 'Crisis', 'Bonanza'])
                st.dataframe(styled_df, use_container_width=True)

            with tab5:
                st.markdown("### Advanced Descriptive Statistics")

                # Selector de variable para análisis
                variable_analisis = st.selectbox(
                    "Variable for detailed statistical analysis:",
                    options=['PCA'] + ['DH', 'CS', 'AV', 'SQ'],
                    index=0
                )

                col1, col2 = st.columns([1, 1])

                with col1:
                    # Histograma comparativo
                    fig_hist_comp = go.Figure()

                    for esc, color in [('baseline', 'blue'), ('crisis', 'red'), ('bonanza', 'green')]:
                        if variable_analisis == 'PCA':
                            datos = st.session_state.resultados_dict[esc]['pca_values']
                        else:
                            datos = st.session_state.resultados_dict[esc][
                                'variables_cognitivas'][variable_analisis]

                        fig_hist_comp.add_trace(go.Histogram(
                            x=datos,
                            name=ESCENARIOS_ECONOMICOS[esc]['nombre'],
                            opacity=0.6,
                            nbinsx=30,
                            marker_color=color
                        ))

                    fig_hist_comp.update_layout(
                        title=f'Comparative Distribution - {variable_analisis}',
                        xaxis_title=variable_analisis,
                        yaxis_title='Frequency',
                        barmode='overlay',
                        height=400
                    )

                    st.plotly_chart(fig_hist_comp, use_container_width=True)

                with col2:
                    # Tests estadísticos
                    st.markdown("#### Statistical Tests")

                    # Preparar datos para tests
                    datos_tests = {}
                    for esc in ['baseline', 'crisis', 'bonanza']:
                        if variable_analisis == 'PCA':
                            datos_tests[esc] = st.session_state.resultados_dict[esc]['pca_values']
                        else:
                            datos_tests[esc] = st.session_state.resultados_dict[esc]['variables_cognitivas'][variable_analisis]

                    # Tests de normalidad y comparaciones
                    test_results = []

                    # Test de normalidad Shapiro-Wilk (muestra pequeña)
                    for esc in ['baseline', 'crisis', 'bonanza']:
                        data_sample = np.random.choice(
                            datos_tests[esc], min(5000, len(datos_tests[esc])))
                        if len(data_sample) <= 5000:
                            # Máximo 5000 para Shapiro
                            stat, p_val = stats.shapiro(data_sample[:5000])
                            test_results.append({
                                'Test': f'Normality {esc}',
                                'Statistic': stat,
                                'P-value': p_val,
                                'Result': 'Normal' if p_val > 0.05 else 'Non-normal'
                            })

                    # Tests de comparación entre escenarios
                    baseline_data = datos_tests['baseline']
                    for esc in ['crisis', 'bonanza']:
                        scenario_data = datos_tests[esc]

                        # Test t de Student
                        t_stat, t_p = stats.ttest_ind(
                            baseline_data, scenario_data)
                        test_results.append({
                            'Test': f'T-test: Baseline vs {esc}',
                            'Statistic': t_stat,
                            'P-value': t_p,
                            'Result': 'Significant' if t_p < 0.05 else 'Not significant'
                        })

                        # Test de Mann-Whitney U (no paramétrico)
                        u_stat, u_p = stats.mannwhitneyu(
                            baseline_data, scenario_data, alternative='two-sided')
                        test_results.append({
                            'Test': f'Mann-Whitney: Baseline vs {esc}',
                            'Statistic': u_stat,
                            'P-value': u_p,
                            'Result': 'Significant' if u_p < 0.05 else 'Not significant'
                        })

                    df_tests = pd.DataFrame(test_results)
                    st.dataframe(df_tests.round(4), use_container_width=True)

            with tab6:
                st.markdown("### Validation & Diagnostics")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Model Residuals Analysis")

                    # Análisis de residuales para el grupo actual
                    residuales_data = []
                    for esc, resultados in st.session_state.resultados_dict.items():
                        for i, (pse, pca_real) in enumerate(zip(resultados['pse_values'], resultados['pca_values'])):
                            # Calcular PCA teórica
                            vars_cog = {var: resultados['variables_cognitivas'][var][i] for var in [
                                'DH', 'CS', 'AV', 'SQ']}
                            pca_teorica = calcular_pca_teorica(
                                pse, vars_cog['DH'], vars_cog['SQ'], vars_cog['CS'], grupo)
                            residual = pca_real - pca_teorica
                            residuales_data.append(
                                {'Scenario': esc, 'Theoretical': pca_teorica, 'Actual': pca_real, 'Residual': residual})

                    df_residuales = pd.DataFrame(residuales_data)

                    # Gráfico Q-Q de residuales
                    fig_qq = go.Figure()

                    for esc, color in [('baseline', 'blue'), ('crisis', 'red'), ('bonanza', 'green')]:
                        residuales_esc = df_residuales[df_residuales['Scenario']
                                                       == esc]['Residual']
                        qq_data = stats.probplot(residuales_esc, dist="norm")

                        fig_qq.add_trace(go.Scatter(
                            x=qq_data[0][0],
                            y=qq_data[0][1],
                            mode='markers',
                            name=f'{esc} residuals',
                            marker=dict(color=color, size=4, opacity=0.6)
                        ))

                    # Línea de referencia
                    x_min, x_max = min(qq_data[0][0]), max(qq_data[0][0])
                    fig_qq.add_trace(go.Scatter(
                        x=[x_min, x_max],
                        y=[x_min * qq_data[1][0] + qq_data[1][1],
                            x_max * qq_data[1][0] + qq_data[1][1]],
                        mode='lines',
                        name='Reference line',
                        line=dict(color='red', dash='dash')
                    ))

                    fig_qq.update_layout(
                        title='Q-Q Plot: Model Residuals',
                        xaxis_title='Theoretical Quantiles',
                        yaxis_title='Sample Quantiles',
                        height=400
                    )

                    st.plotly_chart(fig_qq, use_container_width=True)

                with col2:
                    st.markdown("#### Convergence Diagnostics")

                    # Estadísticas de convergencia
                    convergence_stats = []
                    for esc, resultados in st.session_state.resultados_dict.items():
                        pca_values = resultados['pca_values']

                        # Media móvil para evaluar convergencia
                        window_size = min(500, len(pca_values) // 10)
                        moving_avg = pd.Series(pca_values).rolling(
                            window=window_size).mean()

                        # Estabilidad (varianza de la media móvil en la segunda mitad)
                        second_half = moving_avg[len(moving_avg)//2:]
                        stability = np.std(second_half.dropna())

                        convergence_stats.append({
                            'Scenario': esc.title(),
                            # Últimas 1000 simulaciones
                            'Final Mean': np.mean(pca_values[-1000:]),
                            'Moving Avg Stability': stability,
                            'Effective Sample Size': len(pca_values),
                            'Monte Carlo Error': np.std(pca_values) / np.sqrt(len(pca_values))
                        })

                    df_convergence = pd.DataFrame(convergence_stats)
                    st.dataframe(df_convergence.round(
                        4), use_container_width=True)

                    # Autocorrelación
                    st.markdown("#### Autocorrelation Analysis")

                    fig_acf = go.Figure()

                    for esc, color in [('baseline', 'blue'), ('crisis', 'red'), ('bonanza', 'green')]:
                        pca_values = st.session_state.resultados_dict[esc]['pca_values']

                        # Calcular autocorrelación para los primeros 50 lags
                        lags = range(0, min(51, len(pca_values)//4))
                        autocorr = [np.corrcoef(pca_values[:-lag] if lag > 0 else pca_values,
                                                pca_values[lag:] if lag > 0 else pca_values)[0, 1]
                                    for lag in lags]

                        fig_acf.add_trace(go.Scatter(
                            x=list(lags),
                            y=autocorr,
                            mode='lines+markers',
                            name=f'{esc} ACF',
                            line=dict(color=color)
                        ))

                    # Línea de significancia
                    n_samples = len(
                        st.session_state.resultados_dict['baseline']['pca_values'])
                    significance_level = 1.96 / np.sqrt(n_samples)

                    fig_acf.add_hline(y=significance_level, line_dash="dash", line_color="red",
                                      annotation_text="95% Confidence")
                    fig_acf.add_hline(y=-significance_level,
                                      line_dash="dash", line_color="red")

                    fig_acf.update_layout(
                        title='Autocorrelation Function',
                        xaxis_title='Lag',
                        yaxis_title='Autocorrelation',
                        height=300
                    )

                    st.plotly_chart(fig_acf, use_container_width=True)

        else:  # Single scenario analysis
            st.markdown("### Single Scenario Analysis Results")

            escenario_key = list(st.session_state.resultados_dict.keys())[0]
            resultados = st.session_state.resultados_dict[escenario_key]

            # Mostrar resultados básicos del escenario único
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Mean PCA", f"{np.mean(resultados['pca_values']):.4f}")
                st.metric("Std PCA", f"{np.std(resultados['pca_values']):.4f}")

            with col2:
                st.metric(
                    "Mean DH", f"{np.mean(resultados['variables_cognitivas']['DH']):.4f}")
                st.metric(
                    "Mean CS", f"{np.mean(resultados['variables_cognitivas']['CS']):.4f}")

            with col3:
                st.metric(
                    "Mean AV", f"{np.mean(resultados['variables_cognitivas']['AV']):.4f}")
                st.metric(
                    "Mean SQ", f"{np.mean(resultados['variables_cognitivas']['SQ']):.4f}")

            # Gráficos básicos para escenario único
            fig_single = make_subplots(
                rows=2, cols=2,
                subplot_titles=['PCA Distribution', 'Cognitive Variables',
                                'Economic Models Impact', 'PSE vs PCA'],
                specs=[[{"secondary_y": False}, {"type": "bar"}],
                       [{"type": "bar"}, {"secondary_y": False}]]
            )

            # Distribución PCA
            fig_single.add_trace(
                go.Histogram(x=resultados['pca_values'],
                             nbinsx=30, name='PCA Distribution'),
                row=1, col=1
            )

            # Variables cognitivas
            vars_cog = ['DH', 'CS', 'AV', 'SQ']
            valores_cog = [
                np.mean(resultados['variables_cognitivas'][var]) for var in vars_cog]

            fig_single.add_trace(
                go.Bar(x=vars_cog, y=valores_cog, name='Cognitive Variables'),
                row=1, col=2
            )

            # Impacto en modelos económicos
            modelos = list(MODELOS_EXTERNOS.keys())
            impactos = []
            for modelo in modelos:
                orig_mean = np.mean(
                    resultados['modelos_externos'][modelo]['original'])
                pca_mean = np.mean(
                    resultados['modelos_externos'][modelo]['con_pca'])
                impacto_pct = ((pca_mean - orig_mean) / orig_mean) * 100
                impactos.append(impacto_pct)

            fig_single.add_trace(
                go.Bar(x=modelos, y=impactos, name='Economic Impact %'),
                row=2, col=1
            )

            # PSE vs PCA
            fig_single.add_trace(
                go.Scatter(
                    x=resultados['pse_values'][:500],
                    y=resultados['pca_values'][:500],
                    mode='markers',
                    name='PSE vs PCA'
                ),
                row=2, col=2
            )

            fig_single.update_layout(
                height=800, showlegend=True, title_text=f"Analysis Results - {ESCENARIOS_ECONOMICOS[escenario_key]['nombre']}")
            st.plotly_chart(fig_single, use_container_width=True)


if __name__ == "__main__":
    main()
