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

warnings.filterwarnings('ignore')

# Configuración de la página con diseño premium
st.set_page_config(
    page_title="PCA Simulator v2.0 - Crisis & Bonanza",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para un diseño doctoral premium
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .scenario-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .crisis-theme {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
    }
    .bonanza-theme {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    .baseline-theme {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header principal renovado
html_header = """
<div class="main-header">
    <h1 style='margin: 0; font-size: 2.5rem; font-weight: 900;'>
        🧠 PCA Simulator v2.0 - Crisis & Bonanza
    </h1>
    <h3 style='margin: 1rem 0; font-size: 1.3rem; opacity: 0.9;'>
        Behavioral Propensity to Save under Economic Scenarios
    </h3>
    <hr style='margin: 1.5rem auto; width: 70%; border: 2px solid rgba(255,255,255,0.3);'>
    <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
        <div>
            <strong>👨‍🎓 MSc. Jesús F. Salazar Rojas</strong><br>
            <em>Doctorado en Economía, UCAB – 2025</em>
        </div>
        <div>
            <strong>📊 PLS-SEM + Monte Carlo</strong><br>
            <em>Crisis, Baseline & Bonanza Scenarios</em>
        </div>
        <div>
            <strong>🧭 Sesgos Cognitivos</strong><br>
            <em>DH • CS • AV • SQ</em>
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
        'nombre': '⚖️ Línea Base',
        'descripcion': 'Condiciones económicas normales sin alteraciones externas',
        'color': '#6c5ce7',
        'factor_dh': 1.0,
        'factor_cs': 1.0,
        'factor_av': 1.0,
        'factor_sq': 1.0,
        'volatilidad': 1.0
    },
    'crisis': {
        'nombre': '🔴 Crisis (Rumores Negativos)',
        'descripcion': 'Ambiente de incertidumbre, rumores negativos, mayor descuento hiperbólico',
        'color': '#e74c3c',
        # Mayor descuento hiperbólico (valorar más el presente)
        'factor_dh': 1.4,
        'factor_cs': 1.3,  # Mayor contagio social (influencia de rumores)
        'factor_av': 1.2,  # Mayor aversión a pérdidas
        'factor_sq': 1.1,  # Leve tendencia a mantener status quo
        'volatilidad': 1.5
    },
    'bonanza': {
        'nombre': '🟢 Bonanza (Optimismo)',
        'descripcion': 'Ambiente optimista, confianza económica, menor descuento hiperbólico',
        'color': '#27ae60',
        'factor_dh': 0.7,  # Menor descuento hiperbólico (mayor paciencia)
        'factor_cs': 0.8,  # Menor influencia del contagio social
        'factor_av': 0.9,  # Menor aversión a pérdidas
        'factor_sq': 0.95,  # Menor tendencia al status quo
        'volatilidad': 0.8
    }
}

MODELOS_EXTERNOS = {
    'keynes': {
        'nombre': 'Keynes (1936)',
        'original': 'S = a₀ + a₁Y',
        'con_pca': 'S = a₀ + (a₁ + γ·PCA)Y',
        'descripcion': 'Función ahorro keynesiana',
        'parametros': {'a0': -50, 'a1': 0.2, 'gamma': 0.15}
    },
    'friedman': {
        'nombre': 'Friedman (1957)',
        'original': 'S = f(Yₚ)',
        'con_pca': 'S = f(Yₚ·(1 + δ·PCA))',
        'descripcion': 'Hipótesis del ingreso permanente',
        'parametros': {'base_rate': 0.15, 'delta': 0.1, 'yp_factor': 0.8}
    },
    'modigliani': {
        'nombre': 'Modigliani-Brumberg (1954)',
        'original': 'S = f(W,Y)',
        'con_pca': 'S = a·W(1 + θ·PCA) + b·Y',
        'descripcion': 'Hipótesis del ciclo de vida',
        'parametros': {'a': 0.05, 'b': 0.1, 'theta': 0.08}
    },
    'carroll': {
        'nombre': 'Carroll & Weil (1994)',
        'original': 'S = f(Y,r)',
        'con_pca': 'S = f(Y) + r(1 + φ·PCA)',
        'descripcion': 'Modelo de crecimiento y ahorro',
        'parametros': {'base_saving_rate': 0.12, 'phi': 0.2}
    },
    'deaton': {
        'nombre': 'Deaton-Carroll (1991-92)',
        'original': 'S = f(Y,expectativas)',
        'con_pca': 'S = f(Y,expectativas·(1 + κ·PCA))',
        'descripcion': 'Modelo de expectativas de consumo',
        'parametros': {'base_rate': 0.18, 'kappa': 0.12}
    }
}

# Etiquetas actualizadas
EDAD_LABELS = {
    1: 'Menos de 26', 2: '26-30', 3: '31-35', 4: '36-40', 5: '41-45',
    6: '46-50', 7: '51-55', 8: '56-60', 9: 'Más de 60'
}

EDUCACION_LABELS = {
    1: 'Primaria', 2: 'Bachillerato', 3: 'T.S.U.',
    4: 'Universitario', 5: 'Postgrado', 6: 'Doctorado'
}

INGRESO_LABELS = {
    1: '$3-100', 2: '$101-450', 3: '$451-1800',
    4: '$1801-2500', 5: '$2501-10000', 6: 'Más de $10000'
}

# Valores por defecto (medias)
DEFAULT_VALUES = {
    'edad': 5,      # 41-45 años (valor medio)
    'educacion': 4,  # Universitario (valor medio)
    'ingresos': 3    # $451-1800 (valor medio)
}


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
                st.success(
                    "✅ Datos cargados exitosamente desde archivos locales")
                return scores_df, items_df
            except Exception as e:
                st.error(f"Error al leer archivos Excel: {str(e)}")

        # Generar datos simulados si no hay archivos
        return generar_datos_simulados_avanzados()

    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return generar_datos_simulados_avanzados()


def generar_datos_simulados_avanzados():
    """Genera datos simulados más sofisticados basados en las estadísticas reales"""
    np.random.seed(42)
    n_samples = 1000

    # Generar datos para ambos grupos usando las estadísticas reales
    datos_completos = []

    for grupo in ['Hah', 'Mah']:
        stats_grupo = MODELOS_COEFICIENTES[grupo]['stats']
        n_grupo = n_samples // 2

        # Generar variables con distribuciones realistas
        pse_data = generar_variable_con_stats(stats_grupo['PSE'], n_grupo)
        pca_data = generar_variable_con_stats(stats_grupo['PCA'], n_grupo)
        av_data = generar_variable_con_stats(stats_grupo['AV'], n_grupo)
        dh_data = generar_variable_con_stats(stats_grupo['DH'], n_grupo)
        sq_data = generar_variable_con_stats(stats_grupo['SQ'], n_grupo)
        cs_data = generar_variable_con_stats(stats_grupo['CS'], n_grupo)

        # Crear DataFrame para el grupo
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

    # Generar items_df correspondiente
    items_data = []

    for _, row in scores_df.iterrows():
        items_row = {
            'Case': row['Case'],
            'PCA2': np.random.randint(1, 10),  # Edad
            'PCA4': np.random.randint(1, 7),   # Educación
            'PCA5': np.random.randint(1, 7),   # Ingresos
            'PPCA': row['PCA'] + np.random.normal(0, 0.1),
            'GRUPO': row['GRUPO']
        }

        # Generar items específicos basados en los weights
        grupo = row['GRUPO']
        weights = MODELOS_COEFICIENTES[grupo]['weights']

        # Items AV
        for item in ['AV1', 'AV2', 'AV3', 'AV5']:
            if item in weights['AV']:
                items_row[item] = row['AV'] * \
                    weights['AV'][item] + np.random.normal(0, 0.2)
            else:
                items_row[item] = np.random.normal(0, 1)

        # Items DH
        for item in ['DH2', 'DH3', 'DH4', 'DH5']:
            if item in weights['DH']:
                items_row[item] = row['DH'] * \
                    weights['DH'][item] + np.random.normal(0, 0.2)
            else:
                items_row[item] = np.random.normal(0, 1)

        # Items SQ
        for item in ['SQ1', 'SQ2', 'SQ3']:
            items_row[item] = row['SQ'] * \
                weights['SQ'][item] + np.random.normal(0, 0.2)

        # Items CS
        for item in ['CS2', 'CS3', 'CS5']:
            items_row[item] = row['CS'] * \
                weights['CS'][item] + np.random.normal(0, 0.2)

        items_data.append(items_row)

    items_df = pd.DataFrame(items_data)

    return scores_df, items_df


def generar_variable_con_stats(stats, n):
    """Genera variable con estadísticas específicas usando transformación Johnson"""
    # Generar datos base
    data = np.random.normal(0, 1, n)

    # Ajustar para conseguir asimetría y curtosis aproximadas
    if abs(stats['skew']) > 0.1:
        data = data + stats['skew'] * (data**2 - 1) / 6

    if abs(stats['kurt']) > 0.1:
        data = data + stats['kurt'] * (data**3 - 3*data) / 24

    # Escalar y trasladar
    data = (data - np.mean(data)) / np.std(data) * stats['std'] + stats['mean']

    # Asegurar que esté dentro de los límites
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

    # Si no se proporcionan weights específicos, usar distribución base
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
        # Usar weights de items para mayor precisión
        base_dh = np.array([items_weights.get('DH', 0.0)] * n_samples)
        base_cs = np.array([items_weights.get('CS', 0.0)] * n_samples)
        base_av = np.array([items_weights.get('AV', 0.0)] * n_samples)
        base_sq = np.array([items_weights.get('SQ', 0.0)] * n_samples)

    # Aplicar factores de escenario
    dh_ajustado = base_dh * escenario_config['factor_dh'] + \
        np.random.normal(0, 0.1 * escenario_config['volatilidad'], n_samples)

    cs_ajustado = base_cs * escenario_config['factor_cs'] + \
        np.random.normal(0, 0.1 * escenario_config['volatilidad'], n_samples)

    av_ajustado = base_av * escenario_config['factor_av'] + \
        np.random.normal(0, 0.1 * escenario_config['volatilidad'], n_samples)

    sq_ajustado = base_sq * escenario_config['factor_sq'] + \
        np.random.normal(0, 0.1 * escenario_config['volatilidad'], n_samples)

    # Para Hah, AV afecta a SQ (como mencionas en la especificación)
    if grupo == 'Hah':
        sq_ajustado = sq_ajustado + 0.3 * av_ajustado  # SQ ya incorpora la carga de AV

    # Asegurar que estén dentro de los rangos esperados
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

    # Calcular PSE base
    pse_base = calcular_pse(pca2, pca4, pca5, grupo)

    resultados = {
        'pca_values': [],
        'variables_cognitivas': {'DH': [], 'CS': [], 'AV': [], 'SQ': []},
        'pse_values': [],
        'escenario': escenario,
        'modelos_externos': {modelo: {'original': [], 'con_pca': []} for modelo in MODELOS_EXTERNOS.keys()}
    }

    for i in range(n_simulaciones):
        # Generar variables cognitivas ajustadas por escenario
        vars_cognitivas = generar_variables_cognitivas_con_escenario(
            grupo, escenario)

        # Almacenar variables cognitivas
        for var in ['DH', 'CS', 'AV', 'SQ']:
            resultados['variables_cognitivas'][var].append(
                vars_cognitivas[var])

        # Calcular PSE con variación
        pse_actual = pse_base + np.random.normal(0, 0.1)
        resultados['pse_values'].append(pse_actual)

        # Calcular PCA usando el modelo estructural
        pca_value = calcular_pca_teorica(
            pse_actual,
            vars_cognitivas['DH'],
            vars_cognitivas['SQ'],
            vars_cognitivas['CS'],
            grupo
        )

        # Añadir error residual basado en RMSE del modelo
        error_residual = np.random.normal(
            0, MODELOS_COEFICIENTES[grupo]['rmse'] * 0.1)
        pca_value += error_residual

        resultados['pca_values'].append(pca_value)

        # Generar variables económicas base con variación por escenario
        volatilidad = ESCENARIOS_ECONOMICOS[escenario]['volatilidad']
        y = abs(np.random.normal(1000, 200 * volatilidad))
        w = abs(np.random.normal(5000, 1000 * volatilidad))
        r = np.random.normal(0.05, 0.02 * volatilidad)

        # Simular cada modelo externo
        for modelo_key in MODELOS_EXTERNOS.keys():
            s_orig, s_pca = simular_modelo_externo(
                modelo_key, pca_value, y, w, r)
            resultados['modelos_externos'][modelo_key]['original'].append(
                s_orig)
            resultados['modelos_externos'][modelo_key]['con_pca'].append(s_pca)

    return resultados


def crear_grafico_3d_escenarios(resultados_baseline, resultados_crisis, resultados_bonanza, grupo):
    """Crea visualización 3D de los tres escenarios"""
    fig = go.Figure()

    # Escenario Baseline
    fig.add_trace(go.Scatter3d(
        # Limitar para mejor visualización
        x=resultados_baseline['pse_values'][:500],
        y=resultados_baseline['variables_cognitivas']['CS'][:500],
        z=resultados_baseline['pca_values'][:500],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.6,
            colorscale='Blues'
        ),
        name='⚖️ Baseline',
        hovertemplate='<b>Baseline</b><br>PSE: %{x:.3f}<br>CS: %{y:.3f}<br>PCA: %{z:.3f}<extra></extra>'
    ))

    # Escenario Crisis
    fig.add_trace(go.Scatter3d(
        x=resultados_crisis['pse_values'][:500],
        y=resultados_crisis['variables_cognitivas']['CS'][:500],
        z=resultados_crisis['pca_values'][:500],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.6,
            colorscale='Reds'
        ),
        name='🔴 Crisis',
        hovertemplate='<b>Crisis</b><br>PSE: %{x:.3f}<br>CS: %{y:.3f}<br>PCA: %{z:.3f}<extra></extra>'
    ))

    # Escenario Bonanza
    fig.add_trace(go.Scatter3d(
        x=resultados_bonanza['pse_values'][:500],
        y=resultados_bonanza['variables_cognitivas']['CS'][:500],
        z=resultados_bonanza['pca_values'][:500],
        mode='markers',
        marker=dict(
            size=3,
            color='green',
            opacity=0.6,
            colorscale='Greens'
        ),
        name='🟢 Bonanza',
        hovertemplate='<b>Bonanza</b><br>PSE: %{x:.3f}<br>CS: %{y:.3f}<br>PCA: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'🧠 Visualización 3D: PSE vs CS vs PCA - {grupo}',
        scene=dict(
            xaxis_title='PSE (Perfil Socioeconómico)',
            yaxis_title='CS (Contagio Social)',
            zaxis_title='PCA (Propensión al Ahorro)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=900,
        height=700,
        font=dict(size=12)
    )

    return fig


def crear_dashboard_comparativo(resultados_dict):
    """Crea dashboard comparativo de los tres escenarios"""
    # Preparar datos para comparación
    escenarios = ['baseline', 'crisis', 'bonanza']
    colores = ['#6c5ce7', '#e74c3c', '#27ae60']

    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '📊 Distribución PCA por Escenario',
            '🧠 Variables Cognitivas Promedio',
            '💰 Impacto en Modelos Económicos',
            '📈 Correlaciones PSE-PCA'
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
    modelo_keynes = 'keynes'  # Usar Keynes como ejemplo
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
            name='Impacto % Keynes',
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
                x=resultados['pse_values'][:200],  # Muestra limitada
                y=resultados['pca_values'][:200],
                mode='markers',
                name=f'Correlación {ESCENARIOS_ECONOMICOS[escenario]["nombre"]}',
                marker=dict(color=colores[i], size=4, opacity=0.6)
            ),
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="🎯 Dashboard Comparativo de Escenarios Económicos"
    )

    return fig


def crear_radar_chart_cognitivo(resultados_dict, grupo):
    """Crea gráfico radar para variables cognitivas"""
    categories = ['DH<br>(Descuento<br>Hiperbólico)',
                  'CS<br>(Contagio<br>Social)',
                  'AV<br>(Aversión<br>Pérdidas)',
                  'SQ<br>(Status<br>Quo)']

    fig = go.Figure()

    for escenario, color in [('baseline', 'blue'), ('crisis', 'red'), ('bonanza', 'green')]:
        resultados = resultados_dict[escenario]
        valores = []

        for var in ['DH', 'CS', 'AV', 'SQ']:
            media = np.mean(resultados['variables_cognitivas'][var])
            # Normalizar entre 0 y 1 para el radar
            stats = MODELOS_COEFICIENTES[grupo]['stats'][var]
            valor_norm = (media - stats['min']) / (stats['max'] - stats['min'])
            valores.append(valor_norm)

        fig.add_trace(go.Scatterpolar(
            r=valores + [valores[0]],  # Cerrar el polígono
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
                ticktext=['Mín', '25%', '50%', '75%', 'Máx']
            )
        ),
        showlegend=True,
        title=f"🧠 Radar de Sesgos Cognitivos - {grupo}",
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


def main():
    st.title("🧠 PCA Simulator v2.0 - Crisis & Bonanza Analysis")

    # Sidebar mejorado con diseño premium
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; margin-bottom: 1rem;'>
            <h3 style='margin: 0;'>⚙️ Panel de Control</h3>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>Configuración Avanzada</p>
        </div>
        """, unsafe_allow_html=True)

        # Selección de grupo
        grupo = st.selectbox(
            "👥 **Grupo de Análisis**",
            options=['Hah', 'Mah'],
            format_func=lambda x: f"{'👨 Hombres Ahorradores' if x == 'Hah' else '👩 Mujeres Ahorradoras'} ({x})"
        )

        # Mostrar métricas del modelo seleccionado
        model_stats = MODELOS_COEFICIENTES[grupo]
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Métricas del Modelo {grupo}</h4>
            <p><strong>R²:</strong> {model_stats['r2']:.4f}</p>
            <p><strong>RMSE:</strong> {model_stats['rmse']:.4f}</p>
            <p><strong>Correlación:</strong> {model_stats['correlation']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Selección de escenario con diseño visual
        st.markdown("🎭 **Escenario Económico**")
        escenario = st.radio(
            "Seleccione el contexto económico:",
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

        # Variables socioeconómicas con valores por defecto
        st.markdown("📊 **Variables Socioeconómicas**")

        pca2 = st.slider(
            "🎂 **Edad**",
            min_value=1, max_value=9,
            value=DEFAULT_VALUES['edad'],
            help="Grupo etario del participante"
        )
        st.caption(f"📍 {EDAD_LABELS[pca2]}")

        pca4 = st.slider(
            "🎓 **Nivel Educativo**",
            min_value=1, max_value=6,
            value=DEFAULT_VALUES['educacion'],
            help="Máximo nivel educativo alcanzado"
        )
        st.caption(f"📍 {EDUCACION_LABELS[pca4]}")

        pca5 = st.slider(
            "💰 **Nivel de Ingresos**",
            min_value=1, max_value=6,
            value=DEFAULT_VALUES['ingresos'],
            help="Rango de ingresos mensuales en USD"
        )
        st.caption(f"📍 {INGRESO_LABELS[pca5]}")

        st.markdown("---")

        # Parámetros de simulación
        st.markdown("🔬 **Parámetros de Simulación**")
        n_simulaciones = st.number_input(
            "Número de simulaciones Monte Carlo",
            min_value=1000, max_value=15000,
            value=5000, step=1000,
            help="Mayor número = mayor precisión, pero más tiempo de cálculo"
        )

        # Opción de análisis comparativo
        analisis_comparativo = st.checkbox(
            "📈 **Análisis Comparativo Multi-escenario**",
            value=True,
            help="Ejecuta simulación en los 3 escenarios para análisis comparativo"
        )

    # Cargar datos
    with st.spinner("🔄 Cargando base de datos..."):
        scores_df, items_df = cargar_datos()

    # Mostrar información del modelo actual
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("### 📐 Modelo PLS-SEM Activo")
        st.code(MODELOS_COEFICIENTES[grupo]['ecuacion'], language='text')

        pse_calculado = calcular_pse(pca2, pca4, pca5, grupo)
        st.info(f"💡 **PSE Calculado:** {pse_calculado:.4f}")

    with col2:
        st.markdown("### 👤 Perfil Actual")
        st.metric("Grupo", f"{'👨 Hombres' if grupo == 'Hah' else '👩 Mujeres'}")
        st.metric("Edad", EDAD_LABELS[pca2])
        st.metric("Educación", EDUCACION_LABELS[pca4])

    with col3:
        st.markdown("### 🎯 Contexto")
        st.metric("Ingresos", INGRESO_LABELS[pca5])
        st.metric("Escenario", escenario_info['nombre'])
        st.metric("Simulaciones", f"{n_simulaciones:,}")

    # Botón de simulación con diseño premium
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        if st.button(
            f"🚀 **EJECUTAR SIMULACIÓN AVANZADA**" +
            (f" - {escenario_info['nombre']}" if not analisis_comparativo else " - MULTI-ESCENARIO"),
            type="primary",
            use_container_width=True
        ):
            # Ejecutar simulación(es)
            if analisis_comparativo:
                st.markdown("### 🔄 Ejecutando Análisis Multi-escenario...")

                progress_bar = st.progress(0)
                resultados_dict = {}

                for i, esc in enumerate(['baseline', 'crisis', 'bonanza']):
                    with st.spinner(f"Simulando escenario {ESCENARIOS_ECONOMICOS[esc]['nombre']}..."):
                        resultados_dict[esc] = ejecutar_simulacion_monte_carlo_avanzada(
                            pca2, pca4, pca5, grupo, esc, n_simulaciones
                        )
                    progress_bar.progress((i + 1) / 3)

                st.success(
                    f"✅ **Análisis Multi-escenario Completado:** {n_simulaciones:,} × 3 simulaciones")

                # Crear tabs mejoradas
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🎯 Dashboard Comparativo",
                    "🌐 Visualización 3D",
                    "🧠 Análisis Cognitivo",
                    "💹 Impacto Económico",
                    "📊 Estadísticas Avanzadas",
                    "🔬 Validación & Diagnósticos"
                ])

                with tab1:
                    st.markdown("### 🎯 Dashboard Comparativo de Escenarios")

                    # Dashboard principal
                    fig_dashboard = crear_dashboard_comparativo(
                        resultados_dict)
                    st.plotly_chart(fig_dashboard, use_container_width=True)

                    # Métricas comparativas
                    st.markdown("### 📈 Métricas Comparativas PCA")
                    col1, col2, col3 = st.columns(3)

                    for i, (esc, col) in enumerate(zip(['baseline', 'crisis', 'bonanza'], [col1, col2, col3])):
                        with col:
                            resultados = resultados_dict[esc]
                            media_pca = np.mean(resultados['pca_values'])
                            std_pca = np.std(resultados['pca_values'])

                            st.markdown(f"""
                            <div style="background: {ESCENARIOS_ECONOMICOS[esc]['color']}20; 
                                        padding: 1rem; border-radius: 8px; text-align: center;">
                                <h4 style="color: {ESCENARIOS_ECONOMICOS[esc]['color']};">
                                    {ESCENARIOS_ECONOMICOS[esc]['nombre']}
                                </h4>
                                <p><strong>Media PCA:</strong> {media_pca:.4f}</p>
                                <p><strong>Std PCA:</strong> {std_pca:.4f}</p>
                                <p><strong>CV:</strong> {(std_pca/media_pca)*100 if media_pca != 0 else 0:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                with tab2:
                    st.markdown("### 🌐 Visualización 3D Multi-escenario")

                    # Gráfico 3D
                    fig_3d = crear_grafico_3d_escenarios(
                        resultados_dict['baseline'],
                        resultados_dict['crisis'],
                        resultados_dict['bonanza'],
                        grupo
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                    # Interpretación
                    st.markdown("""
                    **🔍 Interpretación de la Visualización 3D:**
                    - **Eje X (PSE):** Perfil Socioeconómico constante por diseño
                    - **Eje Y (CS):** Contagio Social, varia significativamente entre escenarios
                    - **Eje Z (PCA):** Propensión Conductual al Ahorro, variable dependiente
                    - **Clustering:** Observe cómo se agrupan los puntos según el escenario
                    """)

                with tab3:
                    st.markdown(
                        "### 🧠 Análisis Detallado de Sesgos Cognitivos")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        # Radar chart
                        fig_radar = crear_radar_chart_cognitivo(
                            resultados_dict, grupo)
                        st.plotly_chart(fig_radar, use_container_width=True)

                    with col2:
                        # Tabla de cambios relativos
                        st.markdown("#### 📊 Cambios Relativos vs Baseline")

                        datos_cambios = []
                        baseline_vals = resultados_dict['baseline']['variables_cognitivas']

                        for var in ['DH', 'CS', 'AV', 'SQ']:
                            baseline_mean = np.mean(baseline_vals[var])
                            crisis_mean = np.mean(
                                resultados_dict['crisis']['variables_cognitivas'][var])
                            bonanza_mean = np.mean(
                                resultados_dict['bonanza']['variables_cognitivas'][var])

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
                        st.dataframe(df_cambios.round(
                            3), use_container_width=True)

                        # Interpretaciones psicológicas
                        st.markdown("""
                        **🧠 Interpretación Psicológica:**
                        
                        - **DH (Descuento Hiperbólico):** ↗️ Crisis = mayor impaciencia
                        - **CS (Contagio Social):** ↗️ Crisis = mayor influencia social
                        - **AV (Aversión Pérdidas):** ↗️ Crisis = mayor miedo al riesgo
                        - **SQ (Status Quo):** ↗️ Crisis = mayor resistencia al cambio
                        """)

                with tab4:
                    st.markdown(
                        "### 💹 Análisis de Impacto en Modelos Económicos")

                    # Análisis detallado por modelo económico
                    modelo_analizar = st.selectbox(
                        "Seleccione modelo económico para análisis detallado:",
                        options=list(MODELOS_EXTERNOS.keys()),
                        format_func=lambda x: MODELOS_EXTERNOS[x]['nombre']
                    )

                    # Calcular impactos para todos los escenarios
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Gráfico de violín comparativo
                        fig_violin = go.Figure()

                        colores_violin = ['blue', 'red', 'green']
                        for i, esc in enumerate(['baseline', 'crisis', 'bonanza']):
                            resultados = resultados_dict[esc]
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
                            title=f'💰 Distribución de Ahorro - {MODELOS_EXTERNOS[modelo_analizar]["nombre"]}',
                            yaxis_title='Ahorro Proyectado',
                            height=400
                        )

                        st.plotly_chart(fig_violin, use_container_width=True)

                    with col2:
                        # Tabla de estadísticas del modelo
                        st.markdown("#### 📊 Estadísticas del Modelo")

                        stats_modelo = []
                        for esc in ['baseline', 'crisis', 'bonanza']:
                            resultados = resultados_dict[esc]
                            datos = resultados['modelos_externos'][modelo_analizar]['con_pca']
                            stats = calcular_estadisticas_avanzadas(datos)

                            stats_modelo.append({
                                'Escenario': ESCENARIOS_ECONOMICOS[esc]['nombre'],
                                'Media': stats['media'],
                                'Mediana': stats['mediana'],
                                'Std': stats['std'],
                                'CV%': stats['cv'] * 100
                            })

                        df_stats_modelo = pd.DataFrame(stats_modelo)
                        st.dataframe(df_stats_modelo.round(2),
                                     use_container_width=True)

                    # Matriz de impactos todos los modelos
                    st.markdown("#### 🎯 Matriz de Impactos Multi-modelo")

                    matriz_impactos = []
                    for modelo_key, modelo_info in MODELOS_EXTERNOS.items():
                        fila = {'Modelo': modelo_info['nombre']}

                        for esc in ['baseline', 'crisis', 'bonanza']:
                            resultados = resultados_dict[esc]
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
                    st.markdown("### 📊 Estadísticas Descriptivas Avanzadas")

                    # Selector de variable para análisis
                    variable_analisis = st.selectbox(
                        "Variable para análisis estadístico detallado:",
                        options=['PCA'] + list(['DH', 'CS', 'AV', 'SQ']),
                        index=0
                    )

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        # Histograma comparativo
                        fig_hist_comp = go.Figure()

                        for esc, color in [('baseline', 'blue'), ('crisis', 'red'), ('bonanza', 'green')]:
                            if variable_analisis == 'PCA':
                                datos = resultados_dict[esc]['pca_values']
                            else:
                                datos = resultados_dict[esc]['variables_cognitivas'][variable_analisis]

                            fig_hist_comp.add_trace(go.Histogram(
                                x=datos,
                                name=ESCENARIOS_ECONOMICOS[esc]['nombre'],
                                opacity=0.6,
                                nbinsx=30,
                                marker_color=color
                            ))

                        fig_hist_comp.update_layout(
                            title=f'📈 Distribución Comparativa - {variable_analisis}',
                            xaxis_title=variable_analisis,
                            yaxis_title='Frecuencia',
                            barmode='overlay',
                            height=400
                        )

                        st.plotly_chart(
                            fig_hist_comp, use_container_width=True)

                    with col2:
                        # Tests estadísticos
                        st.markdown("#### 🔬 Tests Estadísticos")

                        # Preparar datos para tests
                        datos_tests = {}
                        for esc in ['baseline', 'crisis', 'bonanza']:
                            if variable_analisis == 'PCA':
                                datos_tests[esc] = resultados_dict[esc]['pca_values']
                            else:
                                datos_tests[esc] = resultados_dict[esc]['variables_cognitivas'][variable_analisis]

                        # Test de normalida
if __name__ == "__main__":
    main()
