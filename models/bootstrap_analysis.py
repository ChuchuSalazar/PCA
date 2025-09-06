"""
Módulo de análisis Bootstrap para el simulador PCA
Implementa remuestreo bootstrap y análisis estadístico avanzado
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.utils import resample
from datetime import datetime
from config.constants import MODELOS_COEFICIENTES, ESCENARIOS_ECONOMICOS
from models.economic_models import simular_modelo_externo
from models.pls_sem_models import calcular_pca_teorica


def ejecutar_bootstrap_avanzado(grupo, escenario, n_bootstrap=3000):
    """
    Ejecuta análisis Bootstrap en lugar de Monte Carlo

    Args:
        grupo: 'Hah' o 'Mah'
        escenario: 'baseline', 'crisis', 'bonanza'
        n_bootstrap: Número de iteraciones bootstrap

    Returns:
        dict: Resultados completos del análisis bootstrap
    """
    try:
        np.random.seed(42)

        # Importar datos (evitar importación circular)
        from data.data_loader import cargar_datos
        scores_df, items_df = cargar_datos()
        grupo_data = scores_df[scores_df['GRUPO'] == grupo].copy()

        if len(grupo_data) == 0:
            st.error(f"No data found for group {grupo}")
            return None

        # Configuración del escenario
        escenario_config = ESCENARIOS_ECONOMICOS[escenario]

        # Inicializar estructura de resultados
        resultados = inicializar_estructura_resultados(
            grupo, escenario, n_bootstrap, len(grupo_data))

        # Ejecutar bootstrap
        ejecutar_iteraciones_bootstrap(
            grupo_data, grupo, escenario_config, n_bootstrap, resultados
        )

        # Calcular estadísticas bootstrap
        calcular_estadisticas_bootstrap(resultados, grupo_data)

        return resultados

    except Exception as e:
        st.error(f"Error in bootstrap analysis: {str(e)}")
        return None


def inicializar_estructura_resultados(grupo, escenario, n_bootstrap, n_original):
    """Inicializa la estructura de resultados bootstrap"""
    return {
        'pca_values': [],
        'variables_cognitivas': {'DH': [], 'CS': [], 'AV': [], 'SQ': []},
        'pse_values': [],
        'escenario': escenario,
        'modelos_externos': {
            'keynes': {'original': [], 'con_pca': []},
            'friedman': {'original': [], 'con_pca': []},
            'modigliani': {'original': [], 'con_pca': []},
            'carroll': {'original': [], 'con_pca': []},
            'deaton': {'original': [], 'con_pca': []}
        },
        'bootstrap_stats': {'original_n': n_original, 'bootstrap_n': n_bootstrap},
        'parametros_simulacion': {
            'grupo': grupo,
            'escenario': escenario,
            'n_bootstrap': n_bootstrap,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'methodology': 'Bootstrap Resampling'
        }
    }


def ejecutar_iteraciones_bootstrap(grupo_data, grupo, escenario_config, n_bootstrap, resultados):
    """
    Ejecuta las iteraciones bootstrap

    Args:
        grupo_data: Datos del grupo
        grupo: Identificador del grupo
        escenario_config: Configuración del escenario
        n_bootstrap: Número de iteraciones
        resultados: Diccionario de resultados
    """
    noise_factor = escenario_config['bootstrap_noise']
    stats = MODELOS_COEFICIENTES[grupo]['stats']

    for i in range(n_bootstrap):
        # Bootstrap resampling
        bootstrap_sample = resample(
            grupo_data, n_samples=len(grupo_data), random_state=i)

        # Procesar cada observación en la muestra bootstrap
        for idx, row in bootstrap_sample.iterrows():
            # Ajustar variables cognitivas según escenario
            variables_ajustadas = ajustar_variables_por_escenario(
                row, escenario_config, noise_factor, stats, grupo
            )

            # Almacenar variables cognitivas
            for var, valor in variables_ajustadas.items():
                if var != 'PSE':
                    resultados['variables_cognitivas'][var].append(valor)

            # PSE con ruido mínimo
            pse_bootstrap = row['PSE'] + \
                np.random.normal(0, noise_factor * 0.5)
            resultados['pse_values'].append(pse_bootstrap)

            # Calcular PCA usando modelo PLS-SEM
            pca_value = calcular_pca_teorica(
                pse_bootstrap,
                variables_ajustadas['DH'],
                variables_ajustadas['SQ'],
                variables_ajustadas['CS'],
                grupo
            )

            # Añadir ruido del modelo
            model_noise = np.random.normal(
                0, MODELOS_COEFICIENTES[grupo]['rmse'] * 0.1)
            pca_value += model_noise

            resultados['pca_values'].append(pca_value)

            # Simular modelos económicos externos
            simular_modelos_economicos_bootstrap(
                pca_value, escenario_config, resultados
            )


def ajustar_variables_por_escenario(row, escenario_config, noise_factor, stats, grupo):
    """
    Ajusta las variables cognitivas según el escenario económico

    Args:
        row: Fila de datos
        escenario_config: Configuración del escenario
        noise_factor: Factor de ruido
        stats: Estadísticas del grupo
        grupo: Identificador del grupo

    Returns:
        dict: Variables ajustadas
    """
    variables_ajustadas = {}

    # Ajustar cada variable cognitiva
    for var in ['DH', 'CS', 'AV', 'SQ']:
        factor_key = f'factor_{var.lower()}'
        factor = escenario_config.get(factor_key, 1.0)

        # Aplicar factor del escenario y ruido
        valor_ajustado = row[var] * factor + np.random.normal(0, noise_factor)

        # Aplicar límites estadísticos
        valor_ajustado = np.clip(
            valor_ajustado,
            stats[var]['min'],
            stats[var]['max']
        )

        variables_ajustadas[var] = valor_ajustado

    return variables_ajustadas


def simular_modelos_economicos_bootstrap(pca_value, escenario_config, resultados):
    """
    Simula modelos económicos externos para la iteración bootstrap

    Args:
        pca_value: Valor PCA calculado
        escenario_config: Configuración del escenario
        resultados: Diccionario de resultados
    """
    volatilidad = escenario_config['volatilidad']

    # Generar variables económicas base
    y = abs(np.random.normal(1000, 200 * volatilidad))
    w = abs(np.random.normal(5000, 1000 * volatilidad))
    r = np.random.normal(0.05, 0.02 * volatilidad)

    # Simular cada modelo económico
    for modelo_key in ['keynes', 'friedman', 'modigliani', 'carroll', 'deaton']:
        s_orig, s_pca = simular_modelo_externo(modelo_key, pca_value, y, w, r)
        resultados['modelos_externos'][modelo_key]['original'].append(s_orig)
        resultados['modelos_externos'][modelo_key]['con_pca'].append(s_pca)


def calcular_estadisticas_bootstrap(resultados, grupo_data):
    """
    Calcula estadísticas finales del análisis bootstrap

    Args:
        resultados: Diccionario de resultados
        grupo_data: Datos originales del grupo
    """
    pca_array = np.array(resultados['pca_values'])
    original_pca_mean = np.mean(grupo_data['PCA'])

    # Estadísticas bootstrap principales
    bootstrap_mean = np.mean(pca_array)
    bootstrap_std = np.std(pca_array)

    # Intervalos de confianza
    ci_lower = np.percentile(pca_array, 2.5)
    ci_upper = np.percentile(pca_array, 97.5)

    # Corrección de sesgo
    bias_corrected_mean = 2 * original_pca_mean - bootstrap_mean

    # Actualizar estadísticas
    resultados['bootstrap_stats'].update({
        'pca_mean': bootstrap_mean,
        'pca_std': bootstrap_std,
        'pca_ci_lower': ci_lower,
        'pca_ci_upper': ci_upper,
        'bias_corrected_mean': bias_corrected_mean,
        'original_mean': original_pca_mean,
        'bootstrap_se': bootstrap_std / np.sqrt(len(pca_array))
    })
