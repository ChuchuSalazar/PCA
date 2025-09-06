"""
Generador de datos simulados para el simulador PCA
Crea datos sintéticos basados en las estadísticas reales del modelo
"""

import pandas as pd
import numpy as np
import streamlit as st
from config.constants import MODELOS_COEFICIENTES


def generar_datos_simulados_avanzados():
    """
    Genera datos simulados más sofisticados basados en las estadísticas reales

    Returns:
        tuple: (scores_df, items_df) - DataFrames simulados
    """
    try:
        np.random.seed(42)  # Para reproducibilidad
        n_samples = 200

        datos_completos = []

        # Generar datos para cada grupo
        for grupo in ['Hah', 'Mah']:
            stats_grupo = MODELOS_COEFICIENTES[grupo]['stats']
            n_grupo = n_samples // 2

            # Generar variables con distribuciones específicas
            variables_data = {}
            for variable in ['PSE', 'PCA', 'AV', 'DH', 'SQ', 'CS']:
                variables_data[variable] = generar_variable_con_stats(
                    stats_grupo[variable], n_grupo
                )

            # Crear DataFrame para el grupo
            grupo_df = pd.DataFrame({
                'Case': range(len(datos_completos) * n_grupo + 1,
                              len(datos_completos) * n_grupo + n_grupo + 1),
                **variables_data,
                'GRUPO': grupo
            })

            datos_completos.append(grupo_df)

        # Combinar datos de ambos grupos
        scores_df = pd.concat(datos_completos, ignore_index=True)

        # Generar items simulados
        items_df = generar_items_simulados(scores_df)

        st.info(f"Generated simulated data: {len(scores_df)} cases")
        return scores_df, items_df

    except Exception as e:
        st.error(f"Error generating simulated data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def generar_variable_con_stats(stats, n):
    """
    Genera variable con estadísticas específicas usando transformación Johnson

    Args:
        stats: Diccionario con estadísticas (min, max, mean, std, skew, kurt)
        n: Número de observaciones a generar

    Returns:
        np.array: Array con los datos generados
    """
    try:
        # Generar datos base con distribución normal
        data = np.random.normal(0, 1, n)

        # Ajustar asimetría usando transformación cúbica
        if abs(stats['skew']) > 0.1:
            skew_factor = stats['skew']
            data = data + skew_factor * (data**2 - 1) / 6

        # Ajustar curtosis usando transformación cuártica
        if abs(stats['kurt']) > 0.1:
            kurt_factor = stats['kurt']
            data = data + kurt_factor * (data**3 - 3*data) / 24

        # Estandarizar y escalar
        data = (data - np.mean(data)) / np.std(data) * \
            stats['std'] + stats['mean']

        # Aplicar límites
        data = np.clip(data, stats['min'], stats['max'])

        return data

    except Exception as e:
        st.error(f"Error generating variable with stats: {str(e)}")
        return np.random.normal(stats['mean'], stats['std'], n)


def generar_items_simulados(scores_df):
    """
    Genera datos de ítems simulados basados en los scores

    Args:
        scores_df: DataFrame con puntuaciones por constructo

    Returns:
        pd.DataFrame: DataFrame con ítems simulados
    """
    try:
        items_data = []

        for _, row in scores_df.iterrows():
            # Datos básicos del caso
            items_row = {
                'Case': row['Case'],
                'GRUPO': row['GRUPO'],
                'PPCA': row['PCA'] + np.random.normal(0, 0.1)  # PCA con ruido
            }

            # Generar ítems PCA específicos
            items_row.update({
                'PCA2': np.random.randint(1, 10),
                'PCA4': np.random.randint(1, 7),
                'PCA5': np.random.randint(1, 7)
            })

            # Generar ítems basados en los weights del modelo
            grupo = row['GRUPO']
            weights = MODELOS_COEFICIENTES[grupo]['weights']

            items_row.update(
                generar_items_por_constructo(row, weights)
            )

            items_data.append(items_row)

        return pd.DataFrame(items_data)

    except Exception as e:
        st.error(f"Error generating simulated items: {str(e)}")
        return pd.DataFrame()


def generar_items_por_constructo(row, weights):
    """
    Genera ítems específicos para cada constructo basado en weights

    Args:
        row: Fila de datos con puntuaciones por constructo
        weights: Diccionario con pesos por constructo e ítem

    Returns:
        dict: Diccionario con ítems generados
    """
    items = {}

    try:
        for construct, items_weights in weights.items():
            # Obtener valor base del constructo
            if construct == 'PSE':
                base_value = row['PSE']
            else:
                base_value = row[construct]

            # Generar cada ítem del constructo
            for item, weight in items_weights.items():
                # Valor del ítem = peso * constructo + error
                item_value = base_value * weight + np.random.normal(0, 0.2)
                items[item] = item_value

        return items

    except Exception as e:
        st.error(f"Error generating items for construct: {str(e)}")
        return {}


def aplicar_correlaciones_realistas(data_matrix, target_correlations=None):
    """
    Aplica correlaciones realistas entre variables

    Args:
        data_matrix: Matriz de datos (n_samples x n_variables)
        target_correlations: Matriz de correlaciones objetivo

    Returns:
        np.array: Matriz de datos con correlaciones ajustadas
    """
    try:
        if target_correlations is None:
            # Correlaciones por defecto basadas en teoría
            target_correlations = np.array([
                [1.00,  0.35, -0.15,  0.20, -0.40,  0.25],  # PSE
                [0.35,  1.00, -0.20,  0.15, -0.55,  0.30],  # PCA
                [-0.15, -0.20,  1.00, -0.10,  0.25, -0.15],  # AV
                [0.20,  0.15, -0.10,  1.00, -0.30,  0.40],  # DH
                [-0.40, -0.55,  0.25, -0.30,  1.00, -0.35],  # SQ
                [0.25,  0.30, -0.15,  0.40, -0.35,  1.00]   # CS
            ])

        # Aplicar transformación de Cholesky para inducir correlaciones
        L = np.linalg.cholesky(target_correlations)

        # Estandarizar datos
        data_std = (data_matrix - np.mean(data_matrix, axis=0)) / \
            np.std(data_matrix, axis=0)

        # Aplicar correlaciones
        data_correlated = data_std @ L.T

        # Re-escalar a las distribuciones originales
        for i in range(data_matrix.shape[1]):
            data_correlated[:, i] = (
                data_correlated[:, i] - np.mean(data_correlated[:, i])) / np.std(data_correlated[:, i])
            data_correlated[:, i] = data_correlated[:, i] * \
                np.std(data_matrix[:, i]) + np.mean(data_matrix[:, i])

        return data_correlated

    except Exception as e:
        st.error(f"Error applying realistic correlations: {str(e)}")
        return data_matrix


def validar_datos_generados(scores_df, items_df):
    """
    Valida que los datos generados cumplan con los requisitos

    Args:
        scores_df: DataFrame con puntuaciones
        items_df: DataFrame con ítems

    Returns:
        dict: Resultados de validación
    """
    try:
        resultados = {
            'valido': True,
            'errores': [],
            'advertencias': [],
            'estadisticas': {}
        }

        # Validar estructura básica
        if len(scores_df) == 0:
            resultados['valido'] = False
            resultados['errores'].append("Scores DataFrame is empty")

        if len(items_df) == 0:
            resultados['valido'] = False
            resultados['errores'].append("Items DataFrame is empty")

        # Validar distribución por grupos
        grupos = scores_df['GRUPO'].value_counts()
        if len(grupos) != 2:
            resultados['valido'] = False
            resultados['errores'].append(
                "Expected exactly 2 groups (Hah, Mah)")

        # Validar rangos de variables
        variables = ['PSE', 'PCA', 'AV', 'DH', 'SQ', 'CS']
        for var in variables:
            if var in scores_df.columns:
                serie = scores_df[var]
                if serie.isnull().any():
                    resultados['advertencias'].append(
                        f"Null values found in {var}")

                # Estadísticas de la variable generada
                resultados['estadisticas'][var] = {
                    'mean': float(serie.mean()),
                    'std': float(serie.std()),
                    'min': float(serie.min()),
                    'max': float(serie.max())
                }

        return resultados

    except Exception as e:
        resultados = {
            'valido': False,
            'errores': [f"Error validating generated data: {str(e)}"],
            'advertencias': [],
            'estadisticas': {}
        }
        return resultados
