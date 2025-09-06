"""
Módulo para carga y manejo de datos del simulador PCA
Maneja tanto archivos Excel como generación de datos simulados
"""

import pandas as pd
import numpy as np
import os
import streamlit as st
from data.data_generator import generar_datos_simulados_avanzados


@st.cache_data(ttl=3600)  # Cache por 1 hora
def cargar_datos():
    """Carga los datos desde archivos Excel o genera datos simulados - OPTIMIZADO"""
    try:
        # Intentar cargar desde cache primero
        if 'cached_data' in st.session_state:
            return st.session_state.cached_data

        ruta_scores = "SCORE HM.xlsx"
        ruta_items = "Standardized Indicator Scores ITEMS.xlsx"

        if os.path.exists(ruta_scores) and os.path.exists(ruta_items):
            try:
                scores_df = pd.read_excel(ruta_scores)
                items_df = pd.read_excel(ruta_items)

                # Guardar en session_state para acceso rápido
                st.session_state.cached_data = (scores_df, items_df)
                st.success("Data successfully loaded from local files")
                return scores_df, items_df
            except Exception as e:
                st.error(f"Error reading Excel files: {str(e)}")

        # Generar datos más pequeños y cachear
        result = generar_datos_simulados_avanzados()
        st.session_state.cached_data = result
        return result

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return generar_datos_simulados_avanzados()


def validar_estructura_datos(scores_df, items_df):
    """
    Valida que los DataFrames tengan la estructura esperada

    Args:
        scores_df: DataFrame con puntuaciones
        items_df: DataFrame con ítems

    Returns:
        bool: True si la estructura es válida
    """
    try:
        # Validar columnas requeridas en scores_df
        columnas_scores_requeridas = [
            'Case', 'PSE', 'PCA', 'AV', 'DH', 'SQ', 'CS', 'GRUPO']

        for col in columnas_scores_requeridas:
            if col not in scores_df.columns:
                st.error(f"Missing column in scores data: {col}")
                return False

        # Validar que hay datos para ambos grupos
        grupos = scores_df['GRUPO'].unique()
        if 'Hah' not in grupos or 'Mah' not in grupos:
            st.error("Missing required groups (Hah/Mah) in data")
            return False

        # Validar columnas básicas en items_df
        columnas_items_requeridas = ['Case', 'GRUPO']

        for col in columnas_items_requeridas:
            if col not in items_df.columns:
                st.error(f"Missing column in items data: {col}")
                return False

        # Validar que no hay valores nulos críticos
        if scores_df[columnas_scores_requeridas].isnull().any().any():
            st.warning("Found null values in critical columns")
            return False

        return True

    except Exception as e:
        st.error(f"Error validating data structure: {str(e)}")
        return False


def procesar_datos_cargados(scores_df, items_df):
    """
    Procesa y limpia los datos cargados

    Args:
        scores_df: DataFrame con puntuaciones
        items_df: DataFrame con ítems

    Returns:
        tuple: (scores_df_processed, items_df_processed)
    """
    try:
        # Procesar scores_df
        scores_processed = scores_df.copy()

        # Convertir Case a entero si es posible
        if 'Case' in scores_processed.columns:
            scores_processed['Case'] = pd.to_numeric(
                scores_processed['Case'], errors='coerce')

        # Asegurar que las variables numéricas sean float
        numeric_cols = ['PSE', 'PCA', 'AV', 'DH', 'SQ', 'CS']
        for col in numeric_cols:
            if col in scores_processed.columns:
                scores_processed[col] = pd.to_numeric(
                    scores_processed[col], errors='coerce')

        # Procesar items_df
        items_processed = items_df.copy()

        # Convertir Case a entero
        if 'Case' in items_processed.columns:
            items_processed['Case'] = pd.to_numeric(
                items_processed['Case'], errors='coerce')

        # Eliminar filas con valores nulos críticos
        scores_processed = scores_processed.dropna(subset=['Case', 'GRUPO'])
        items_processed = items_processed.dropna(subset=['Case', 'GRUPO'])

        return scores_processed, items_processed

    except Exception as e:
        st.error(f"Error processing loaded data: {str(e)}")
        return scores_df, items_df


def obtener_estadisticas_datos(scores_df):
    """
    Obtiene estadísticas básicas de los datos cargados

    Args:
        scores_df: DataFrame con puntuaciones

    Returns:
        dict: Diccionario con estadísticas por grupo
    """
    try:
        estadisticas = {}

        for grupo in ['Hah', 'Mah']:
            datos_grupo = scores_df[scores_df['GRUPO'] == grupo]

            if len(datos_grupo) > 0:
                estadisticas[grupo] = {
                    'n_casos': len(datos_grupo),
                    'estadisticas': {}
                }

                # Calcular estadísticas para cada variable
                variables = ['PSE', 'PCA', 'AV', 'DH', 'SQ', 'CS']

                for var in variables:
                    if var in datos_grupo.columns:
                        serie = datos_grupo[var].dropna()

                        if len(serie) > 0:
                            estadisticas[grupo]['estadisticas'][var] = {
                                'mean': float(serie.mean()),
                                'std': float(serie.std()),
                                'min': float(serie.min()),
                                'max': float(serie.max()),
                                'count': len(serie)
                            }

        return estadisticas

    except Exception as e:
        st.error(f"Error calculating data statistics: {str(e)}")
        return {}


def validar_integridad_datos(scores_df, items_df):
    """
    Valida la integridad entre scores_df e items_df

    Args:
        scores_df: DataFrame con puntuaciones
        items_df: DataFrame con ítems

    Returns:
        dict: Diccionario con resultados de validación
    """
    try:
        resultados = {
            'casos_coincidentes': 0,
            'casos_solo_scores': 0,
            'casos_solo_items': 0,
            'grupos_consistentes': True,
            'errores': []
        }

        # Obtener casos únicos
        casos_scores = set(scores_df['Case'].dropna().astype(int))
        casos_items = set(items_df['Case'].dropna().astype(int))

        # Calcular coincidencias
        resultados['casos_coincidentes'] = len(
            casos_scores.intersection(casos_items))
        resultados['casos_solo_scores'] = len(casos_scores - casos_items)
        resultados['casos_solo_items'] = len(casos_items - casos_scores)

        # Validar consistencia de grupos para casos coincidentes
        casos_comunes = casos_scores.intersection(casos_items)

        for caso in casos_comunes:
            grupo_scores = scores_df[scores_df['Case']
                                     == caso]['GRUPO'].iloc[0]
            grupo_items = items_df[items_df['Case'] == caso]['GRUPO'].iloc[0]

            if grupo_scores != grupo_items:
                resultados['grupos_consistentes'] = False
                resultados['errores'].append(
                    f"Inconsistent group for Case {caso}: {grupo_scores} vs {grupo_items}")

        return resultados

    except Exception as e:
        st.error(f"Error validating data integrity: {str(e)}")
        return {'error': str(e)}


def exportar_datos_muestra(scores_df, items_df, n_muestra=10):
    """
    Exporta una muestra de los datos para inspección

    Args:
        scores_df: DataFrame con puntuaciones
        items_df: DataFrame con ítems
        n_muestra: Número de casos a incluir en la muestra

    Returns:
        tuple: (muestra_scores, muestra_items)
    """
    try:
        # Seleccionar muestra aleatoria
        casos_disponibles = list(scores_df['Case'].dropna().astype(int))

        if len(casos_disponibles) < n_muestra:
            casos_muestra = casos_disponibles
        else:
            casos_muestra = np.random.choice(
                casos_disponibles, n_muestra, replace=False)

        # Filtrar DataFrames
        muestra_scores = scores_df[scores_df['Case'].isin(
            casos_muestra)].copy()
        muestra_items = items_df[items_df['Case'].isin(casos_muestra)].copy()

        return muestra_scores, muestra_items

    except Exception as e:
        st.error(f"Error creating data sample: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()
