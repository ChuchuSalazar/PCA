"""
Implementación de modelos PLS-SEM para cálculo de PCA
Contiene las ecuaciones estructurales y funciones de cálculo
"""

import numpy as np
from config.constants import MODELOS_COEFICIENTES


def calcular_pca_teorica(pse, dh, sq, cs, grupo):
    """
    Calcula PCA usando la ecuación del modelo PLS-SEM

    Args:
        pse: Propensión al ahorro esperada
        dh: Descuento hiperbólico
        sq: Status quo bias
        cs: Contagio social
        grupo: 'Hah' o 'Mah'

    Returns:
        float: Valor calculado de PCA
    """
    if grupo not in MODELOS_COEFICIENTES:
        raise ValueError(f"Unknown group: {grupo}")

    coef = MODELOS_COEFICIENTES[grupo]['coef']

    pca = (coef['PSE'] * pse +
           coef['DH'] * dh +
           coef['SQ'] * sq +
           coef['CS'] * cs)

    return pca


def calcular_efectos_directos(grupo, variables):
    """
    Calcula todos los efectos directos del modelo PLS-SEM

    Args:
        grupo: 'Hah' o 'Mah'
        variables: Dict con valores de variables

    Returns:
        dict: Efectos directos calculados
    """
    coef = MODELOS_COEFICIENTES[grupo]['coef']

    efectos = {
        'PSE_to_PCA': coef['PSE'] * variables.get('PSE', 0),
        'DH_to_PCA': coef['DH'] * variables.get('DH', 0),
        'SQ_to_PCA': coef['SQ'] * variables.get('SQ', 0),
        'CS_to_PCA': coef['CS'] * variables.get('CS', 0)
    }

    return efectos


def calcular_r_cuadrado_ajustado(grupo, n_observaciones):
    """
    Calcula R² ajustado para el modelo

    Args:
        grupo: 'Hah' o 'Mah'
        n_observaciones: Número de observaciones

    Returns:
        float: R² ajustado
    """
    modelo_info = MODELOS_COEFICIENTES[grupo]
    r2 = modelo_info['r2']

    # Número de predictores en el modelo
    k = 4  # PSE, DH, SQ, CS

    if n_observaciones <= k:
        return r2

    r2_ajustado = 1 - ((1 - r2) * (n_observaciones - 1) /
                       (n_observaciones - k - 1))

    return r2_ajustado


def evaluar_bondad_ajuste(grupo, valores_observados, valores_predichos):
    """
    Evalúa la bondad de ajuste del modelo PLS-SEM

    Args:
        grupo: 'Hah' o 'Mah'
        valores_observados: Valores observados de PCA
        valores_predichos: Valores predichos por el modelo

    Returns:
        dict: Métricas de bondad de ajuste
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # Métricas calculadas
    rmse_calculado = np.sqrt(mean_squared_error(
        valores_observados, valores_predichos))
    mae_calculado = mean_absolute_error(valores_observados, valores_predichos)
    r2_calculado = r2_score(valores_observados, valores_predichos)

    # Métricas del modelo original
    modelo_info = MODELOS_COEFICIENTES[grupo]
    rmse_original = modelo_info['rmse']
    mae_original = modelo_info['mae']
    r2_original = modelo_info['r2']

    bondad_ajuste = {
        'rmse_calculado': rmse_calculado,
        'rmse_original': rmse_original,
        'rmse_diferencia': abs(rmse_calculado - rmse_original),
        'mae_calculado': mae_calculado,
        'mae_original': mae_original,
        'mae_diferencia': abs(mae_calculado - mae_original),
        'r2_calculado': r2_calculado,
        'r2_original': r2_original,
        'r2_diferencia': abs(r2_calculado - r2_original),
        'ajuste_aceptable': abs(rmse_calculado - rmse_original) < 0.1
    }

    return bondad_ajuste
