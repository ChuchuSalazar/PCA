"""
Modelos económicos externos para integración con PCA
Implementa los modelos clásicos de ahorro con ajustes conductuales
"""

import numpy as np
from config.constants import MODELOS_EXTERNOS


def simular_modelo_externo(modelo_key, pca_value, y_base=1000, w_base=5000, r_base=0.05):
    """
    Simula un modelo económico externo con y sin PCA

    Args:
        modelo_key: Clave del modelo ('keynes', 'friedman', etc.)
        pca_value: Valor de PCA para ajuste conductual
        y_base: Ingreso base
        w_base: Riqueza base
        r_base: Tasa de interés base

    Returns:
        tuple: (ahorro_original, ahorro_con_pca)
    """
    if modelo_key not in MODELOS_EXTERNOS:
        raise ValueError(f"Unknown economic model: {modelo_key}")

    modelo = MODELOS_EXTERNOS[modelo_key]
    params = modelo['parametros']

    if modelo_key == 'keynes':
        return simular_keynes(params, pca_value, y_base)
    elif modelo_key == 'friedman':
        return simular_friedman(params, pca_value, y_base)
    elif modelo_key == 'modigliani':
        return simular_modigliani(params, pca_value, y_base, w_base)
    elif modelo_key == 'carroll':
        return simular_carroll(params, pca_value, y_base, r_base)
    elif modelo_key == 'deaton':
        return simular_deaton(params, pca_value, y_base)
    else:
        raise ValueError(f"Model simulation not implemented for: {modelo_key}")


def simular_keynes(params, pca_value, y_base):
    """
    Modelo de ahorro keynesiano: S = a₀ + a₁Y
    Con PCA: S = a₀ + (a₁ + γ·PCA)Y

    Args:
        params: Parámetros del modelo
        pca_value: Valor PCA
        y_base: Ingreso base

    Returns:
        tuple: (ahorro_original, ahorro_con_pca)
    """
    a0 = params['a0']
    a1 = params['a1']
    gamma = params['gamma']

    # Modelo original
    s_original = a0 + a1 * y_base

    # Modelo con PCA - ajuste en propensión marginal al ahorro
    propension_ajustada = a1 + gamma * pca_value
    s_con_pca = a0 + propension_ajustada * y_base

    return s_original, s_con_pca


def simular_friedman(params, pca_value, y_base):
    """
    Hipótesis del ingreso permanente: S = f(Yₚ)
    Con PCA: S = f(Yₚ·(1 + δ·PCA))

    Args:
        params: Parámetros del modelo
        pca_value: Valor PCA
        y_base: Ingreso base

    Returns:
        tuple: (ahorro_original, ahorro_con_pca)
    """
    base_rate = params['base_rate']
    delta = params['delta']
    yp_factor = params['yp_factor']

    # Calcular ingreso permanente
    yp = y_base * yp_factor

    # Modelo original
    s_original = base_rate * yp

    # Modelo con PCA - ajuste en percepción del ingreso permanente
    yp_ajustado = yp * (1 + delta * pca_value)
    s_con_pca = base_rate * yp_ajustado

    return s_original, s_con_pca


def simular_modigliani(params, pca_value, y_base, w_base):
    """
    Hipótesis del ciclo de vida: S = f(W,Y)
    Con PCA: S = a·W(1 + θ·PCA) + b·Y

    Args:
        params: Parámetros del modelo
        pca_value: Valor PCA
        y_base: Ingreso base
        w_base: Riqueza base

    Returns:
        tuple: (ahorro_original, ahorro_con_pca)
    """
    a = params['a']
    b = params['b']
    theta = params['theta']

    # Modelo original
    s_original = a * w_base + b * y_base

    # Modelo con PCA - ajuste en sensibilidad al patrimonio
    sensibilidad_patrimonio = a * (1 + theta * pca_value)
    s_con_pca = sensibilidad_patrimonio * w_base + b * y_base

    return s_original, s_con_pca


def simular_carroll(params, pca_value, y_base, r_base):
    """
    Modelo de crecimiento y ahorro: S = f(Y,r)
    Con PCA: S = f(Y) + r(1 + φ·PCA)

    Args:
        params: Parámetros del modelo
        pca_value: Valor PCA
        y_base: Ingreso base
        r_base: Tasa de interés base

    Returns:
        tuple: (ahorro_original, ahorro_con_pca)
    """
    base_saving_rate = params['base_saving_rate']
    phi = params['phi']

    # Componente base del ahorro
    ahorro_base = base_saving_rate * y_base

    # Modelo original
    s_original = ahorro_base + r_base * y_base

    # Modelo con PCA - ajuste en sensibilidad a tasa de interés
    elasticidad_ajustada = r_base * (1 + phi * pca_value)
    s_con_pca = ahorro_base + elasticidad_ajustada * y_base

    return s_original, s_con_pca


def simular_deaton(params, pca_value, y_base):
    """
    Modelo de expectativas de consumo: S = f(Y,expectations)
    Con PCA: S = f(Y,expectations·(1 + κ·PCA))

    Args:
        params: Parámetros del modelo
        pca_value: Valor PCA
        y_base: Ingreso base

    Returns:
        tuple: (ahorro_original, ahorro_con_pca)
    """
    base_rate = params['base_rate']
    kappa = params['kappa']

    # Modelo original
    s_original = base_rate * y_base

    # Modelo con PCA - ajuste en formación de expectativas
    factor_expectativas = 1 + kappa * pca_value
    s_con_pca = base_rate * y_base * factor_expectativas

    return s_original, s_con_pca


def calcular_impacto_economico(resultados_modelos):
    """
    Calcula el impacto económico agregado de los sesgos conductuales

    Args:
        resultados_modelos: Resultados de simulaciones de modelos

    Returns:
        dict: Estadísticas de impacto económico
    """
    impactos = {}

    for modelo_key, datos in resultados_modelos.items():
        original_values = np.array(datos['original'])
        pca_values = np.array(datos['con_pca'])

        # Calcular impactos
        diferencias_absolutas = pca_values - original_values
        diferencias_relativas = diferencias_absolutas / original_values * 100

        impactos[modelo_key] = {
            'impacto_promedio_absoluto': np.mean(diferencias_absolutas),
            'impacto_promedio_relativo': np.mean(diferencias_relativas),
            'impacto_std_absoluto': np.std(diferencias_absolutas),
            'impacto_std_relativo': np.std(diferencias_relativas),
            'impacto_maximo': np.max(diferencias_absolutas),
            'impacto_minimo': np.min(diferencias_absolutas),
            'direccion_predominante': 'Positivo' if np.mean(diferencias_absolutas) > 0 else 'Negativo'
        }

    return impactos
