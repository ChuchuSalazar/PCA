"""
Funciones estadísticas avanzadas para análisis Bootstrap y PCA
"""

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def calcular_estadisticas_avanzadas(datos):
    """
    Calcula estadísticas descriptivas avanzadas

    Args:
        datos: Array o lista de datos

    Returns:
        dict: Estadísticas descriptivas completas
    """
    datos_array = np.array(datos)

    return {
        'media': np.mean(datos_array),
        'std': np.std(datos_array),
        'min': np.min(datos_array),
        'max': np.max(datos_array),
        'p5': np.percentile(datos_array, 5),
        'p25': np.percentile(datos_array, 25),
        'mediana': np.percentile(datos_array, 50),
        'p75': np.percentile(datos_array, 75),
        'p95': np.percentile(datos_array, 95),
        'asimetria': sp_stats.skew(datos_array),
        'curtosis': sp_stats.kurtosis(datos_array),
        'cv': np.std(datos_array) / np.mean(datos_array) if np.mean(datos_array) != 0 else 0,
        'iqr': np.percentile(datos_array, 75) - np.percentile(datos_array, 25),
        'rango': np.max(datos_array) - np.min(datos_array),
        'varianza': np.var(datos_array),
        'n_observaciones': len(datos_array)
    }


def calcular_intervalos_confianza_bootstrap(datos, alpha=0.05):
    """
    Calcula intervalos de confianza bootstrap

    Args:
        datos: Datos bootstrap
        alpha: Nivel de significancia (default 0.05 para 95% CI)

    Returns:
        dict: Intervalos de confianza
    """
    datos_array = np.array(datos)

    # Percentiles para intervalos de confianza
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        'ci_lower': np.percentile(datos_array, lower_percentile),
        'ci_upper': np.percentile(datos_array, upper_percentile),
        'ci_width': np.percentile(datos_array, upper_percentile) - np.percentile(datos_array, lower_percentile),
        'alpha': alpha,
        'confidence_level': (1 - alpha) * 100
    }


def correccion_sesgo_bootstrap(bootstrap_estimates, original_estimate):
    """
    Aplica corrección de sesgo bootstrap

    Args:
        bootstrap_estimates: Estimaciones bootstrap
        original_estimate: Estimación original

    Returns:
        dict: Estimaciones corregidas por sesgo
    """
    bootstrap_mean = np.mean(bootstrap_estimates)
    bias = bootstrap_mean - original_estimate
    bias_corrected = original_estimate - bias

    return {
        'original_estimate': original_estimate,
        'bootstrap_mean': bootstrap_mean,
        'estimated_bias': bias,
        'bias_corrected_estimate': bias_corrected,
        'bias_percentage': (bias / original_estimate) * 100 if original_estimate != 0 else 0
    }


def test_normalidad_bootstrap(datos):
    """
    Realiza tests de normalidad en datos bootstrap

    Args:
        datos: Datos a testear

    Returns:
        dict: Resultados de tests de normalidad
    """
    datos_array = np.array(datos)

    # Limitar muestra para Shapiro-Wilk
    sample_size = min(5000, len(datos_array))
    sample_data = np.random.choice(datos_array, sample_size, replace=False)

    # Test de Shapiro-Wilk
    shapiro_stat, shapiro_p = sp_stats.shapiro(sample_data)

    # Test de Kolmogorov-Smirnov
    ks_stat, ks_p = sp_stats.kstest(
        datos_array, 'norm',
        args=(np.mean(datos_array), np.std(datos_array))
    )

    # Test de Anderson-Darling
    ad_stat, ad_critical, ad_significance = sp_stats.anderson(
        datos_array, dist='norm')

    return {
        'shapiro_wilk': {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05,
            'sample_size': sample_size
        },
        'kolmogorov_smirnov': {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': ks_p > 0.05
        },
        'anderson_darling': {
            'statistic': ad_stat,
            'critical_values': ad_critical,
            'significance_levels': ad_significance,
            'is_normal': ad_stat < ad_critical[2]  # 5% level
        }
    }


def analisis_convergencia_bootstrap(datos, window_size=100):
    """
    Analiza la convergencia del proceso bootstrap

    Args:
        datos: Secuencia de estimaciones bootstrap
        window_size: Tamaño de ventana para análisis

    Returns:
        dict: Métricas de convergencia
    """
    datos_array = np.array(datos)
    n = len(datos_array)

    if n < window_size:
        window_size = max(10, n // 4)

    # Medias acumulativas
    cumulative_means = np.cumsum(datos_array) / np.arange(1, n + 1)

    # Estabilidad de medias móviles
    moving_means = pd.Series(datos_array).rolling(window=window_size).mean()
    moving_stds = pd.Series(datos_array).rolling(window=window_size).std()

    # Métricas de convergencia
    final_stability = np.std(moving_means.dropna().tail(n // 4))
    convergence_ratio = final_stability / \
        np.std(datos_array) if np.std(datos_array) > 0 else 0

    return {
        'cumulative_means': cumulative_means,
        'moving_means': moving_means.values,
        'moving_stds': moving_stds.values,
        'final_mean': cumulative_means[-1],
        'final_stability': final_stability,
        'convergence_ratio': convergence_ratio,
        'converged': convergence_ratio < 0.05,
        'window_size': window_size
    }


def comparacion_escenarios_bootstrap(resultados_dict):
    """
    Compara resultados bootstrap entre diferentes escenarios

    Args:
        resultados_dict: Diccionario con resultados por escenario

    Returns:
        dict: Comparaciones estadísticas entre escenarios
    """
    comparaciones = {}
    escenarios = list(resultados_dict.keys())

    for i, escenario1 in enumerate(escenarios):
        for j, escenario2 in enumerate(escenarios[i+1:], i+1):

            datos1 = resultados_dict[escenario1]['pca_values']
            datos2 = resultados_dict[escenario2]['pca_values']

            # Tests estadísticos
            t_stat, t_p = sp_stats.ttest_ind(datos1, datos2)
            u_stat, u_p = sp_stats.mannwhitneyu(
                datos1, datos2, alternative='two-sided')

            # Diferencias en estadísticas descriptivas
            stats1 = calcular_estadisticas_avanzadas(datos1)
            stats2 = calcular_estadisticas_avanzadas(datos2)

            key = f"{escenario1}_vs_{escenario2}"
            comparaciones[key] = {
                'escenarios': [escenario1, escenario2],
                'diferencia_medias': stats2['media'] - stats1['media'],
                'diferencia_std': stats2['std'] - stats1['std'],
                'test_t': {
                    'statistic': t_stat,
                    'p_value': t_p,
                    'significativo': t_p < 0.05
                },
                'test_mann_whitney': {
                    'statistic': u_stat,
                    'p_value': u_p,
                    'significativo': u_p < 0.05
                },
                'efecto_size_cohen_d': (stats2['media'] - stats1['media']) / np.sqrt((stats1['varianza'] + stats2['varianza']) / 2),
                'interpretacion': interpretar_diferencia_escenarios(stats1, stats2)
            }

    return comparaciones


def interpretar_diferencia_escenarios(stats1, stats2):
    """Interpreta las diferencias entre escenarios"""
    diff_rel = abs(stats2['media'] - stats1['media']) / \
        abs(stats1['media']) * 100 if stats1['media'] != 0 else 0

    if diff_rel < 5:
        return "Diferencia mínima"
    elif diff_rel < 15:
        return "Diferencia moderada"
    elif diff_rel < 30:
        return "Diferencia sustancial"
    else:
        return "Diferencia muy significativa"


def calcular_poder_estadistico(n_bootstrap, efecto_esperado, alpha=0.05):
    """
    Calcula el poder estadístico del análisis bootstrap

    Args:
        n_bootstrap: Tamaño de muestra bootstrap
        efecto_esperado: Tamaño del efecto esperado
        alpha: Nivel de significancia

    Returns:
        dict: Métricas de poder estadístico
    """
    # Cálculo aproximado del poder usando distribución t
    from scipy.stats import t

    df = n_bootstrap - 1
    t_critical = t.ppf(1 - alpha/2, df)

    # Poder estadístico aproximado
    poder = 1 - t.cdf(t_critical - efecto_esperado * np.sqrt(n_bootstrap), df)

    return {
        'poder_estadistico': poder,
        'n_bootstrap': n_bootstrap,
        'efecto_esperado': efecto_esperado,
        'alpha': alpha,
        'poder_adecuado': poder >= 0.8
    }
