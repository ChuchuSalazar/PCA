"""
Utilidades para exportación de resultados Bootstrap a Excel
"""

import pandas as pd
import numpy as np
import io
from datetime import datetime
from config.constants import MODELOS_EXTERNOS, MODELOS_COEFICIENTES
from utils.statistics import calcular_estadisticas_avanzadas


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
        crear_hoja_configuracion(writer, parametros)

        # 2. Resultados Bootstrap por escenario
        for escenario, resultados in resultados_dict.items():
            crear_hoja_bootstrap_escenario(writer, escenario, resultados)
            crear_hoja_estadisticas_escenario(writer, escenario, resultados)

        # 3. Comparaciones Bootstrap
        if len(resultados_dict) > 1:
            crear_hoja_comparaciones(writer, resultados_dict)

        # 4. Modelos Económicos
        crear_hoja_modelos_economicos(writer, resultados_dict)

    output.seek(0)
    return output, filename


def crear_hoja_configuracion(writer, parametros):
    """Crea hoja de configuración del análisis"""

    config_data = [
        ['Parameter', 'Value'],
        ['Analysis Group', parametros.get('grupo', 'N/A')],
        ['Economic Scenario', parametros.get('escenario', 'N/A')],
        ['Bootstrap Iterations', parametros.get('n_bootstrap', 'N/A')],
        ['Methodology', 'Bootstrap Resampling'],
        ['Timestamp', parametros.get('timestamp', 'N/A')],
        ['Multi-Scenario Analysis',
            parametros.get('analisis_comparativo', False)]
    ]

    config_df = pd.DataFrame(config_data[1:], columns=config_data[0])
    config_df.to_excel(writer, sheet_name='Bootstrap_Config', index=False)


def crear_hoja_bootstrap_escenario(writer, escenario, resultados):
    """Crea hoja con datos bootstrap completos por escenario"""

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


def crear_hoja_estadisticas_escenario(writer, escenario, resultados):
    """Crea hoja con estadísticas bootstrap del escenario"""

    if 'bootstrap_stats' not in resultados:
        return

    stats = resultados['bootstrap_stats']
    stats_data = pd.DataFrame([
        ['Bootstrap Iterations', stats.get('bootstrap_n', 'N/A')],
        ['Original Sample Size', stats.get('original_n', 'N/A')],
        ['PCA Bootstrap Mean', stats.get('pca_mean', 'N/A')],
        ['PCA Bootstrap Std', stats.get('pca_std', 'N/A')],
        ['PCA CI Lower (2.5%)', stats.get('pca_ci_lower', 'N/A')],
        ['PCA CI Upper (97.5%)', stats.get('pca_ci_upper', 'N/A')],
        ['Bias-Corrected Mean', stats.get('bias_corrected_mean', 'N/A')],
        ['Original Mean', stats.get('original_mean', 'N/A')],
        ['Bootstrap SE', stats.get('bootstrap_se', 'N/A')]
    ], columns=['Statistic', 'Value'])

    stats_sheet = f'Stats_{escenario.title()}'
    stats_data.to_excel(writer, sheet_name=stats_sheet, index=False)


def crear_hoja_comparaciones(writer, resultados_dict):
    """Crea hoja comparativa entre escenarios"""

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


def crear_hoja_modelos_economicos(writer, resultados_dict):
    """Crea hoja con resultados de modelos económicos"""

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
