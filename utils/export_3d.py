"""
Utilidades para exportación de resultados 3D y análisis completo
"""

import pandas as pd
import numpy as np
import io
import plotly.io as pio
from datetime import datetime
from config.constants import ESCENARIOS_ECONOMICOS


def crear_excel_3d_completo(resultados_dict, tipo_analisis, fig_3d=None):
    """Crea Excel con todos los datos 3D y análisis relacionado"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"PCA_ANALYSIS_3D_{tipo_analisis}_{timestamp}.xlsx"

    output = io.BytesIO()

    # Exportar datos 3D completos
    datos_3d_df = exportar_datos_3d_completos(resultados_dict, tipo_analisis)

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formato
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#2c3e50', 'font_color': 'white', 'border': 1
        })

        # Hoja principal con datos 3D
        datos_3d_df.to_excel(
            writer, sheet_name='Datos_3D_Completos', index=False)

        # Estadísticas por escenario
        crear_hoja_estadisticas_3d(writer, resultados_dict)

        # Matriz de correlaciones 3D
        crear_hoja_correlaciones_3d(writer, resultados_dict)

        # Análisis de dispersión por niveles
        crear_hoja_analisis_niveles(writer, resultados_dict)

        # Metadatos
        crear_hoja_metadatos_3d(writer, tipo_analisis, resultados_dict)

        # Si hay gráfico 3D, generar reporte HTML
        if fig_3d:
            generar_reporte_html_3d(fig_3d, tipo_analisis, timestamp)

    output.seek(0)
    return output, filename


def exportar_datos_3d_completos(resultados_dict, tipo_analisis):
    """Exporta todos los datos utilizados en visualizaciones 3D"""

    datos_3d = []

    for escenario, resultados in resultados_dict.items():
        n_samples = len(resultados['pca_values'])

        pca = np.array(resultados['pca_values'])
        pse = np.array(resultados['pse_values'])
        dh = np.array(resultados['variables_cognitivas']['DH'])
        cs = np.array(resultados['variables_cognitivas']['CS'])
        av = np.array(resultados['variables_cognitivas']['AV'])
        sq = np.array(resultados['variables_cognitivas']['SQ'])

        # Calcular índice de sesgos combinados
        indice_sesgos = (0.3 * np.abs(sq) + 0.25 * np.abs(dh) +
                         0.25 * np.abs(cs) + 0.2 * np.abs(av))

        # Clasificar niveles
        niveles_sesgos = np.where(indice_sesgos < np.percentile(indice_sesgos, 33), 'Bajo',
                                  np.where(indice_sesgos < np.percentile(indice_sesgos, 67), 'Medio', 'Alto'))

        sq_abs = np.abs(sq)
        niveles_sq = np.where(sq_abs < np.percentile(sq_abs, 33), 'SQ_Bajo',
                              np.where(sq_abs < np.percentile(sq_abs, 67), 'SQ_Medio', 'SQ_Alto'))

        # Simular ítems PCA para análisis de consistencia
        np.random.seed(42)
        pca2 = pca * 0.6 + np.random.normal(0, 0.4, n_samples)
        pca4 = pca * 0.8 + np.random.normal(0, 0.3, n_samples)
        pca5 = pca * 0.85 + np.random.normal(0, 0.2, n_samples)

        # Factores del escenario
        config_escenario = ESCENARIOS_ECONOMICOS[escenario]

        for i in range(n_samples):
            datos_3d.append({
                'Bootstrap_ID': i + 1,
                'Escenario': escenario,
                'Escenario_Nombre': config_escenario['nombre'],
                'Factor_DH': config_escenario['factor_dh'],
                'Factor_CS': config_escenario['factor_cs'],
                'Factor_AV': config_escenario['factor_av'],
                'Factor_SQ': config_escenario['factor_sq'],
                'Volatilidad': config_escenario['volatilidad'],
                'PCA': pca[i],
                'PSE': pse[i],
                'DH': dh[i],
                'CS': cs[i],
                'AV': av[i],
                'SQ': sq[i],
                'Indice_Sesgos_Combinados': indice_sesgos[i],
                'Nivel_Sesgos': niveles_sesgos[i],
                'Nivel_SQ': niveles_sq[i],
                'PCA2_Simulado': pca2[i],
                'PCA4_Simulado': pca4[i],
                'PCA5_Simulado': pca5[i],
                'Tipo_Analisis_3D': tipo_analisis,
                'Coordenada_X_3D': pse[i] if tipo_analisis == 'pca_pse_sesgos' else pca2[i],
                'Coordenada_Y_3D': indice_sesgos[i] if tipo_analisis == 'pca_pse_sesgos' else pca4[i],
                'Coordenada_Z_3D': pca[i],
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    return pd.DataFrame(datos_3d)


def crear_hoja_estadisticas_3d(writer, resultados_dict):
    """Crea hoja con estadísticas 3D por escenario"""

    stats_3d = []

    for escenario, resultados in resultados_dict.items():
        if 'bootstrap_stats' in resultados:
            stats = resultados['bootstrap_stats']
            config_escenario = ESCENARIOS_ECONOMICOS[escenario]

            # Calcular estadísticas adicionales para análisis 3D
            pca = np.array(resultados['pca_values'])
            pse = np.array(resultados['pse_values'])

            # Correlación PCA-PSE
            corr_pca_pse = np.corrcoef(pca, pse)[0, 1]

            # Dispersión 3D (usando desviación estándar multidimensional)
            variables_3d = np.column_stack([
                pse, pca,
                resultados['variables_cognitivas']['DH'],
                resultados['variables_cognitivas']['SQ']
            ])
            dispersion_3d = np.mean(np.std(variables_3d, axis=0))

            stats_3d.append({
                'Escenario': escenario,
                'Escenario_Nombre': config_escenario['nombre'],
                'Bootstrap_N': stats.get('bootstrap_n', 0),
                'PCA_Mean': stats.get('pca_mean', 0),
                'PCA_Std': stats.get('pca_std', 0),
                'PCA_CI_Width': stats.get('pca_ci_upper', 0) - stats.get('pca_ci_lower', 0),
                'PSE_Mean': np.mean(pse),
                'PSE_Std': np.std(pse),
                'Correlacion_PCA_PSE': corr_pca_pse,
                'Dispersion_3D_Promedio': dispersion_3d,
                'Factor_DH_Aplicado': config_escenario['factor_dh'],
                'Factor_CS_Aplicado': config_escenario['factor_cs'],
                'Factor_SQ_Aplicado': config_escenario['factor_sq'],
                'Volatilidad_Escenario': config_escenario['volatilidad']
            })

    if stats_3d:
        stats_df = pd.DataFrame(stats_3d)
        stats_df.to_excel(
            writer, sheet_name='Estadisticas_3D_Escenarios', index=False)


def crear_hoja_correlaciones_3d(writer, resultados_dict):
    """Crea hoja con matriz de correlaciones 3D"""

    correlaciones_3d = []

    for escenario, resultados in resultados_dict.items():
        data_matrix = np.column_stack([
            resultados['pca_values'],
            resultados['pse_values'],
            resultados['variables_cognitivas']['DH'],
            resultados['variables_cognitivas']['CS'],
            resultados['variables_cognitivas']['AV'],
            resultados['variables_cognitivas']['SQ']
        ])

        variables = ['PCA', 'PSE', 'DH', 'CS', 'AV', 'SQ']
        corr_matrix = np.corrcoef(data_matrix.T)

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                correlaciones_3d.append({
                    'Escenario': escenario,
                    'Variable_1': var1,
                    'Variable_2': var2,
                    'Correlacion': corr_matrix[i, j],
                    'Correlacion_Abs': abs(corr_matrix[i, j]),
                    'Es_Diagonal': i == j,
                    'Intensidad': 'Alta' if abs(corr_matrix[i, j]) > 0.5 else 'Media' if abs(corr_matrix[i, j]) > 0.3 else 'Baja'
                })

    corr_df = pd.DataFrame(correlaciones_3d)
    corr_df.to_excel(writer, sheet_name='Correlaciones_3D', index=False)


def crear_hoja_analisis_niveles(writer, resultados_dict):
    """Crea hoja con análisis por niveles de sesgos"""

    niveles_analysis = []

    for escenario, resultados in resultados_dict.items():
        # Análisis por niveles de Status Quo
        sq = np.array(resultados['variables_cognitivas']['SQ'])
        pca = np.array(resultados['pca_values'])

        sq_abs = np.abs(sq)
        niveles_sq = np.where(sq_abs < np.percentile(sq_abs, 33), 'SQ_Bajo',
                              np.where(sq_abs < np.percentile(sq_abs, 67), 'SQ_Medio', 'SQ_Alto'))

        for nivel in ['SQ_Bajo', 'SQ_Medio', 'SQ_Alto']:
            mask = niveles_sq == nivel
            if np.any(mask):
                pca_nivel = pca[mask]
                sq_nivel = sq[mask]

                niveles_analysis.append({
                    'Escenario': escenario,
                    'Nivel_SQ': nivel,
                    'N_Observaciones': len(pca_nivel),
                    'Porcentaje_Muestra': (len(pca_nivel) / len(pca)) * 100,
                    'PCA_Mean': np.mean(pca_nivel),
                    'PCA_Std': np.std(pca_nivel),
                    'PCA_Min': np.min(pca_nivel),
                    'PCA_Max': np.max(pca_nivel),
                    'SQ_Mean': np.mean(sq_nivel),
                    'SQ_Std': np.std(sq_nivel),
                    'Variabilidad_PCA': np.std(pca_nivel) / np.mean(pca_nivel) if np.mean(pca_nivel) != 0 else 0
                })

    if niveles_analysis:
        niveles_df = pd.DataFrame(niveles_analysis)
        niveles_df.to_excel(
            writer, sheet_name='Analisis_Niveles_Sesgos', index=False)


def crear_hoja_metadatos_3d(writer, tipo_analisis, resultados_dict):
    """Crea hoja con metadatos del análisis 3D"""

    interpretaciones_3d = {
        'pca_pse_sesgos': 'Dispersión PCA en función de PSE e índice combinado de sesgos cognitivos',
        'pca_items': 'Análisis de consistencia interna usando ítems simulados PCA2, PCA4, PCA5',
        'pca_pse_sq': 'Análisis específico del sesgo Status Quo como variable dominante',
        'sesgos_combinados': 'Espacio tridimensional de interacción entre sesgos cognitivos',
        'comparativo_escenarios': 'Comparación multi-escenario con superficies de tendencia'
    }

    metadatos = pd.DataFrame([
        ['Tipo_Analisis_3D', tipo_analisis],
        ['Interpretacion_3D', interpretaciones_3d.get(
            tipo_analisis, 'Análisis 3D general')],
        ['Fecha_Generacion', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ['Escenarios_Incluidos', ', '.join(resultados_dict.keys())],
        ['Numero_Escenarios', len(resultados_dict)],
        ['Metodologia_Base', 'Bootstrap Resampling + PLS-SEM'],
        ['Framework_Teorico',
            'DH (Descuento Hiperbólico), CS (Contagio Social), AV (Aversión Pérdidas), SQ (Status Quo)'],
        ['Variables_Principales_3D', 'PCA, PSE, DH, CS, AV, SQ'],
        ['Dimensiones_Analisis', '3D + Color/Tamaño como 4ta dimensión'],
        ['Software', 'Python + Streamlit + Plotly'],
        ['Autor', 'MSc. Jesús Fernando Salazar Rojas'],
        ['Institucion', 'UCAB - Doctorado en Economía'],
        ['Version', 'PCA Simulator v3.2'],
        ['Copyright', '© 2025 - Propensión Conductual al Ahorro']
    ], columns=['Parametro', 'Valor'])

    metadatos.to_excel(writer, sheet_name='Metadatos_3D', index=False)


def generar_reporte_html_3d(fig_3d, tipo_analisis, timestamp):
    """Genera reporte HTML interactivo con gráfico 3D"""

    fig_html = pio.to_html(fig_3d, include_plotlyjs='cdn', div_id="grafico3d")

    interpretaciones = {
        'pca_pse_sesgos': 'Análisis tridimensional de la dispersión PCA considerando expectativas racionales (PSE) y sesgos cognitivos combinados',
        'pca_items': 'Validación de consistencia interna mediante análisis multidimensional de ítems del constructo PCA',
        'pca_pse_sq': 'Exploración específica del impacto del sesgo Status Quo como variable cognitiva dominante',
        'sesgos_combinados': 'Visualización del espacio de interacción entre los tres sesgos cognitivos principales',
        'comparativo_escenarios': 'Análisis comparativo multi-escenario con identificación de patrones de dispersión contextual'
    }

    html_completo = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Análisis 3D PCA - {tipo_analisis.title()} | Reporte Interactivo</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; padding: 20px; background: #f8f9fa; 
            }}
            .header {{ 
                background: linear-gradient(135deg, #2c3e50, #34495e); 
                color: white; padding: 30px; border-radius: 15px; 
                text-align: center; margin-bottom: 30px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            .content {{
                background: white; padding: 25px; border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px;
            }}
            .interpretacion {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white; padding: 20px; border-radius: 10px;
                margin: 20px 0; line-height: 1.6;
            }}
            .footer {{ 
                text-align: center; color: #666; font-size: 0.9em; 
                margin-top: 30px; padding: 15px;
                border-top: 2px solid #dee2e6;
            }}
            #grafico3d {{ margin: 20px 0; }}
            .nota-metodologica {{
                background: #e8f4f8; border-left: 4px solid #3498db;
                padding: 15px; margin: 20px 0; font-size: 0.95em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Análisis de Dispersión 3D - PCA Bootstrap</h1>
            <p style="font-size: 1.2em; margin: 10px 0;">
                {tipo_analisis.replace('_', ' ').title()} | Simulador PCA v3.2
            </p>
            <p style="opacity: 0.9;">Análisis Conductual del Ahorro - Metodología Bootstrap</p>
        </div>
        
        <div class="content">
            <h2>🎯 Interpretación del Análisis</h2>
            <div class="interpretacion">
                {interpretaciones.get(tipo_analisis, 'Análisis 3D especializado de propensión conductual al ahorro')}
            </div>
            
            <div class="nota-metodologica">
                <strong>📋 Metadatos del Análisis:</strong><br>
                • <strong>Fecha de generación:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
                • <strong>Tipo de análisis:</strong> {tipo_analisis}<br>
                • <strong>Metodología:</strong> Bootstrap Resampling + PLS-SEM<br>
                • <strong>Iteraciones:</strong> 3,000 por escenario<br>
                • <strong>Framework teórico:</strong> DH • CS • AV • SQ Analysis
            </div>
        </div>
        
        <div class="content">
            <h2>📈 Visualización Interactiva 3D</h2>
            {fig_html}
        </div>
        
        <div class="footer">
            <strong>Simulador PCA - Tesis Doctoral</strong><br>
            MSc. Jesús Fernando Salazar Rojas | UCAB - Economía Conductual 2025<br>
            La Propensión Conductual al Ahorro: Un estudio desde los sesgos cognitivos<br>
            <em>Generado el {timestamp} | © 2025 - Todos los derechos reservados</em>
        </div>
    </body>
    </html>
    """

    # Guardar archivo HTML
    filename_html = f"PCA_3D_Report_{tipo_analisis}_{timestamp}.html"
    try:
        with open(filename_html, 'w', encoding='utf-8') as f:
            f.write(html_completo)
        print(f"Reporte HTML generado: {filename_html}")
    except Exception as e:
        print(f"Error generando HTML: {e}")

    return html_completo
