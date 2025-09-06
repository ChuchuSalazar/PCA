"""
Configuraciones y constantes para el simulador PCA
Contiene todos los modelos, coeficientes y escenarios económicos
"""
# MODELOS ECONÓMICOS EXTERNOS
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
        'descripcion': 'Growth and saving model, Carroll & Weil (1994)',
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

# MODELOS PLS-SEM CON COEFICIENTES Y ESTADÍSTICAS
MODELOS_COEFICIENTES = {
    'Hah': {
        'ecuacion': 'PCA = 0.3777·PSE + 0.2226·DH - 0.5947·SQ + 0.2866·CS',
        'coef': {'PSE': 0.3777, 'DH': 0.2226, 'SQ': -0.5947, 'CS': 0.2866},
        'r2': -0.639818,
        'rmse': 1.270350,
        'mae': 1.043280,
        'correlation': -0.129289,
        'grupo_stats': {
            'PSE_mean': 0.125, 'DH_mean': 0.089, 'CS_mean': 0.156,
            'AV_mean': 0.098, 'SQ_mean': -0.067, 'PCA_mean': -0.088
        },
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
        },
        'sesgo_dominante': 'SQ',
        'interpretacion_sesgos': {
            'DH': 'Descuento Hiperbólico moderado - Los hombres muestran tendencia a valorar más el presente',
            'CS': 'Contagio Social moderado - Influencia media de pares en decisiones de ahorro',
            'SQ': 'Status Quo muy fuerte - Resistencia alta al cambio en patrones de ahorro',
            'AV': 'Aversión a Pérdidas presente - Consideración moderada del riesgo'
        }
    },
    'Mah': {
        'ecuacion': 'PCA = 0.3485·PSE - 0.2013·DH - 0.5101·SQ + 0.3676·CS',
        'coef': {'PSE': 0.3485, 'DH': -0.2013, 'SQ': -0.5101, 'CS': 0.3676},
        'r2': 0.571136,
        'rmse': 0.650872,
        'mae': 0.519483,
        'correlation': 0.759797,
        'grupo_stats': {
            'PSE_mean': -0.125, 'DH_mean': -0.089, 'CS_mean': -0.156,
            'AV_mean': -0.098, 'SQ_mean': 0.067, 'PCA_mean': 0.088
        },
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
        },
        'sesgo_dominante': 'SQ',
        'interpretacion_sesgos': {
            'DH': 'Descuento Hiperbólico negativo - Las mujeres consideran más el futuro',
            'CS': 'Contagio Social alto - Mayor influencia social en decisiones de ahorro',
            'SQ': 'Status Quo fuerte - Resistencia significativa al cambio, pero menor que hombres',
            'AV': 'Aversión a Pérdidas moderada - Evaluación equilibrada del riesgo'
        }
    }
}

# ESCENARIOS ECONÓMICOS CON FACTORES DE AJUSTE
# ESCENARIOS ECONÓMICOS CON FACTORES ACTUALIZADOS
ESCENARIOS_ECONOMICOS = {
    'baseline': {
        'nombre': 'Baseline Scenario',
        'descripcion': 'Condiciones económicas normales sin alteraciones externas',
        'color': '#34495e',
        'factor_dh': 1.0,   # Descuento Hiperbólico
        'factor_cs': 1.0,   # Contagio Social (Efecto Manada)
        'factor_av': 1.0,   # Aversión a las Pérdidas
        'factor_sq': 1.0,   # Status Quo
        'volatilidad': 1.0,
        'bootstrap_noise': 0.1
    },
    'crisis': {
        'nombre': 'Economic Crisis',
        'descripcion': 'Entorno de incertidumbre, rumores negativos, sesgos cognitivos intensificados',
        'color': '#e74c3c',
        'factor_dh': 1.6,   # Se intensifica significativamente
        'factor_cs': 1.5,   # Pánico de venta, efecto manada extremo
        'factor_av': 1.7,   # Hipersensibilidad a pérdidas
        'factor_sq': 1.3,   # Renuencia extrema al cambio
        'volatilidad': 1.8,  # Mayor volatilidad por decisiones impulsivas
        'bootstrap_noise': 0.2
    },
    'bonanza': {
        'nombre': 'Economic Bonanza',
        'descripcion': 'Entorno optimista, confianza económica, sesgos presentes pero con diferente motivación',
        'color': '#27ae60',
        'factor_dh': 1.2,   # Gratificación inmediata, menor planificación
        'factor_cs': 1.4,   # Inversión en activos populares, burbujas
        'factor_av': 1.3,   # Cautela excesiva, pérdida de oportunidades
        'factor_sq': 1.2,   # Apego a lo conocido, pérdida de innovación
        'volatilidad': 1.3,  # Volatilidad por sobrevaloración de activos
        'bootstrap_noise': 0.12
    }
}


# INFORMACIÓN DETALLADA DE MODELOS EXTERNOS
MODELOS_EXTERNOS_INFO = {
    'keynes': {
        'variable_dependiente': 'Ahorro nacional bruto (S)',
        'variables_independientes': 'Económica: Producto nacional bruto (Y)',
        'tipologia': 'Económica',
        'modificacion': 'La PMA se ajusta según los sesgos conductuales; refleja cómo la inclinación a ahorrar cambia por heurísticas como aversión a la pérdida o exceso de confianza.',
        'justificacion_teorica': 'Los sesgos cognitivos alteran la propensión marginal al ahorro, haciendo que la relación entre ingreso y ahorro no sea lineal ni constante.',
        'impacto_esperado': 'Moderado a Alto'
    },
    'friedman': {
        'variable_dependiente': 'Consumo y ahorro',
        'variables_independientes': 'Económicas: ingreso permanente, ingreso transitorio',
        'tipologia': 'Económica',
        'modificacion': 'El ingreso permanente percibido Yp se ajusta; sesgos influyen en la estimación de recursos futuros y, por tanto, en el ahorro actual.',
        'justificacion_teorica': 'La percepción del ingreso permanente está sesgada por heurísticas de disponibilidad y representatividad, afectando las decisiones intertemporales.',
        'impacto_esperado': 'Alto'
    },
    'modigliani': {
        'variable_dependiente': 'Ahorro individual / riqueza',
        'variables_independientes': 'Económicas: ingreso esperado futuro, edad, riqueza actual',
        'tipologia': 'Económica-Demográfica',
        'modificacion': 'La sensibilidad al patrimonio W cambia según la PCA; refleja cómo la percepción de riqueza y su efecto en la PMA se modifica por sesgos cognitivos.',
        'justificacion_teorica': 'Los sesgos afectan la planificación del ciclo de vida, alterando la percepción de necesidades futuras y la valoración del patrimonio actual.',
        'impacto_esperado': 'Moderado'
    },
    'carroll': {
        'variable_dependiente': 'S = medida del ahorro',
        'variables_independientes': 'g = componente predecible del crecimiento del ingreso, Q = otras variables, h₀ = constante',
        'tipologia': 'Económica-Financiera',
        'modificacion': 'La elasticidad respecto a la tasa de interés r se ajusta; los sesgos afectan la reacción del ahorro frente a incentivos financieros.',
        'justificacion_teorica': 'Los sesgos de descuento temporal y aversión al riesgo modifican la sensibilidad a las tasas de interés, alterando los patrones de ahorro.',
        'impacto_esperado': 'Alto'
    },
    'deaton': {
        'variable_dependiente': 'Tasa de ahorro personal (Sₜ)',
        'variables_independientes': 'g = tasa de crecimiento del ingreso, w* = riqueza-ingreso objetivo',
        'tipologia': 'Económica-Expectacional',
        'modificacion': 'Las expectativas de consumo futuro se ajustan; la PCA refleja cómo sesgos cambian la asignación actual al ahorro.',
        'justificacion_teorica': 'La formación de expectativas está influenciada por sesgos de confirmación y anclaje, afectando las decisiones de ahorro presentes.',
        'impacto_esperado': 'Moderado a Alto'
    }
}

# CONFIGURACIONES GENERALES
DEFAULT_BOOTSTRAP_ITERATIONS = 3000
MIN_BOOTSTRAP_ITERATIONS = 1000
MAX_BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_STEP = 500

# CONFIGURACIONES DE EXPORTACIÓN
EXCEL_ENGINE = 'xlsxwriter'
DATE_FORMAT = "%Y%m%d_%H%M"
