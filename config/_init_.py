"""
Módulo de configuración del simulador PCA
"""

from .constants import (
    MODELOS_COEFICIENTES,
    ESCENARIOS_ECONOMICOS,
    MODELOS_EXTERNOS,
    MODELOS_EXTERNOS_INFO
)

from .styles import apply_custom_styles, get_header_html

__all__ = [
    'MODELOS_COEFICIENTES',
    'ESCENARIOS_ECONOMICOS',
    'MODELOS_EXTERNOS',
    'MODELOS_EXTERNOS_INFO',
    'apply_custom_styles',
    'get_header_html'
]
