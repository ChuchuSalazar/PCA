"""
Estilos CSS personalizados para la aplicación PCA Simulator
"""

import streamlit as st


def apply_custom_styles():
    """Aplica todos los estilos CSS personalizados"""
    st.markdown(get_css_styles(), unsafe_allow_html=True)


def get_css_styles():
    """Retorna todos los estilos CSS como string"""
    return """
    <style>
        .main-header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .scenario-card {
            background: linear-gradient(45deg, #34495e 0%, #2c3e50 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            border-left: 4px solid #3498db;
        }
        
        .metric-card {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(52, 152, 219, 0.3);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .profile-card-enhanced {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .cognitive-stats-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 6px 18px rgba(240, 147, 251, 0.3);
        }
        
        .download-section {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
            text-align: center;
        }
        
        .model-images {
            background: rgba(255,255,255,0.95);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 2px solid #3498db;
        }
        
        .external-models-info {
            background: rgba(248, 249, 250, 0.95);
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .bias-interpretation-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            margin: 1rem 0;
        }
        
        .profile-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: #f9f9f9;
            padding: 5px 0;
            text-align: center;
            color: #666;
            font-size: 0.85em;
            border-top: 1px solid #ddd;
            line-height: 1.2;
        }
        
        .footer p {
            margin: 2px 0;
        }
    </style>
    """


def get_header_html():
    """Retorna el HTML del header principal"""
    return """
    <div class="main-header">
        <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700; letter-spacing: 2px;'>
            PCA SIMULATOR v3.2
        </h1>
        <h2 style='margin: 1rem 0; font-size: 1.5rem; opacity: 0.9; font-weight: 400;'>
            La Propensión Conductual al Ahorro:
            Un estudio desde los sesgos cognitivos
            para la toma de decisiones en el ahorro
            de los hogares
        </h2>
        <hr style='margin: 1.5rem auto; width: 70%; border: 2px solid rgba(255,255,255,0.3);'>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 1.5rem;'>
            <div style='text-align: center; margin: 0.5rem;'>
                <strong style='font-size: 1.1rem;'>MSc. Jesús Fernando Salazar Rojas</strong><br>
                <em style='opacity: 0.8;'>Doctorado en Economía, UCAB – 2025</em>
            </div>
            <div style='text-align: center; margin: 0.5rem;'>
                <strong style='font-size: 1.1rem;'>Dr. Fernando Spiritto</strong><br>
                <em style='opacity: 0.8;'>Tutor</em>
            </div>
            <div style='text-align: center; margin: 0.5rem;'>
                <strong style='font-size: 1.1rem;'>Methodology</strong><br>
                <em style='opacity: 0.8;'>PLS-SEM + Bootstrap Resampling</em>
            </div>
        </div>
    </div>
    """
