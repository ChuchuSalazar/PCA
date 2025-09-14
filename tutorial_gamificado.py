"""
Tutorial Interactivo Gamificado para PCA Simulator v3.2
Sistema de aprendizaje progresivo con elementos de juego y narrativa
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime


class TutorialGamePCA:
    """Clase principal del tutorial gamificado"""
    
    def __init__(self):
        self.init_session_state()
        self.niveles = self.configurar_niveles()
        self.personajes = self.crear_personajes()
        
    def init_session_state(self):
        """Inicializar estado del tutorial"""
        if "tutorial_activo" not in st.session_state:
            st.session_state.tutorial_activo = False
        if "nivel_actual" not in st.session_state:
            st.session_state.nivel_actual = 1
        if "puntos_totales" not in st.session_state:
            st.session_state.puntos_totales = 0
        if "badges_conseguidos" not in st.session_state:
            st.session_state.badges_conseguidos = []
        if "personaje_seleccionado" not in st.session_state:
            st.session_state.personaje_seleccionado = None
        if "historia_usuario" not in st.session_state:
            st.session_state.historia_usuario = {}
        if "tutorial_completado" not in st.session_state:
            st.session_state.tutorial_completado = {}
    
    def configurar_niveles(self):
        """Configurar los niveles del tutorial"""
        return {
            1: {
                "titulo": "🎯 El Misterio del Ahorro Perdido",
                "descripcion": "Descubre por qué las personas no ahorran como esperamos",
                "objetivos": ["Entender PCA vs PSE", "Identificar sesgos básicos"],
                "duracion": "5-7 minutos",
                "dificultad": "Principiante",
                "puntos_max": 100
            },
            2: {
                "titulo": "🧠 La Mente del Ahorrador",
                "descripcion": "Explora los sesgos cognitivos en acción",
                "objetivos": ["Dominar SQ, CS, DH", "Interpretar Bootstrap"],
                "duracion": "8-10 minutos", 
                "dificultad": "Intermedio",
                "puntos_max": 150
            },
            3: {
                "titulo": "📊 El Laboratorio 3D",
                "descripcion": "Navega por el espacio multidimensional de decisiones",
                "objetivos": ["Análisis 3D", "Cuadrantes avanzados"],
                "duracion": "10-12 minutos",
                "dificultad": "Avanzado", 
                "puntos_max": 200
            },
            4: {
                "titulo": "💼 El Consultor Experto",
                "descripcion": "Aplica todo el conocimiento en casos reales",
                "objetivos": ["Recomendaciones", "Análisis completo"],
                "duracion": "12-15 minutos",
                "dificultad": "Experto",
                "puntos_max": 250
            }
        }
    
    def crear_personajes(self):
        """Crear personajes guía del tutorial"""
        return {
            "ana_analista": {
                "nombre": "Ana la Analista",
                "emoji": "👩‍💼",
                "personalidad": "Meticulosa y detallista",
                "especialidad": "Análisis estadístico y métricas",
                "frase_tipica": "Los datos nunca mienten, pero hay que saber leerlos"
            },
            "carlos_cognitivo": {
                "nombre": "Carlos el Cognitivo", 
                "emoji": "🧠",
                "personalidad": "Curioso sobre comportamiento humano",
                "especialidad": "Sesgos cognitivos y psicología",
                "frase_tipica": "La mente humana es fascinante... y predeciblemente irracional"
            },
            "sofia_simuladora": {
                "nombre": "Sofía la Simuladora",
                "emoji": "🔬", 
                "personalidad": "Experimental y práctica",
                "especialidad": "Modelado y simulaciones",
                "frase_tipica": "¡Experimentemos! Los modelos cobran vida con datos reales"
            }
        }

def mostrar_tutorial_principal():
    """Función principal del tutorial - punto de entrada"""
    
    tutorial = TutorialGamePCA()
    
    # Header del tutorial
    mostrar_header_tutorial()

    col_header1, col_header2, col_header3 = st.columns([1, 2, 1])
    with col_header3:
        if st.button("❌ Salir Tutorial", key="salir_tutorial"):
            salir_tutorial()
    
    # Panel de control del tutorial
    if not st.session_state.tutorial_activo:
        mostrar_inicio_tutorial(tutorial)
    else:
        ejecutar_tutorial_activo(tutorial)

def mostrar_header_tutorial():
    """Header atractivo del tutorial"""
    
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 20px; text-align: center; color: white;
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); margin-bottom: 2rem;">
            <h1 style="font-size: 2.5rem; margin: 0 0 1rem 0; font-weight: 700;">
                🎮 Tutorial Interactivo PCA
            </h1>
            <h3 style="margin: 0 0 1rem 0; opacity: 0.9; font-weight: 400;">
                Domina la Economía Conductual del Ahorro de forma Divertida
            </h3>
            <p style="margin: 0; font-size: 1.1rem; opacity: 0.8;">
                Aprende Bootstrap, Sesgos Cognitivos y Análisis 3D con misiones gamificadas
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def mostrar_inicio_tutorial(tutorial):
    """Pantalla de inicio del tutorial"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🚀 ¡Bienvenido a tu Aventura de Aprendizaje!")
        
        # Selección de personaje guía
        st.markdown("#### Elige tu Compañero de Aventura:")
        
        cols_personajes = st.columns(3)
        
        for i, (key, personaje) in enumerate(tutorial.personajes.items()):
            with cols_personajes[i]:
                if st.button(
                    f"{personaje['emoji']} {personaje['nombre']}",
                    key=f"personaje_{key}",
                    help=f"{personaje['personalidad']} - {personaje['especialidad']}",
                    use_container_width=True
                ):
                    st.session_state.personaje_seleccionado = key
                    st.session_state.tutorial_activo = True
                    st.rerun()
                
                # Mostrar info del personaje
                st.markdown(
                    f"""
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; 
                                text-align: center; margin-top: 0.5rem;">
                        <strong>{personaje['personalidad']}</strong><br>
                        <small>{personaje['especialidad']}</small><br>
                        <em>"{personaje['frase_tipica']}"</em>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Preview de niveles
        st.markdown("---")
        st.markdown("#### 🎯 Tu Ruta de Aprendizaje:")
        
        for nivel, info in tutorial.niveles.items():
            dificultad_color = {
                "Principiante": "#27ae60",
                "Intermedio": "#f39c12", 
                "Avanzado": "#e74c3c",
                "Experto": "#8e44ad"
            }
            
            st.markdown(
                f"""
                <div style="background: {dificultad_color[info['dificultad']]}20; 
                            padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
                            border-left: 4px solid {dificultad_color[info['dificultad']]};">
                    <strong>{info['titulo']}</strong> - {info['dificultad']}<br>
                    <small>{info['descripcion']}</small><br>
                    ⏱️ {info['duracion']} | 🏆 {info['puntos_max']} puntos máx.
                </div>
                """,
                unsafe_allow_html=True
            )

def ejecutar_tutorial_activo(tutorial):
    """Ejecutar el tutorial activo"""
    
    # Barra de progreso general
    mostrar_barra_progreso_global(tutorial)
    
    # Panel del personaje guía
    mostrar_panel_personaje(tutorial)
    
    # Ejecutar nivel actual
    nivel_actual = st.session_state.nivel_actual
    
    if nivel_actual == 1:
        ejecutar_nivel_1(tutorial)
    elif nivel_actual == 2:
        ejecutar_nivel_2(tutorial)
    elif nivel_actual == 3:
        ejecutar_nivel_3(tutorial)
    elif nivel_actual == 4:
        ejecutar_nivel_4(tutorial)
    
    # Panel de logros y puntos
    mostrar_panel_logros(tutorial)

def mostrar_barra_progreso_global(tutorial):
    """Mostrar progreso general del tutorial"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nivel Actual", f"{st.session_state.nivel_actual}/4")
    
    with col2:
        st.metric("Puntos Totales", st.session_state.puntos_totales)
    
    with col3:
        badges_count = len(st.session_state.badges_conseguidos)
        st.metric("Logros", f"{badges_count}/12")
    
    with col4:
        progreso_pct = (st.session_state.nivel_actual - 1) / 4 * 100
        st.metric("Progreso", f"{progreso_pct:.0f}%")
    
    # Barra de progreso visual
    progress_bar = st.progress(progreso_pct / 100)

def mostrar_panel_personaje(tutorial):
    """Panel del personaje guía"""
    
    personaje_key = st.session_state.personaje_seleccionado
    personaje = tutorial.personajes[personaje_key]
    
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); 
                    padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;
                    box-shadow: 0 5px 15px rgba(116, 185, 255, 0.3);">
            <h4 style="margin: 0 0 1rem 0;">
                {personaje['emoji']} {personaje['nombre']} - Tu Guía
            </h4>
            <div id="dialogo-personaje" style="background: rgba(255,255,255,0.2); 
                                              padding: 1rem; border-radius: 10px;
                                              min-height: 60px;">
                <span id="texto-dialogo">¡Hola! Estoy aquí para ayudarte en esta aventura...</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def ejecutar_nivel_1(tutorial):
    """Nivel 1: El Misterio del Ahorro Perdido"""
    
    st.markdown("## 🎯 Nivel 1: El Misterio del Ahorro Perdido")
    
    # Historia introductoria
    with st.expander("📖 Historia del Nivel", expanded=True):
        st.markdown(
            """
            **La Situación:** Te encuentras en el banco central de un país donde algo extraño sucede. 
            Los ciudadanos tienen capacidad económica para ahorrar, pero no lo hacen como se esperaba. 
            
            **Tu Misión:** Descubrir la diferencia entre lo que la economía tradicional predice (PSE) 
            y lo que realmente sucede (PCA). ¡Los sesgos cognitivos tienen la clave del misterio!
            
            **Herramientas:** Vas a usar el simulador para generar datos y descubrir patrones ocultos.
            """
        )
    
    # Misión 1.1: Entender PCA vs PSE
    st.markdown("### 🔍 Misión 1.1: Decodifica la Diferencia")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulador visual simple PCA vs PSE
        st.markdown("**🎮 Simulador Interactivo**")
        
        # Controles del simulador
        persona_tipo = st.selectbox(
            "Elige el tipo de persona:",
            ["Joven estudiante", "Adulto trabajador", "Profesional senior"],
            key="persona_nivel1"
        )
        
        ingresos = st.slider("Ingresos mensuales (USD)", 500, 5000, 2000, key="ingresos_nivel1")
        
        if st.button("🔮 Simular Comportamiento", key="simular_nivel1"):
            # Generar datos simulados
            pse_esperado, pca_real = simular_comportamiento_basico(persona_tipo, ingresos)
            
            # Mostrar resultados con animación
            crear_grafico_comparativo_nivel1(pse_esperado, pca_real, persona_tipo)
            
            # Sistema de puntos
            diferencia = abs(pca_real - pse_esperado)
            if diferencia > 0.3:
                st.success("🏆 ¡Gran descubrimiento! Hay una diferencia significativa (+20 puntos)")
                agregar_puntos(20)
                agregar_badge("detective_diferencias")
    
    with col2:
        # Pregunta interactiva
        st.markdown("**❓ Pregunta Detective**")
        
        respuesta = st.radio(
            "Si PSE (predicción económica) es 0.8 y PCA (real) es 0.5, ¿qué significa?",
            [
                "La persona ahorra más de lo esperado",
                "La persona ahorra menos de lo esperado", 
                "La predicción fue perfecta",
                "Los datos están mal"
            ],
            key="pregunta_nivel1"
        )
        
        if st.button("✅ Verificar Respuesta", key="verificar_nivel1"):
            if respuesta == "La persona ahorra menos de lo esperado":
                st.success("🎉 ¡Correcto! +30 puntos")
                agregar_puntos(30)
                st.markdown("**Explicación:** PCA < PSE indica que factores psicológicos reducen el ahorro esperado.")
            else:
                st.error("❌ Incorrecto. Inténtalo de nuevo.")
    
    # Misión 1.2: Identificar sesgos básicos
    st.markdown("### 🧠 Misión 1.2: Los Culpables Psicológicos")
    
    crear_quiz_sesgos_nivel1()
    
    # Verificar si puede pasar al siguiente nivel
    if verificar_completitud_nivel1():
        mostrar_boton_siguiente_nivel()

def ejecutar_nivel_2(tutorial):
    """Nivel 2: La Mente del Ahorrador"""
    
    st.markdown("## 🧠 Nivel 2: La Mente del Ahorrador")
    
    # Desafío Bootstrap
    st.markdown("### 📊 Desafío: Domina el Bootstrap")
    
    # Simulador de Bootstrap interactivo
    crear_simulador_bootstrap_interactivo()
    
    # Juego de identificación de sesgos
    st.markdown("### 🎯 Juego: Identifica el Sesgo Dominante")
    crear_juego_identificacion_sesgos()

def ejecutar_nivel_3(tutorial):
    """Nivel 3: El Laboratorio 3D"""
    
    st.markdown("## 📊 Nivel 3: El Laboratorio 3D")
    
    # Navegador 3D interactivo
    crear_navegador_3d_tutorial()
    
    # Misión de cuadrantes
    crear_mision_cuadrantes()

def ejecutar_nivel_4(tutorial):
    """Nivel 4: El Consultor Experto"""
    
    st.markdown("## 💼 Nivel 4: El Consultor Experto")
    
    # Caso de estudio completo
    crear_caso_estudio_final()

def simular_comportamiento_basico(persona_tipo, ingresos):
    """Simular comportamiento básico para el nivel 1"""
    
    # Lógica simplificada para demostración
    base_pse = ingresos / 5000  # PSE basado en ingresos
    
    # Sesgos según tipo de persona
    sesgos = {
        "Joven estudiante": {"sq": 0.3, "dh": 0.8, "cs": 0.6},
        "Adulto trabajador": {"sq": 0.5, "dh": 0.4, "cs": 0.3},
        "Profesional senior": {"sq": 0.7, "dh": 0.2, "cs": 0.2}
    }
    
    persona_sesgos = sesgos[persona_tipo]
    
    # Calcular PCA con impacto de sesgos
    impacto_sesgos = np.mean(list(persona_sesgos.values()))
    pca_real = base_pse - (impacto_sesgos * 0.3)  # Los sesgos reducen el ahorro
    
    return max(0, min(1, base_pse)), max(0, min(1, pca_real))

def crear_grafico_comparativo_nivel1(pse, pca, persona_tipo):
    """Crear gráfico comparativo animado"""
    
    fig = go.Figure()
    
    # Barras comparativas
    fig.add_trace(go.Bar(
        x=["PSE (Esperado)", "PCA (Real)"],
        y=[pse, pca], 
        marker_color=["#3498db", "#e74c3c"],
        text=[f"{pse:.2f}", f"{pca:.2f}"],
        textposition="auto"
    ))
    
    fig.update_layout(
        title=f"Comportamiento de Ahorro: {persona_tipo}",
        yaxis_title="Propensión al Ahorro (0-1)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación automática
    diferencia = pca - pse
    if diferencia < -0.2:
        st.warning("🔍 **Hallazgo:** Los sesgos cognitivos están reduciendo significativamente el ahorro")
    elif diferencia > 0.2:
        st.success("🔍 **Hallazgo:** Esta persona ahorra más de lo esperado - ¡factores positivos en juego!")
    else:
        st.info("🔍 **Hallazgo:** Comportamiento bastante predecible según la teoría económica")

def crear_quiz_sesgos_nivel1():
    """Quiz interactivo sobre sesgos básicos"""
    
    st.markdown("**🎮 Quiz Interactivo: Identifica los Sesgos**")
    
    # Casos prácticos
    casos = [
        {
            "situacion": "María siempre dice 'mañana empiezo a ahorrar' pero nunca lo hace",
            "respuesta_correcta": "Descuento Hiperbólico",
            "opciones": ["Status Quo", "Descuento Hiperbólico", "Contagio Social", "Aversión al Riesgo"],
            "explicacion": "Prefiere gratificación inmediata sobre beneficios futuros"
        },
        {
            "situacion": "Pedro no cambia de banco aunque le ofrezcan mejores tasas de ahorro",
            "respuesta_correcta": "Status Quo", 
            "opciones": ["Status Quo", "Descuento Hiperbólico", "Contagio Social", "Aversión al Riesgo"],
            "explicacion": "Tendencia a mantener el estado actual por inercia"
        }
    ]
    
    for i, caso in enumerate(casos):
        with st.expander(f"🔍 Caso {i+1}: {caso['situacion']}", expanded=True):
            respuesta_usuario = st.radio(
                "¿Qué sesgo está operando?",
                caso["opciones"],
                key=f"quiz_caso_{i}"
            )
            
            if st.button(f"Verificar Caso {i+1}", key=f"verificar_caso_{i}"):
                if respuesta_usuario == caso["respuesta_correcta"]:
                    st.success(f"🎉 ¡Correcto! {caso['explicacion']} (+15 puntos)")
                    agregar_puntos(15)
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es: {caso['respuesta_correcta']}")
                    st.info(f"💡 Explicación: {caso['explicacion']}")

def crear_simulador_bootstrap_interactivo():
    """Simulador de Bootstrap interactivo para nivel 2"""
    
    st.markdown("**🎲 Simulador Bootstrap: Ve la Magia Estadística**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_bootstrap = st.slider("Iteraciones Bootstrap", 100, 1000, 500, step=100)
        escenario = st.selectbox("Escenario económico", ["Estable", "Crisis", "Bonanza"])
        
        if st.button("🎯 Ejecutar Bootstrap", key="bootstrap_nivel2"):
            # Simular bootstrap simple
            resultados = simular_bootstrap_tutorial(n_bootstrap, escenario)
            
            # Mostrar animación de resultados
            mostrar_resultados_bootstrap_animados(resultados)
            
            # Puntos por comprensión
            agregar_puntos(25)
            st.success(f"🏆 ¡Dominas el Bootstrap! +25 puntos")
    
    with col2:
        st.markdown("**📚 ¿Qué está pasando?**")
        st.info(
            """
            El **Bootstrap** es como crear múltiples universos paralelos con tus datos:
            
            1. 🎲 Toma muestras aleatorias de tus datos originales
            2. 🔄 Repite el proceso cientos/miles de veces  
            3. 📊 Analiza la distribución de resultados
            4. 🎯 Obtén intervalos de confianza robustos
            
            ¡Es magia estadística en acción!
            """
        )

def simular_bootstrap_tutorial(n_iterations, escenario):
    """Simulación simplificada de bootstrap para tutorial"""
    
    # Generar datos base según escenario
    if escenario == "Crisis":
        base_pca = np.random.normal(0.3, 0.2, 1000)
    elif escenario == "Bonanza":
        base_pca = np.random.normal(0.7, 0.15, 1000) 
    else:  # Estable
        base_pca = np.random.normal(0.5, 0.1, 1000)
    
    # Simular bootstrap
    bootstrap_results = []
    for _ in range(n_iterations):
        sample = np.random.choice(base_pca, size=len(base_pca), replace=True)
        bootstrap_results.append(np.mean(sample))
    
    return {
        "resultados": bootstrap_results,
        "media": np.mean(bootstrap_results),
        "std": np.std(bootstrap_results),
        "ci_lower": np.percentile(bootstrap_results, 2.5),
        "ci_upper": np.percentile(bootstrap_results, 97.5)
    }

def mostrar_resultados_bootstrap_animados(resultados):
    """Mostrar resultados de bootstrap con animación"""
    
    # Histograma animado
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=resultados["resultados"],
        nbinsx=30,
        opacity=0.7,
        marker_color="#3498db",
        name="Distribución Bootstrap"
    ))
    
    # Líneas de intervalo de confianza
    fig.add_vline(x=resultados["ci_lower"], line_dash="dash", line_color="red", 
                  annotation_text="CI 2.5%")
    fig.add_vline(x=resultados["ci_upper"], line_dash="dash", line_color="red",
                  annotation_text="CI 97.5%")
    fig.add_vline(x=resultados["media"], line_color="green", line_width=3,
                  annotation_text="Media")
    
    fig.update_layout(
        title="Distribución Bootstrap de PCA",
        xaxis_title="Valor PCA",
        yaxis_title="Frecuencia",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas clave
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Media Bootstrap", f"{resultados['media']:.4f}")
    with col2:
        st.metric("Desv. Estándar", f"{resultados['std']:.4f}")
    with col3:
        ci_width = resultados['ci_upper'] - resultados['ci_lower']
        st.metric("Ancho IC 95%", f"{ci_width:.4f}")

def agregar_puntos(puntos):
    """Agregar puntos al jugador"""
    st.session_state.puntos_totales += puntos

def agregar_badge(badge_id):
    """Agregar badge al jugador"""
    if badge_id not in st.session_state.badges_conseguidos:
        st.session_state.badges_conseguidos.append(badge_id)

def verificar_completitud_nivel1():
    """Verificar si el nivel 1 está completo"""
    # Lógica simplificada - en implementación real sería más robusta
    return st.session_state.puntos_totales >= 50

def mostrar_boton_siguiente_nivel():
    """Mostrar botón para avanzar al siguiente nivel"""
    
    st.markdown("---")
    st.success("🎉 ¡Nivel Completado!")
    
    if st.button("🚀 Avanzar al Siguiente Nivel", key="siguiente_nivel", type="primary"):
        st.session_state.nivel_actual += 1
        st.rerun()

def mostrar_panel_logros(tutorial):
    """Panel de logros y badges"""
    
    with st.expander("🏆 Logros y Badges", expanded=False):
        
        todos_badges = {
            "detective_diferencias": {"emoji": "🔍", "nombre": "Detective de Diferencias"},
            "maestro_sesgos": {"emoji": "🧠", "nombre": "Maestro de Sesgos"},
            "bootstrap_ninja": {"emoji": "🥷", "nombre": "Bootstrap Ninja"},
            "explorador_3d": {"emoji": "🚀", "nombre": "Explorador 3D"},
            "consultor_experto": {"emoji": "💼", "nombre": "Consultor Experto"}
        }
        
        cols = st.columns(5)
        
        for i, (badge_id, badge_info) in enumerate(todos_badges.items()):
            with cols[i]:
                if badge_id in st.session_state.badges_conseguidos:
                    st.markdown(
                        f"""
                        <div style="background: #d4edda; padding: 1rem; border-radius: 10px; text-align: center;">
                            <div style="font-size: 2rem;">{badge_info['emoji']}</div>
                            <strong>{badge_info['nombre']}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; opacity: 0.5;">
                            <div style="font-size: 2rem;">🔒</div>
                            <small>{badge_info['nombre']}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Funciones auxiliares para niveles avanzados
def crear_juego_identificacion_sesgos():
    """Juego de identificación de sesgos para nivel 2"""
    
    st.markdown("**🎮 Juego: Diagnóstica el Sesgo**")
    
    # Casos más complejos con múltiples sesgos
    casos_avanzados = [
        {
            "personaje": "Ana, 28 años, Marketing",
            "historia": "Ana ve que sus colegas están invirtiendo en criptomonedas. Aunque no entiende bien el mercado, decide seguir la tendencia y deja de ahorrar tradicionalmente.",
            "sesgos_presentes": ["Contagio Social", "Aversión al Riesgo (inversa)"],
            "sesgo_dominante": "Contagio Social",
            "puntos": 30
        },
        {
            "personaje": "Roberto, 45 años, Ingeniero",
            "historia": "Roberto lleva 15 años con el mismo plan de pensiones. Le han ofrecido mejores opciones pero dice 'si ha funcionado hasta ahora, ¿para qué cambiar?'",
            "sesgos_presentes": ["Status Quo", "Descuento Hiperbólico"],
            "sesgo_dominante": "Status Quo",
            "puntos": 35
        }
    ]
    
    for i, caso in enumerate(casos_avanzados):
        with st.expander(f"🎭 {caso['personaje']}", expanded=True):
            st.markdown(f"**Historia:** {caso['historia']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sesgo_usuario = st.selectbox(
                    "¿Cuál es el sesgo DOMINANTE?",
                    ["Status Quo", "Descuento Hiperbólico", "Contagio Social", "Aversión al Riesgo"],
                    key=f"sesgo_dominante_{i}"
                )
                
                if st.button(f"Diagnosticar {caso['personaje'].split(',')[0]}", key=f"diagnosticar_{i}"):
                    if sesgo_usuario == caso['sesgo_dominante']:
                        st.success(f"🎯 ¡Diagnóstico correcto! +{caso['puntos']} puntos")
                        agregar_puntos(caso['puntos'])
                        if caso['puntos'] >= 35:
                            agregar_badge("maestro_sesgos")
                    else:
                        st.error(f"❌ El sesgo dominante es: {caso['sesgo_dominante']}")
            
            with col2:
                st.markdown("**💡 Sesgos Detectados:**")
                for sesgo in caso['sesgos_presentes']:
                    st.markdown(f"• {sesgo}")

def crear_navegador_3d_tutorial():
    """Navegador 3D interactivo simplificado para tutorial"""
    
    st.markdown("**🚀 Navegador 3D: Explora el Espacio de Decisiones**")
    
    # Simulador 3D básico con controles
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**🎛️ Panel de Control**")
        
        rotacion_x = st.slider("Rotación X", -180, 180, 45, key="rot_x_tutorial")
        rotacion_y = st.slider("Rotación Y", -180, 180, 45, key="rot_y_tutorial") 
        zoom = st.slider("Zoom", 0.5, 3.0, 1.0, key="zoom_tutorial")
        
        mostrar_sesgos = st.multiselect(
            "Mostrar sesgos:",
            ["Status Quo", "Desc. Hiperbólico", "Contagio Social"],
            default=["Status Quo"],
            key="sesgos_3d_tutorial"
        )
        
        if st.button("🎯 Generar Vista 3D", key="generar_3d_tutorial"):
            crear_grafico_3d_tutorial(rotacion_x, rotacion_y, zoom, mostrar_sesgos)
    
    with col2:
        st.markdown("**📊 Interpretación 3D**")
        st.info("""
        **Cómo leer el espacio 3D:**
        
        🔵 **Puntos azules**: Ahorradores consistentes
        🔴 **Puntos rojos**: Comportamiento impredecible  
        🟡 **Puntos amarillos**: Influenciados por sesgos
        
        **Clusters**: Grupos de comportamiento similar
        **Dispersión**: Variabilidad en decisiones
        **Altura**: Intensidad del sesgo dominante
        """)

def crear_grafico_3d_tutorial(rot_x, rot_y, zoom, sesgos_seleccionados):
    """Crear gráfico 3D simplificado para tutorial"""
    
    # Generar datos simulados
    n_points = 100
    np.random.seed(42)
    
    x = np.random.normal(0, 1, n_points)  # PCA
    y = np.random.normal(0, 1, n_points)  # PSE
    z = np.random.normal(0, 1, n_points)  # Sesgo dominante
    
    # Colores según sesgos seleccionados
    colors = ['blue'] * n_points
    if 'Status Quo' in sesgos_seleccionados:
        colors = ['red' if zi > 0.5 else c for zi, c in zip(z, colors)]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.7
        ),
        hovertemplate="PCA: %{x:.2f}<br>PSE: %{y:.2f}<br>Sesgo: %{z:.2f}<extra></extra>"
    )])
    
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=rot_x/100, y=rot_y/100, z=zoom)
            )
        ),
        title="Espacio 3D: PCA × PSE × Sesgos",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Agregar puntos por exploración
    st.success("🎯 ¡Exploración 3D completada! +40 puntos")
    agregar_puntos(40)
    agregar_badge("explorador_3d")

def crear_mision_cuadrantes():
    """Misión de cuadrantes interactiva"""
    
    st.markdown("**🎯 Misión: Clasifica los Perfiles de Ahorro**")
    
    # Casos para clasificar en cuadrantes
    perfiles = [
        {
            "nombre": "Carlos, Ejecutivo Senior",
            "pca": 0.8, "pse": 0.9,
            "cuadrante_correcto": "I",
            "descripcion": "Alta capacidad, alta propensión"
        },
        {
            "nombre": "María, Estudiante Disciplinada", 
            "pca": 0.7, "pse": 0.3,
            "cuadrante_correcto": "II",
            "descripcion": "Baja capacidad, alta propensión"
        },
        {
            "nombre": "Luis, Trabajador Desorganizado",
            "pca": 0.2, "pse": 0.1, 
            "cuadrante_correcto": "III",
            "descripcion": "Baja capacidad, baja propensión"
        }
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎮 Clasificador de Cuadrantes**")
        
        for i, perfil in enumerate(perfiles):
            with st.container():
                st.markdown(f"**{perfil['nombre']}**")
                st.markdown(f"PCA: {perfil['pca']} | PSE: {perfil['pse']}")
                
                cuadrante_usuario = st.selectbox(
                    f"¿En qué cuadrante clasificarías a {perfil['nombre'].split(',')[0]}?",
                    ["I (Alta PCA + Alta PSE)", "II (Alta PCA + Baja PSE)", 
                     "III (Baja PCA + Baja PSE)", "IV (Baja PCA + Alta PSE)"],
                    key=f"cuadrante_{i}"
                )
                
                if st.button(f"Clasificar {perfil['nombre'].split(',')[0]}", key=f"clasificar_{i}"):
                    cuadrante_letra = cuadrante_usuario.split()[0]
                    if cuadrante_letra == perfil['cuadrante_correcto']:
                        st.success(f"🎯 ¡Correcto! {perfil['descripcion']} (+25 puntos)")
                        agregar_puntos(25)
                    else:
                        st.error(f"❌ Cuadrante correcto: {perfil['cuadrante_correcto']}")
    
    with col2:
        # Mapa de cuadrantes visual
        crear_mapa_cuadrantes_tutorial()

def crear_mapa_cuadrantes_tutorial():
    """Crear mapa visual de cuadrantes"""
    
    st.markdown("**📊 Mapa de Cuadrantes**")
    
    fig = go.Figure()
    
    # Líneas divisorias
    fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="gray", dash="dash"))
    fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="gray", dash="dash"))
    
    # Etiquetas de cuadrantes
    cuadrantes_info = [
        {"x": 0.5, "y": 0.5, "text": "I<br>Ahorrador Ideal", "color": "#27ae60"},
        {"x": 0.5, "y": -0.5, "text": "II<br>Ahorrador Resiliente", "color": "#f39c12"},
        {"x": -0.5, "y": -0.5, "text": "III<br>Doble Riesgo", "color": "#e74c3c"},
        {"x": -0.5, "y": 0.5, "text": "IV<br>Paradoja", "color": "#9b59b6"}
    ]
    
    for cuad in cuadrantes_info:
        fig.add_annotation(
            x=cuad["x"], y=cuad["y"],
            text=cuad["text"],
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor=cuad["color"],
            bordercolor="white",
            borderwidth=2
        )
    
    fig.update_layout(
        xaxis_title="PSE (Perfil Socioeconómico)",
        yaxis_title="PCA (Propensión Conductual)",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        height=400,
        title="Matriz de Cuadrantes PCA-PSE"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def crear_caso_estudio_final():
    """Caso de estudio completo para nivel 4"""
    
    st.markdown("**💼 Caso Final: Consultoría para Banco Nacional**")
    
    with st.expander("📋 Briefing del Cliente", expanded=True):
        st.markdown("""
        **Cliente:** Banco Nacional de Ahorros
        **Problema:** Sus campañas de ahorro no funcionan como esperaban
        **Datos disponibles:** 1000 clientes con análisis PCA completo
        **Tu misión:** Proporcionar recomendaciones estratégicas basadas en análisis conductual
        """)
    
    # Simulación de datos del banco
    if st.button("📊 Cargar Datos del Cliente", key="cargar_datos_final"):
        datos_banco = generar_datos_caso_final()
        
        # Mostrar análisis automático
        mostrar_analisis_caso_final(datos_banco)
        
        # Solicitar recomendaciones
        crear_formulario_recomendaciones()

def generar_datos_caso_final():
    """Generar datos simulados para caso final"""
    
    np.random.seed(42)
    n_clientes = 1000
    
    # Generar datos realistas
    datos = {
        "edad": np.random.normal(40, 15, n_clientes),
        "ingresos": np.random.lognormal(8, 0.5, n_clientes),
        "pca": np.random.beta(2, 3, n_clientes),
        "pse": np.random.beta(3, 2, n_clientes),
        "sq": np.random.normal(0, 0.3, n_clientes),
        "dh": np.random.normal(0, 0.4, n_clientes),
        "cs": np.random.normal(0, 0.2, n_clientes)
    }
    
    return pd.DataFrame(datos)

def mostrar_analisis_caso_final(datos):
    """Mostrar análisis automático del caso final"""
    
    st.markdown("### 📊 Análisis Automático de la Cartera")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pca_promedio = datos['pca'].mean()
        st.metric("PCA Promedio", f"{pca_promedio:.3f}")
    
    with col2:
        pse_promedio = datos['pse'].mean() 
        st.metric("PSE Promedio", f"{pse_promedio:.3f}")
    
    with col3:
        correlacion = datos['pca'].corr(datos['pse'])
        st.metric("Correlación PCA-PSE", f"{correlacion:.3f}")
    
    with col4:
        # Calcular cuadrante modal
        datos['cuadrante'] = datos.apply(lambda row: 
            "I" if row['pca'] > 0.5 and row['pse'] > 0.5 else
            "II" if row['pca'] > 0.5 and row['pse'] <= 0.5 else  
            "III" if row['pca'] <= 0.5 and row['pse'] <= 0.5 else "IV", axis=1)
        
        cuadrante_modal = datos['cuadrante'].mode().iloc[0]
        st.metric("Cuadrante Dominante", cuadrante_modal)
    
    # Gráfico de dispersión
    fig = go.Figure(data=go.Scatter(
        x=datos['pse'],
        y=datos['pca'],
        mode='markers',
        marker=dict(
            color=datos['sq'],
            colorscale='RdYlBu',
            size=5,
            opacity=0.6,
            colorbar=dict(title="Status Quo")
        ),
        hovertemplate="PSE: %{x:.3f}<br>PCA: %{y:.3f}<extra></extra>"
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Distribución de Clientes: PCA vs PSE",
        xaxis_title="PSE (Perfil Socioeconómico)",
        yaxis_title="PCA (Propensión Conductual)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def crear_formulario_recomendaciones():
    """Formulario para recomendaciones finales"""
    
    st.markdown("### 💡 Tu Diagnóstico y Recomendaciones")
    
    with st.form("recomendaciones_finales"):
        problema_principal = st.selectbox(
            "¿Cuál es el problema principal identificado?",
            [
                "Desconexión entre capacidad económica y comportamiento",
                "Sesgos cognitivos dominantes interfieren con ahorro",
                "Segmentación inadecuada de clientes", 
                "Estrategias no personalizadas por cuadrante"
            ]
        )
        
        estrategia_recomendada = st.selectbox(
            "¿Cuál sería tu estrategia principal?",
            [
                "Segmentación por cuadrantes con mensajes diferenciados",
                "Intervenciones específicas anti-sesgos",
                "Productos adaptativos según perfil conductual",
                "Programa de educación financiera conductual"
            ]
        )
        
        justificacion = st.text_area(
            "Justifica tu recomendación (100-200 palabras):",
            placeholder="Basándome en el análisis PCA realizado..."
        )
        
        submitted = st.form_submit_button("🎯 Enviar Diagnóstico Final")
        
        if submitted:
            if justificacion and len(justificacion.split()) >= 20:
                st.success("🏆 ¡Consultoría completada! Has demostrado dominio completo del sistema PCA")
                st.balloons()
                
                agregar_puntos(100)
                agregar_badge("consultor_experto")
                
                # Mostrar certificado
                mostrar_certificado_completitud()
            else:
                st.error("Por favor, proporciona una justificación más detallada (mínimo 20 palabras)")

def mostrar_certificado_completitud():
    """Mostrar certificado de completitud del tutorial"""
    
    st.markdown("---")
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #ffd700, #ffed4e); 
                    padding: 3rem; border-radius: 20px; text-align: center; 
                    border: 5px solid #f39c12; margin: 2rem 0;
                    box-shadow: 0 15px 35px rgba(255, 215, 0, 0.3);">
            <h1 style="color: #8e44ad; margin: 0 0 1rem 0; font-size: 2.5rem;">
                🎓 CERTIFICADO DE COMPLETITUD
            </h1>
            <h2 style="color: #2c3e50; margin: 0 0 2rem 0;">
                Simulador PCA v3.2 - Tutorial Interactivo
            </h2>
            <div style="background: white; padding: 2rem; border-radius: 15px; margin: 1rem 0;">
                <h3 style="color: #e74c3c; margin: 0 0 1rem 0;">
                    ¡FELICITACIONES!
                </h3>
                <p style="color: #2c3e50; font-size: 1.2rem; margin: 0;">
                    Has completado exitosamente el tutorial interactivo<br>
                    <strong>Puntuación Final: {st.session_state.puntos_totales} puntos</strong><br>
                    <strong>Badges Conseguidos: {len(st.session_state.badges_conseguidos)}/5</strong>
                </p>
            </div>
            <p style="color: #8e44ad; margin: 1rem 0 0 0; font-size: 1.1rem;">
                Ahora estás preparado para usar el simulador PCA como un experto<br>
                en análisis conductual del ahorro
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Botón para volver al simulador principal
    if st.button("🚀 Ir al Simulador Principal", key="ir_simulador_principal", type="primary"):
        st.session_state.tutorial_activo = False
        st.session_state.nivel_actual = 1  # Reset para futuros usos
        st.rerun()


# Función principal de integración con el sistema
def integrar_tutorial_en_main():
    """Integración del tutorial en main.py"""
    
    # Esta función se llamaría desde main.py como:
    # from tutorial_interactivo import mostrar_tutorial_principal
    # 
    # En el sidebar de main.py:
    # if st.sidebar.button("🎮 Tutorial Interactivo", type="secondary"):
    #     mostrar_tutorial_principal()
    
    return mostrar_tutorial_principal

def salir_tutorial():
    """Función para salir del tutorial"""
    st.session_state.modo_tutorial = False
    st.session_state.tutorial_activo = False
    st.rerun()