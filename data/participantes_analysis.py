import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
Análisis descriptivo de participantes del estudio
"""


def cargar_datos_participantes():
    """Carga datos de participantes desde Excel"""
    try:
        if os.path.exists("mae_participantes.xlsx"):
            df = pd.read_excel("mae_participantes.xlsx")
            return df
        else:
            # Generar datos simulados basados en la estructura proporcionada
            return generar_datos_participantes_simulados()
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return generar_datos_participantes_simulados()


def generar_datos_participantes_simulados():
    """Genera datos simulados de participantes"""
    np.random.seed(42)
    n_participantes = 500

    data = {
        "Item": range(1, n_participantes + 1),
        "PCA1": np.random.choice(
            [1, 2], n_participantes, p=[0.45, 0.55]
        ),  # Sexo: más mujeres
        "PCA2": np.random.choice(
            range(1, 10),
            n_participantes,
            p=[0.05, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02],
        ),  # Edad
        "PCA4": np.random.choice(
            range(1, 7), n_participantes, p=[0.02, 0.08, 0.15, 0.45, 0.25, 0.05]
        ),  # Estudios
        "PCA5": np.random.choice(
            range(1, 7), n_participantes, p=[0.20, 0.30, 0.25, 0.15, 0.08, 0.02]
        ),  # Ingresos
        "PCA6": np.random.choice([1, 2], n_participantes, p=[0.35, 0.65]),  # Ahorro
        "PCA7": np.random.choice(
            [1, 2], n_participantes, p=[0.40, 0.60]
        ),  # Estado civil
    }

    return pd.DataFrame(data)


def crear_analisis_participantes():
    """Crea análisis completo de participantes"""

    st.markdown("### 👥 Información Descriptiva de Participantes")
    st.markdown("**Caracterización sociodemográfica de la muestra del estudio**")

    # Cargar datos
    df = cargar_datos_participantes()

    if df.empty:
        st.error("No se pudieron cargar los datos de participantes")
        return

    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Resumen General",
            "📈 Distribuciones",
            "🔍 Análisis Cruzado",
            "📋 Tablas Detalladas",
        ]
    )

    with tab1:
        mostrar_resumen_general(df)

    with tab2:
        mostrar_distribuciones_detalladas(df)

    with tab3:
        mostrar_analisis_cruzado(df)

    with tab4:
        mostrar_tablas_detalladas(df)


def mostrar_resumen_general(df):
    """Muestra resumen general de participantes"""

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_participantes = len(df)
        st.metric("Total Participantes", total_participantes)

    with col2:
        mujeres_pct = (df["PCA1"] == 2).mean() * 100
        st.metric("% Mujeres", f"{mujeres_pct:.1f}%")

    with col3:
        ahorradores_pct = (df["PCA6"] == 2).mean() * 100
        st.metric("% Ahorradores", f"{ahorradores_pct:.1f}%")

    with col4:
        universitarios_pct = (df["PCA4"] >= 4).mean() * 100
        st.metric("% Nivel Universitario+", f"{universitarios_pct:.1f}%")

    # Gráfico de composición por sexo y ahorro
    fig_composicion = create_composition_chart(df)
    st.plotly_chart(fig_composicion, use_container_width=True)


def create_composition_chart(df):
    """Crea gráfico de composición de la muestra"""

    # Crear datos para sunburst
    df_comp = df.copy()
    df_comp["Sexo"] = df_comp["PCA1"].map({1: "Hombre", 2: "Mujer"})
    df_comp["Ahorra"] = df_comp["PCA6"].map({1: "No Ahorra", 2: "Sí Ahorra"})
    df_comp["Estado_Civil"] = df_comp["PCA7"].map(
        {1: "Soltero/Div/Viudo", 2: "Casado/U.Libre"}
    )

    # Contar combinaciones
    combinaciones = (
        df_comp.groupby(["Sexo", "Ahorra", "Estado_Civil"])
        .size()
        .reset_index(name="count")
    )

    # Preparar datos para sunburst
    ids = []
    labels = []
    parents = []
    values = []

    # Nivel 1: Sexo
    for sexo in df_comp["Sexo"].unique():
        ids.append(sexo)
        labels.append(sexo)
        parents.append("")
        values.append(df_comp[df_comp["Sexo"] == sexo].shape[0])

    # Nivel 2: Sexo + Ahorro
    for _, row in df_comp.groupby(["Sexo", "Ahorra"]).size().reset_index(name="count").iterrows():
        id_nivel2 = f"{row['Sexo']} - {row['Ahorra']}"
        ids.append(id_nivel2)
        labels.append(row["Ahorra"])
        parents.append(row["Sexo"])
        values.append(row["count"])

    # Nivel 3: Sexo + Ahorro + Estado Civil
    for _, row in combinaciones.iterrows():
        id_nivel3 = f"{row['Sexo']} - {row['Ahorra']} - {row['Estado_Civil']}"
        parent_nivel3 = f"{row['Sexo']} - {row['Ahorra']}"
        ids.append(id_nivel3)
        labels.append(row["Estado_Civil"])
        parents.append(parent_nivel3)
        values.append(row["count"])

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            maxdepth=3,
            branchvalues="total",
        )
    )

    fig.update_layout(
        title="Composición de la Muestra: Sexo → Comportamiento de Ahorro → Estado Civil",
        height=500,
    )

    return fig


def mostrar_distribuciones_detalladas(df):
    """Muestra distribuciones detalladas de variables"""

    # Mapeos para etiquetas
    mapeos = {
        "PCA1": {1: "Hombre", 2: "Mujer"},
        "PCA2": {
            1: "<26",
            2: "26-30",
            3: "31-35",
            4: "36-40",
            5: "41-45",
            6: "46-50",
            7: "51-55",
            8: "56-60",
            9: ">60",
        },
        "PCA4": {
            1: "Primaria",
            2: "Bachillerato",
            3: "TSU",
            4: "Universitario",
            5: "Postgrado",
            6: "Doctorado",
        },
        "PCA5": {
            1: "$3-100",
            2: "$101-450",
            3: "$451-1800",
            4: "$1801-2500",
            5: "$2501-10000",
            6: "+$10000",
        },
        "PCA6": {1: "No Ahorra", 2: "Sí Ahorra"},
        "PCA7": {1: "Soltero/Div/Viudo", 2: "Casado/U.Libre"},
    }

    titulos = {
        "PCA1": "Distribución por Sexo",
        "PCA2": "Distribución por Edad",
        "PCA4": "Nivel de Estudios",
        "PCA5": "Ingresos Mensuales (USD)",
        "PCA6": "Comportamiento de Ahorro",
        "PCA7": "Estado Civil",
    }

    # Crear subplots
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            titulos[col] for col in ["PCA1", "PCA2", "PCA4", "PCA5", "PCA6", "PCA7"]
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
        ],
    )

    variables = ["PCA1", "PCA2", "PCA4", "PCA5", "PCA6", "PCA7"]
    posiciones = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    colores = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for var, pos, color in zip(variables, posiciones, colores):
        freq = df[var].value_counts().sort_index()
        labels = [mapeos[var].get(idx, str(idx)) for idx in freq.index]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=freq.values,
                marker_color=color,
                name=titulos[var],
                showlegend=False,
            ),
            row=pos[0],
            col=pos[1],
        )

    fig.update_layout(
        height=600, title_text="Distribuciones Sociodemográficas de Participantes"
    )
    st.plotly_chart(fig, use_container_width=True)


def mostrar_analisis_cruzado(df):
    """Análisis cruzado entre variables clave"""

    st.markdown("#### Análisis de Relaciones entre Variables")

    # Análisis Sexo vs Ahorro vs Ingresos
    df_analysis = df.copy()
    df_analysis["Sexo"] = df_analysis["PCA1"].map({1: "Hombre", 2: "Mujer"})
    df_analysis["Ahorra"] = df_analysis["PCA6"].map({1: "No", 2: "Sí"})
    df_analysis["Ingresos"] = df_analysis["PCA5"].map(
        {
            1: "<$100",
            2: "$101-450",
            3: "$451-1800",
            4: "$1801-2500",
            5: "$2501-10000",
            6: "+$10000",
        }
    )

    # Tabla cruzada
    tabla_cruzada = pd.crosstab(
        [df_analysis["Sexo"], df_analysis["Ahorra"]],
        df_analysis["Ingresos"],
        margins=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tabla: Sexo × Ahorro × Nivel de Ingresos**")
        st.dataframe(tabla_cruzada)

    with col2:
        # Heatmap de correlaciones
        correlations = df[["PCA1", "PCA2", "PCA4", "PCA5", "PCA6", "PCA7"]].corr()

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=correlations.values,
                x=["Sexo", "Edad", "Estudios", "Ingresos", "Ahorro", "Est.Civil"],
                y=["Sexo", "Edad", "Estudios", "Ingresos", "Ahorro", "Est.Civil"],
                colorscale="RdBu",
                zmid=0,
                text=np.round(correlations.values, 2),
                texttemplate="%{text}",
                showscale=True,
            )
        )

        fig_corr.update_layout(title="Matriz de Correlaciones", height=400)
        st.plotly_chart(fig_corr, use_container_width=True)


def mostrar_tablas_detalladas(df):
    """Muestra tablas detalladas con filtros"""

    st.markdown("#### Exploración Detallada de Datos")

    # Filtros
    col_filter1, col_filter2, col_filter3 = st.columns(3)

    with col_filter1:
        sexo_filter = st.selectbox("Filtrar por sexo:", ["Todos", "Hombres", "Mujeres"])

    with col_filter2:
        ahorro_filter = st.selectbox(
            "Filtrar por ahorro:", ["Todos", "Sí ahorra", "No ahorra"]
        )

    with col_filter3:
        estudios_filter = st.selectbox(
            "Estudios mínimos:", ["Todos", "Universitario+", "Postgrado+"]
        )

    # Aplicar filtros
    df_filtered = df.copy()

    if sexo_filter == "Hombres":
        df_filtered = df_filtered[df_filtered["PCA1"] == 1]
    elif sexo_filter == "Mujeres":
        df_filtered = df_filtered[df_filtered["PCA1"] == 2]

    if ahorro_filter == "Sí ahorra":
        df_filtered = df_filtered[df_filtered["PCA6"] == 2]
    elif ahorro_filter == "No ahorra":
        df_filtered = df_filtered[df_filtered["PCA6"] == 1]

    if estudios_filter == "Universitario+":
        df_filtered = df_filtered[df_filtered["PCA4"] >= 4]
    elif estudios_filter == "Postgrado+":
        df_filtered = df_filtered[df_filtered["PCA4"] >= 5]

    # Mostrar resultados filtrados
    st.markdown(f"**Participantes que cumplen los criterios: {len(df_filtered)}**")

    if not df_filtered.empty:
        # Crear labels interpretables
        df_display = df_filtered.copy()
        df_display["Sexo"] = df_display["PCA1"].map({1: "Hombre", 2: "Mujer"})
        df_display["Edad"] = df_display["PCA2"].map(
            {
                1: "<26",
                2: "26-30",
                3: "31-35",
                4: "36-40",
                5: "41-45",
                6: "46-50",
                7: "51-55",
                8: "56-60",
                9: ">60",
            }
        )
        df_display["Estudios"] = df_display["PCA4"].map(
            {
                1: "Primaria",
                2: "Bachillerato",
                3: "TSU",
                4: "Universitario",
                5: "Postgrado",
                6: "Doctorado",
            }
        )
        df_display["Ingresos"] = df_display["PCA5"].map(
            {
                1: "$3-100",
                2: "$101-450",
                3: "$451-1800",
                4: "$1801-2500",
                5: "$2501-10000",
                6: "+$10000",
            }
        )
        df_display["Ahorra"] = df_display["PCA6"].map({1: "No", 2: "Sí"})
        df_display["Estado_Civil"] = df_display["PCA7"].map(
            {1: "Soltero/Div/Viudo", 2: "Casado/U.Libre"}
        )

        # Mostrar tabla interpretable
        display_cols = [
            "Item",
            "Sexo",
            "Edad",
            "Estudios",
            "Ingresos",
            "Ahorra",
            "Estado_Civil",
        ]
        st.dataframe(df_display[display_cols], use_container_width=True)

        # Estadísticas del subset
        col_stat1, col_stat2, col_stat3 = st.columns(3)

        with col_stat1:
            pct_mujeres = (df_filtered["PCA1"] == 2).mean() * 100
            st.metric("% Mujeres en subset", f"{pct_mujeres:.1f}%")

        with col_stat2:
            pct_ahorradores = (df_filtered["PCA6"] == 2).mean() * 100
            st.metric("% Ahorradores en subset", f"{pct_ahorradores:.1f}%")

        with col_stat3:
            ingreso_modal = (
                df_filtered["PCA5"].mode().iloc[0]
                if not df_filtered["PCA5"].mode().empty
                else 1
            )
            mapeo_ingresos = {
                1: "$3-100",
                2: "$101-450",
                3: "$451-1800",
                4: "$1801-2500",
                5: "$2501-10000",
                6: "+$10000",
            }
            st.metric(
                "Ingreso Modal",
                mapeo_ingresos.get(ingreso_modal, "N/A")
            )

        # Gráfico de distribución del subset filtrado
        st.markdown("#### Distribución del Subset Filtrado")
        
        # Crear gráfico de barras agrupadas
        fig_subset = go.Figure()

        # Distribución por sexo
        sexo_counts = df_filtered["PCA1"].value_counts()
        sexo_labels = [mapeo_ingresos.get(1, "Hombre") if i == 1 else "Mujer" for i in sexo_counts.index]
        
        fig_subset.add_trace(
            go.Bar(
                name="Distribución por Sexo",
                x=["Hombre" if i == 1 else "Mujer" for i in sexo_counts.index],
                y=sexo_counts.values,
                marker_color="#3498db"
            )
        )

        fig_subset.update_layout(
            title=f"Distribución del Subset Filtrado (N={len(df_filtered)})",
            xaxis_title="Categorías",
            yaxis_title="Frecuencia",
            height=400
        )

        st.plotly_chart(fig_subset, use_container_width=True)

    else:
        st.warning("No hay participantes que cumplan todos los criterios de filtrado seleccionados.")


def crear_estadisticas_adicionales(df):
    """Crea estadísticas descriptivas adicionales"""
    
    st.markdown("#### Estadísticas Descriptivas Adicionales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Estadísticas por Sexo**")
        stats_sexo = df.groupby("PCA1").agg({
            "PCA2": "mean",  # Edad promedio
            "PCA4": "mean",  # Nivel educativo promedio
            "PCA5": "mean",  # Ingresos promedio
            "PCA6": lambda x: (x == 2).mean()  # Proporción de ahorradores
        }).round(2)
        
        stats_sexo.index = ["Hombres", "Mujeres"]
        stats_sexo.columns = ["Edad Media", "Educación Media", "Ingresos Medios", "% Ahorradores"]
        
        st.dataframe(stats_sexo)
    
    with col2:
        st.markdown("**Estadísticas por Comportamiento de Ahorro**")
        stats_ahorro = df.groupby("PCA6").agg({
            "PCA2": "mean",  # Edad promedio
            "PCA4": "mean",  # Nivel educativo promedio
            "PCA5": "mean",  # Ingresos promedio
            "PCA1": lambda x: (x == 2).mean()  # Proporción de mujeres
        }).round(2)
        
        stats_ahorro.index = ["No Ahorradores", "Ahorradores"]
        stats_ahorro.columns = ["Edad Media", "Educación Media", "Ingresos Medios", "% Mujeres"]
        
        st.dataframe(stats_ahorro)