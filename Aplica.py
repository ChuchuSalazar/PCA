import streamlit as st
import os
import ast

# Función para mapear la estructura del proyecto


def map_project_structure(base_path):
    structure = {}
    for root, dirs, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)
        structure[rel_path] = {
            'folders': dirs,
            'py_files': [f for f in files if f.endswith('.py')]
        }
    return structure

# Función para detectar fallos comunes


def detect_issues(base_path, structure):
    issues = []
    for path, content in structure.items():
        full_path = os.path.join(base_path, path)

        # Verificar si es un paquete sin __init__.py
        if content['py_files'] and '__init__.py' not in content['py_files']:
            issues.append(f"Falta __init__.py en el paquete: {path}")

        # Verificar funciones sin docstring
        for py_file in content['py_files']:
            file_path = os.path.join(full_path, py_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=py_file)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if not ast.get_docstring(node):
                                issues.append(
                                    f"Función sin docstring: {node.name} en {path}/{py_file}")
            except Exception as e:
                issues.append(f"Error al analizar {path}/{py_file}: {str(e)}")

    return issues


# Interfaz de Streamlit
st.title("🧠 Mapeo y Análisis de Proyecto Python Modular")

base_path = st.text_input("📂 Ruta del proyecto:", value=".")

if st.button("🔍 Escanear Proyecto"):
    structure = map_project_structure(base_path)
    issues = detect_issues(base_path, structure)

    st.subheader("📦 Estructura del Proyecto")
    for path, content in structure.items():
        st.markdown(f"**{path}/**")
        for folder in content['folders']:
            st.markdown(f"- 📁 {folder}/")
        for py_file in content['py_files']:
            st.markdown(f"- 📄 {py_file}")

    st.subheader("⚠️ Problemas Detectados")
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("✅ No se detectaron problemas.")

    # Generar archivo TXT con el mapeo
    with open("reporte_mapeo.txt", "w", encoding="utf-8") as f:
        f.write("MAPEO DE ESTRUCTURA DEL PROYECTO PYTHON\n")
        f.write("=======================================\n\n")
        for path, content in structure.items():
            f.write(f"{path}/\n")
            for folder in content['folders']:
                f.write(f"  └── [DIR] {folder}/\n")
            for py_file in content['py_files']:
                f.write(f"  └── [PY]  {py_file}\n")

        f.write("\nPROBLEMAS DETECTADOS\n")
        f.write("====================\n")
        if issues:
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write("No se detectaron problemas.\n")

    st.success("📄 Archivo 'reporte_mapeo.txt' generado correctamente.")
