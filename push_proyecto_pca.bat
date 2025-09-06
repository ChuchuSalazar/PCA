@echo off
REM push_proyecto_pca.bat
REM Uso: push_proyecto_pca.bat "Mensaje de commit"

IF "%~1"=="" (
    echo Error: Debes proporcionar un mensaje de commit.
    echo Uso: push_proyecto_pca.bat "Mensaje de commit"
    exit /b 1
)

echo ==== Estado actual de Git ====
git status

echo ==== Agregando cambios ====
git add .

echo ==== Haciendo commit ====
git commit -m "%~1"

echo ==== Subiendo cambios a GitHub ====
git push origin main

echo ==== ¡Listo! Cambios subidos correctamente ====
pause
