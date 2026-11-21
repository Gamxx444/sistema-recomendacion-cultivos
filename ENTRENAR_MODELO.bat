@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if not exist ".venv\Scripts\python.exe" (
    echo Primero ejecuta INSTALAR_WINDOWS.bat
    pause
    exit /b 1
)
echo Reentrenando modelo con 50 epocas. Se reemplazaran los archivos de modelos.
".venv\Scripts\python.exe" "entrenar_modelo.py" --epochs 50 --batch-size 32
if errorlevel 1 (
    echo ERROR: El entrenamiento no termino.
    pause
    exit /b 1
)
echo Modelo actualizado. Puedes abrir INICIAR_APP.bat
pause
