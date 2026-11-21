@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if not exist ".venv\Scripts\python.exe" (
    echo Primero ejecuta INSTALAR_WINDOWS.bat
    pause
    exit /b 1
)
if not exist "modelos\red_neuronal_cultivos.keras" (
    echo No se encuentra el modelo entrenado.
    echo Ejecuta ENTRENAR_MODELO.bat y luego vuelve a abrir la aplicacion.
    pause
    exit /b 1
)
if not exist "modelos\modelo_info.json" (
    echo Falta modelos\modelo_info.json
    echo Ejecuta ENTRENAR_MODELO.bat
    pause
    exit /b 1
)
echo Iniciando la aplicacion en http://localhost:8501
echo Para detenerlo, cierra esta ventana o presiona Ctrl+C.
".venv\Scripts\python.exe" -m streamlit run "app.py" --server.address 127.0.0.1 --server.port 8501
if errorlevel 1 (
    echo.
    echo Hubo un error al iniciar. Comprueba los archivos y la instalacion.
    pause
    exit /b 1
)
