@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul

echo ===========================================================
echo       INSTALACION DEL SISTEMA DE RECOMENDACION
echo ===========================================================
echo.
set "PYTHON_CMD="
set "PYTHON_VERSION="

for %%V in (3.14 3.13 3.12 3.11 3.10) do (
    if not defined PYTHON_CMD (
        py -%%V -c "import struct; assert struct.calcsize('P') * 8 == 64" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_CMD=py -%%V"
            set "PYTHON_VERSION=%%V"
        )
    )
)

if not defined PYTHON_CMD (
    python -c "import sys,struct; assert (3,10) <= sys.version_info[:2] < (3,15) and struct.calcsize('P') * 8 == 64" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        for /f "delims=" %%V in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set "PYTHON_VERSION=%%V"
    )
)

if not defined PYTHON_CMD (
    echo ERROR: No se encontro Python de 64 bits entre las versiones 3.10 y 3.14.
    echo Instala Python 3.10, 3.11, 3.12, 3.13 o 3.14 desde python.org.
    pause
    exit /b 1
)

echo Python detectado: !PYTHON_VERSION!
if "!PYTHON_VERSION!"=="3.14" (
    echo Se utilizara TensorFlow 2.22.0rc1 para Python 3.14.
    echo Esta version de TensorFlow es preliminar.
) else (
    echo Se utilizara TensorFlow 2.21.0 estable.
)
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; assert f'{sys.version_info.major}.{sys.version_info.minor}' == '!PYTHON_VERSION!'" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: La carpeta .venv fue creada con otra version de Python.
        echo Elimina solamente la carpeta .venv y vuelve a ejecutar este instalador.
        pause
        exit /b 1
    )
) else (
    echo Creando entorno virtual...
    %PYTHON_CMD% -m venv ".venv"
    if errorlevel 1 goto :failure
)

echo.
echo Actualizando pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failure

echo.
echo Instalando TensorFlow...
if "!PYTHON_VERSION!"=="3.14" (
    ".venv\Scripts\python.exe" -m pip install "tensorflow==2.22.0rc1"
) else (
    ".venv\Scripts\python.exe" -m pip install "tensorflow==2.21.0"
)
if errorlevel 1 goto :failure

echo.
echo Instalando las demas bibliotecas del proyecto...
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto :failure

echo.
echo Verificando bibliotecas, modelo y prediccion de prueba...
".venv\Scripts\python.exe" -c "import sys,numpy as np,pandas,seaborn,matplotlib,sklearn,streamlit,tensorflow as tf; from tensorflow.keras.models import load_model; from ml_core import MODEL_PATH,read_info; info=read_info(); model=load_model(MODEL_PATH,compile=False); assert model.input_shape[-1]==7 and model.output_shape[-1]==22==len(info['labels']); y=model.predict(np.zeros((1,7),dtype=np.float32),verbose=0); assert y.shape==(1,22) and np.isfinite(y).all(); print('Python:',sys.version.split()[0]); print('TensorFlow:',tf.__version__); print('Modelo verificado:',model.count_params(),'parametros'); print('Prediccion de prueba: correcta')"
if errorlevel 1 goto :failure

".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto :failure

echo.
echo INSTALACION CORRECTA.
echo Ejecuta INICIAR_APP.bat para abrir la aplicacion.
pause
exit /b 0

:failure
echo.
echo ERROR: No se completo la instalacion o la verificacion.
echo Conserva esta ventana y envia una captura del error.
pause
exit /b 1
