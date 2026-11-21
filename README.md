# Sistema de recomendación de cultivos

Proyecto académico de clasificación de cultivos mediante una red neuronal. El sistema utiliza siete variables del suelo y del clima para clasificar registros entre 22 tipos de cultivo.

## Contenido

- `redNeuronal.ipynb`: notebook original de la Fase 1. Se conserva sin modificaciones.
- `Crop_recommendation.csv`: conjunto de datos original de 2,200 registros.
- `app.py`: interfaz local desarrollada con Streamlit.
- `ml_core.py`: carga, validación y predicción.
- `entrenar_modelo.py`: entrenamiento de la red neuronal.
- `modelos/`: modelo guardado y metadatos.
- `documentacion/`: descripción del dataset, preprocesamiento, arquitectura y uso.

## Requisitos

Windows de 64 bits y Python 3.10 a 3.14. El instalador selecciona TensorFlow 2.21.0 para Python 3.10–3.13 y TensorFlow 2.22.0rc1 (versión preliminar) para Python 3.14. Para utilizar una versión estable de TensorFlow, se recomienda Python 3.10–3.13.

Las bibliotecas del notebook original se mantienen: pandas, NumPy, scikit-learn, TensorFlow/Keras, Matplotlib y Seaborn. Streamlit se agrega para la interfaz web local.

## Instalación y ejecución en Windows

1. Extrae la carpeta del proyecto.
2. Ejecuta `INSTALAR_WINDOWS.bat`.
3. Cuando finalice correctamente, ejecuta `INICIAR_APP.bat`.
4. Abre `http://localhost:8501` si el navegador no se abre automáticamente.

Para detener el servidor, presiona `Ctrl+C` en la ventana de comandos.

Para volver a entrenar el modelo, ejecuta `ENTRENAR_MODELO.bat`.

## Documentación

Los cuatro documentos de `documentacion/` conservan la información de la Fase 1 y agregan, en apartados separados, los cambios de la aplicación actual.

## Alcance

Las recomendaciones se basan en el dataset incluido. Antes de utilizar el resultado para decisiones de siembra en México, se requiere validación con datos y mediciones de campo locales. La puntuación del modelo no garantiza el éxito de una cosecha.
