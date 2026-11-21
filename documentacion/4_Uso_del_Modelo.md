# Resultados y uso del modelo

Este documento reúne el funcionamiento descrito en la Fase 1 y la forma de utilizar la aplicación actual.

## 1. Resultados reportados en la Fase 1

La documentación original describe que, después del entrenamiento, la red logró representar la relación entre los requerimientos edafoclimáticos y las etiquetas de cultivo presentes en el dataset. También señala que las métricas fueron similares y que, en las pruebas realizadas entonces, se habían reducido los problemas de sobreajuste observados con datos aleatorios. Los resultados reportados fueron:

- **Precisión en entrenamiento (`accuracy`):** aproximadamente 96 %.
- **Precisión en validación (`val_accuracy`):** aproximadamente 95 %.

En esa entrega, la similitud entre ambas métricas se tomó como indicio de que el modelo había aprendido patrones del conjunto de datos sin mostrar una diferencia grande entre entrenamiento y validación. Estos valores corresponden a la ejecución de la Fase 1 y se conservan como referencia histórica.

## 2. Función de predicción de la Fase 1

En el notebook original se creó la función `recomendar_cultivo()` para recibir datos de un análisis de suelo y condiciones climáticas y devolver un cultivo.

El procedimiento descrito en la primera entrega es:

1. **Recepción de datos:** la persona introduce N, P, K, temperatura, humedad, pH y precipitación.
2. **Formato:** los siete valores se agrupan en una matriz NumPy de una fila por siete columnas.
3. **Escalamiento:** se utiliza el mismo objeto `StandardScaler` ajustado durante el entrenamiento, mediante `scaler.transform()`.
4. **Predicción:** el modelo recibe los datos escalados y devuelve 22 puntuaciones, una por cada cultivo.
5. **Selección:** con `np.argmax` se obtiene el índice de la salida con el valor más alto.
6. **Decodificación:** `label_encoder.inverse_transform()` convierte el índice seleccionado en el nombre del cultivo, por ejemplo, `chickpea`.

En la documentación de la Fase 1, las 22 salidas se describieron como porcentajes o probabilidades de cada cultivo. En la versión actual se aclara que la puntuación del modelo no representa la probabilidad de éxito de una cosecha.

En la primera versión, la función devuelve una sola recomendación.

## 3. Simulación regional incluida en la Fase 1

Como parte de las pruebas del PIA, el notebook incluyó una simulación para Monterrey, Nuevo León, dentro del noreste de México. La descripción original menciona suelos calcáreos, alcalinidad alta (pH 7.8), potasio alto y condiciones cálidas y semiáridas.

En el comentario del notebook aparece este conjunto de valores de referencia: N=20, P=120, K=200, temperatura=24 °C, humedad=80 %, pH=6.0 y lluvia=200 mm. La llamada ejecutada en la celda utiliza otros valores: `recomendar_cultivo(30, 45, 80, 28.5, 60.0, 7.8, 550.0)`. Se deja constancia de esta diferencia para que no se confundan el comentario y los datos enviados a la función.

## 4. Resultados del modelo incluido en la aplicación

El archivo `modelos/modelo_info.json` contiene las métricas del modelo que se distribuye con esta versión:

| Métrica | Resultado |
|---|---:|
| Exactitud de entrenamiento | 98.44 % |
| Exactitud de validación | 99.09 % |
| Exactitud de prueba independiente | 98.79 % |

La evaluación actual utiliza tres conjuntos: entrenamiento, validación y prueba independiente. Por esa razón, los resultados no se comparan directamente con los porcentajes aproximados de la Fase 1.

## 5. Cómo utilizar la aplicación

La aplicación se ejecuta localmente en Windows. No es necesario abrir el notebook para realizar una consulta.

### 5.1 Instalación

1. Extrae la carpeta del proyecto.
2. Ejecuta `INSTALAR_WINDOWS.bat`.
3. Espera a que se cree el entorno virtual `.venv`, se instalen las bibliotecas y termine la verificación.
4. Si el instalador informa que la instalación fue correcta, continúa con el siguiente paso.

El instalador busca una versión de Python de 64 bits entre 3.10 y 3.14. Para Python 3.10 a 3.13 utiliza TensorFlow 2.21.0. Para Python 3.14 utiliza TensorFlow 2.22.0rc1, versión preliminar. La opción más estable es utilizar Python 3.10 a 3.13.

### 5.2 Inicio

1. Ejecuta `INICIAR_APP.bat`.
2. Mantén abierta la ventana de comandos mientras utilices el sistema.
3. Abre `http://localhost:8501` en el navegador si no se abre automáticamente.
4. Para detener la aplicación, presiona `Ctrl+C` en la ventana de comandos.

### 5.3 Sección Recomendación

1. Selecciona un ejemplo del conjunto de datos o elige **Introducir valores**.
2. Captura los siete valores: nitrógeno, fósforo, potasio, temperatura, humedad relativa, pH y precipitación.
3. Presiona **Obtener recomendación**.
4. Consulta el cultivo con la puntuación más alta y las otras dos opciones del listado.
5. Si lo necesitas, utiliza **Descargar resultado** para guardar las entradas y las tres opciones en un archivo CSV.

El porcentaje mostrado es una puntuación de salida del modelo. No representa la probabilidad de éxito de una cosecha ni garantiza el rendimiento del cultivo.

La aplicación muestra advertencias cuando algún dato está fuera de los rangos observados en el dataset. Esto no sustituye la revisión de un análisis de suelo ni la asesoría de un especialista agrícola.

### 5.4 Otras secciones

- **Datos:** presenta el número de registros, el número de cultivos, la distribución por clase, promedios por variable, estadísticas descriptivas y una muestra del CSV.
- **Entrenamiento:** muestra la arquitectura, el número de parámetros, la distribución de los conjuntos y las métricas. También presenta las curvas de exactitud y pérdida por época cuando están guardadas en los metadatos.
- **Información:** resume el propósito del proyecto, la institución, las variables utilizadas, los archivos principales y las consideraciones de uso.

## 6. Archivos principales

| Archivo o carpeta | Función |
|---|---|
| `app.py` | Interfaz de la aplicación Streamlit. |
| `ml_core.py` | Carga del dataset, validaciones, escalamiento y predicción. |
| `entrenar_modelo.py` | Entrenamiento y guardado del modelo. |
| `redNeuronal.ipynb` | Notebook original de la Fase 1, conservado sin cambios. |
| `Crop_recommendation.csv` | Dataset agrícola original. |
| `modelos/red_neuronal_cultivos.keras` | Archivo de la red neuronal. |
| `modelos/modelo_info.json` | Etiquetas, medias, escalas, rangos, arquitectura y métricas. |
| `requirements.txt` | Bibliotecas necesarias para el proyecto. |
| `INSTALAR_WINDOWS.bat` | Crea el entorno e instala las dependencias. |
| `INICIAR_APP.bat` | Inicia la aplicación local. |
| `ENTRENAR_MODELO.bat` | Ejecuta nuevamente el entrenamiento. |

## 7. Alcances y limitaciones

El sistema clasifica registros de acuerdo con los patrones que aprendió del archivo utilizado. El dataset está relacionado con condiciones agrícolas de India. Antes de utilizar las recomendaciones para tomar decisiones de siembra en México, se necesita validarlas con análisis de suelo, información climática y resultados de campo de la región correspondiente.

La aplicación no calcula el rendimiento esperado, la rentabilidad ni la disponibilidad de agua. Tampoco sustituye el diagnóstico de un especialista agrícola. Sus resultados deben tomarse como una orientación basada en los datos de entrenamiento.
