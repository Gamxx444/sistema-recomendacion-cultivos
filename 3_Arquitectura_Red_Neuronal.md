# Arquitectura de la red neuronal

El modelo de la Fase 1 se construyó con TensorFlow y Keras mediante el modelo secuencial (`Sequential`). Se trata de una red neuronal de tipo *Feed Forward* para clasificación multiclase: recibe siete variables del suelo y del clima y clasifica el registro entre 22 tipos de cultivo.

La arquitectura original se conserva en la versión actual.

## 1. Estructura del modelo

| Etapa | Configuración | Función |
|---|---|---|
| Entrada | 7 valores | Recibe N, P, K, temperatura, humedad, pH y precipitación. |
| Capa oculta 1 | 64 neuronas, `ReLU` | Aprende relaciones entre las variables de entrada. |
| Regularización | `Dropout(0.2)` | Desactiva aleatoriamente el 20 % de las salidas de la capa anterior durante el entrenamiento para ayudar a reducir el sobreajuste. |
| Capa oculta 2 | 32 neuronas, `ReLU` | Aprende combinaciones de características de la capa anterior. |
| Capa de salida | 22 neuronas, `Softmax` | Produce una puntuación para cada uno de los 22 cultivos. |

La arquitectura tiene **3,318 parámetros entrenables**.

### Funciones de activación

- **ReLU (`Rectified Linear Unit`):** permite aprender relaciones no lineales y devuelve cero para entradas negativas.
- **Softmax:** transforma la salida de la última capa en valores que suman 1. La clase con el valor más alto es la primera opción del modelo.

## 2. Compilación y parámetros de aprendizaje

La red se compila con los siguientes elementos:

- **Optimizador (`adam`):** `Adaptive Moment Estimation`. Actualiza los pesos de la red durante el aprendizaje y adapta la tasa de aprendizaje para cada parámetro.
- **Función de pérdida (`sparse_categorical_crossentropy`):** mide la diferencia entre la clase real y la salida de la red. Se utiliza con etiquetas enteras, por lo que no es necesario convertirlas manualmente a formato *one-hot*.
- **Métrica (`accuracy`):** calcula la proporción de clasificaciones correctas.

En el notebook de la Fase 1 se entrenó durante **50 épocas** con `batch_size=32`. La documentación original mencionaba que el entrenamiento podía realizarse típicamente entre 50 y 100 épocas; el código entregado utiliza 50.

## 3. Evaluación en la Fase 1

En el notebook original se separó el 80 % de los registros para entrenamiento y el 20 % restante se proporcionó como `validation_data` durante `model.fit`. Por lo tanto, la métrica `val_accuracy` de esa ejecución se calculó con la partición del 20 % reservada, no con un tercer conjunto independiente.

## 4. Entrenamiento en la versión actual

El script `entrenar_modelo.py` conserva la arquitectura y utiliza TensorFlow/Keras. La separación actual es estratificada:

| Conjunto | Porcentaje | Registros |
|---|---:|---:|
| Entrenamiento | 70 % | 1,540 |
| Validación | 15 % | 330 |
| Prueba independiente | 15 % | 330 |

La configuración incluida utiliza 50 épocas, lotes de 32 registros y semilla 42. El conjunto de prueba se evalúa después de terminar el entrenamiento.

Las métricas guardadas para el modelo incluido son:

| Métrica | Resultado |
|---|---:|
| Exactitud de entrenamiento | 98.44 % |
| Exactitud de validación | 99.09 % |
| Exactitud de prueba independiente | 98.79 % |

Estos resultados corresponden al entrenamiento de la versión actual y no deben confundirse con los resultados reportados en la Fase 1. Las particiones utilizadas son distintas.

## 5. Bibliotecas y ejecución

El notebook original utiliza las siguientes bibliotecas:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow` / `tensorflow.keras`
- `matplotlib`
- `seaborn`

La aplicación agrega **Streamlit** para presentar el formulario y los resultados en el navegador. La lógica de lectura, validación y predicción se encuentra separada en `ml_core.py`.

El instalador incluido está preparado para Windows de 64 bits y busca Python 3.10, 3.11, 3.12, 3.13 o 3.14. Para Python 3.10 a 3.13 instala TensorFlow 2.21.0. Para Python 3.14 instala TensorFlow 2.22.0rc1, que es una versión preliminar. Para utilizar una versión estable de TensorFlow se recomienda Python 3.10 a 3.13. No se ofrece garantía para versiones anteriores a 3.10 ni posteriores a 3.14.

El archivo de modelo distribuido está en formato `.keras`. Los metadatos incluidos registran Keras 3.13.2 y backend Torch al momento de guardar ese archivo; la aplicación lo carga mediante TensorFlow/Keras. El script `entrenar_modelo.py` de esta carpeta utiliza TensorFlow/Keras para los entrenamientos que se realicen desde esta versión.
