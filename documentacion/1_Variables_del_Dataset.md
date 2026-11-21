# Variables del dataset agrícola (`Crop_recommendation.csv`)

## Descripción general

El archivo `Crop_recommendation.csv` contiene **2,200 registros** correspondientes a **22 tipos de cultivo**. Hay 100 registros por cada cultivo, por lo que el conjunto está balanceado.

El dataset reúne variables edafoclimáticas, es decir, datos relacionados con el suelo y el clima. La red neuronal utiliza siete variables de entrada para clasificar el tipo de cultivo registrado en la columna `label`.

## Características de entrada (X)

Estas son las variables ambientales y químicas que utiliza el modelo para aprender y realizar sus predicciones.

| N.º | Columna | Descripción | Importancia indicada en la documentación original | Unidad o formato |
|---:|---|---|---|---|
| 1 | `n` (Nitrógeno) | Proporción de nitrógeno en el suelo. | Esencial para el crecimiento de las hojas y el desarrollo general de la planta. | Numérico entero (ratio). |
| 2 | `p` (Fósforo) | Proporción de fósforo en el suelo. | Crucial para el desarrollo de raíces, flores y frutos. En el análisis exploratorio se observó una correlación positiva de aproximadamente **0.74** con el potasio en este dataset. | Numérico entero (ratio). |
| 3 | `k` (Potasio) | Proporción de potasio en el suelo. | Ayuda a la resistencia a enfermedades y a la absorción de agua. | Numérico entero (ratio). |
| 4 | `temperature` (Temperatura) | Temperatura promedio durante el ciclo de crecimiento del cultivo. | Es una de las condiciones ambientales consideradas por el conjunto de datos. | Grados Celsius (°C), decimal continuo. |
| 5 | `humidity` (Humedad) | Humedad relativa promedio del ambiente. | Es una de las condiciones ambientales consideradas por el conjunto de datos. | Porcentaje (%), decimal continuo. |
| 6 | `ph` (Potencial de hidrógeno) | Nivel de acidez o alcalinidad del suelo. | Los valores menores a 7 indican acidez y los mayores a 7 indican alcalinidad. Los cultivos tienen distintos niveles de tolerancia al pH. | Escala de pH de 0 a 14, decimal continuo. |
| 7 | `rainfall` (Precipitación o lluvia) | Cantidad total de lluvia recibida durante el ciclo de vida del cultivo. | Es una condición climática considerada para clasificar los cultivos. | Milímetros (mm), decimal continuo. |

**Nota sobre N, P y K:** el archivo maneja valores numéricos para estos nutrientes. La documentación original los describe como proporciones y no especifica una unidad de laboratorio distinta. Por ello, en la aplicación se deben introducir utilizando la misma escala numérica del dataset.

## Variable objetivo (y)

La columna `label` identifica el tipo de cultivo asociado con cada registro. En la documentación de la Fase 1 se describe como el cultivo que fue sembrado y prosperó bajo las condiciones registradas.

Los 22 valores posibles son:

| Etiqueta del archivo | Nombre en español |
|---|---|
| `rice` | Arroz |
| `maize` | Maíz |
| `chickpea` | Garbanzo |
| `kidneybeans` | Frijoles |
| `pigeonpeas` | Gandules |
| `mothbeans` | Frijol tepari/moth |
| `mungbean` | Frijol mungo |
| `blackgram` | Lenteja negra |
| `lentil` | Lenteja |
| `pomegranate` | Granada |
| `banana` | Plátano |
| `mango` | Mango |
| `grapes` | Uvas |
| `watermelon` | Sandía |
| `muskmelon` | Melón |
| `apple` | Manzana |
| `orange` | Naranja |
| `papaya` | Papaya |
| `coconut` | Coco |
| `cotton` | Algodón |
| `jute` | Yute |
| `coffee` | Café |

La aplicación muestra nombres en español para facilitar la consulta. Algunas traducciones visibles en la interfaz pueden ser más específicas que las utilizadas en la lista anterior; la etiqueta original en inglés se conserva en los datos y en el archivo de resultados.

## Uso de las variables en la aplicación actual

La pantalla de recomendación solicita los siete valores de entrada: nitrógeno, fósforo, potasio, temperatura, humedad relativa, pH y precipitación. El sistema prepara los valores con el escalamiento guardado durante el entrenamiento y los envía a la red neuronal. La salida corresponde a una clasificación entre los 22 cultivos.

## Consideraciones sobre el conjunto de datos

- El archivo original tiene 2,200 registros y 22 clases, con 100 registros por clase.
- Los valores de N, P y K deben capturarse con la misma escala del archivo.
- El conjunto de datos utilizado está relacionado con condiciones agrícolas de India. Las recomendaciones para México requieren validación con datos y mediciones de campo de la región donde se pretenda utilizar el sistema.
- La etiqueta del dataset representa la clase registrada; por sí sola no demuestra que el cultivo vaya a tener éxito en cualquier terreno con valores similares.
