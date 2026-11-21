# Preprocesamiento de datos

Antes de introducir los valores a la red neuronal, los datos se limpian y transforman. El modelo necesita recibir información numérica y, para el entrenamiento, se utiliza `StandardScaler` para colocar las variables en una escala comparable.

Este documento conserva el procedimiento descrito para la Fase 1 y explica los cambios que se incorporaron en la aplicación actual.

## 1. Procedimiento de la Fase 1

En el notebook original `redNeuronal.ipynb` se realizaron los siguientes pasos:

### 1.1 Carga y estandarización de nombres de columnas

El archivo se carga con pandas y los nombres de las columnas se convierten a minúsculas para facilitar su uso y evitar diferencias entre mayúsculas y minúsculas:

```python
df = pd.read_csv('Crop_recommendation.csv')
df.columns = df.columns.str.lower()
```

### 1.2 Limpieza de datos

Se revisó la existencia de valores nulos con:

```python
df.isnull().sum()
```

Después se eliminaron las filas que contenían valores faltantes mediante `df.dropna()`. También se consultó la cantidad de registros por tipo de cultivo con `df['label'].value_counts()`.

### 1.3 Análisis exploratorio

El notebook incluye consultas y gráficas para revisar el comportamiento del dataset:

- Se calculó el promedio de temperatura y precipitación por cultivo.
- Se utilizaron tablas cruzadas (`pd.crosstab`) para revisar la relación entre el cultivo y la temperatura, así como entre el cultivo y el pH.
- Se seleccionaron las columnas numéricas y se calculó la correlación de Pearson.
- Se generó un mapa de calor de correlaciones con Seaborn.
- Se elaboraron diagramas de caja para comparar humedad, pH y temperatura por cultivo.

Estos análisis permiten observar la distribución de los datos y las relaciones entre las variables antes del entrenamiento.

### 1.4 Separación de variables (X e y)

Se separaron las variables independientes y la variable que se desea clasificar:

- **`X` (características):** `n`, `p`, `k`, `temperature`, `humidity`, `ph` y `rainfall`.
- **`y` (objetivo):** `label`, que contiene el tipo de cultivo.

### 1.5 Codificación de la variable objetivo

La red neuronal no trabaja directamente con etiquetas de texto como `rice` o `coffee`. Por eso se utilizó `LabelEncoder` de `scikit-learn` para convertir las 22 etiquetas en índices numéricos.

```python
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```

### 1.6 División de entrenamiento y prueba

En la Fase 1 se utilizó `train_test_split` con `test_size=0.2` y `random_state=42`:

- **80 % para entrenamiento:** datos utilizados para que la red aprenda los patrones.
- **20 % para prueba:** datos separados del entrenamiento para revisar el desempeño.

En el notebook original, ese 20 % se pasó al parámetro `validation_data` durante `model.fit`; por eso la métrica que aparece como `val_accuracy` corresponde a esa partición.

### 1.7 Escalamiento con StandardScaler

Las variables tienen escalas distintas. Por ejemplo, la precipitación puede estar en cientos, mientras que el pH suele tener valores cercanos a 6 o 7. Se utilizó `StandardScaler` para transformar las columnas numéricas y obtener una media de 0 y una desviación estándar de 1, de acuerdo con los datos de entrenamiento.

El escalador se ajustó (`fit`) únicamente con `X_train`; después se utilizó `transform` para convertir tanto los datos de entrenamiento como los de prueba. Así se evita utilizar información de la partición de prueba para calcular la media y la desviación estándar.

## 2. Procedimiento de la versión actual

La aplicación mantiene las siete variables, el uso de `LabelEncoder` y `StandardScaler`, pero separa los datos en tres conjuntos para distinguir la validación del entrenamiento y la evaluación final.

La preparación se encuentra en `ml_core.py` y `entrenar_modelo.py`:

1. Se carga `Crop_recommendation.csv` con pandas y se convierten los encabezados a minúsculas.
2. Se comprueba que existan las siete columnas de entrada y la columna `label`.
3. Se eliminan filas con valores nulos y se verifica que las variables numéricas sean válidas y finitas.
4. Se codifica `label` con `LabelEncoder`, conservando el orden de las 22 clases.
5. Se realiza una división estratificada con semilla 42: **70 % entrenamiento (1,540 registros), 15 % validación (330 registros) y 15 % prueba (330 registros)**.
6. `StandardScaler` se ajusta únicamente con los 1,540 registros de entrenamiento. La misma transformación se aplica a validación y prueba.
7. Se guardan las medias, escalas, etiquetas, rangos observados y datos de evaluación en `modelos/modelo_info.json`.

El conjunto de prueba se evalúa después del entrenamiento. No se utiliza como `validation_data` durante `model.fit`.

## 3. Preparación de datos al realizar una consulta

Cuando una persona utiliza la aplicación, `ml_core.predict_crop()`:

- comprueba que estén presentes los siete valores;
- verifica que sean numéricos y finitos;
- evita valores negativos en N, P, K y precipitación;
- limita la humedad al intervalo de 0 a 100 % y el pH al intervalo de 0 a 14;
- aplica las medias y escalas guardadas en `modelo_info.json`, sin volver a ajustar el escalador;
- compara los valores capturados con los rangos observados en el dataset y muestra una advertencia si alguno queda fuera de ellos.

La consulta no vuelve a entrenar la red neuronal. Para cambiar el modelo se debe ejecutar el procedimiento de entrenamiento.
