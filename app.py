"""Interfaz local del sistema de recomendación de cultivos."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ml_core import (
    DATA_PATH, FEATURE_NAMES, FEATURES, MODEL_PATH, SPANISH_NAMES,
    data_range_warnings, dataset_sha256, load_dataset, predict_crop, read_info,
)

st.set_page_config(
    page_title="Recomendación de cultivos",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Colores de la aplicacion */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #F7F9F8 !important;
        color: #193C36 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #E9F2ED !important;
        color: #193C36 !important;
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stMarkdownContainer"] h4,
    .stApp [data-testid="stMarkdownContainer"] h5,
    .stApp [data-testid="stMarkdownContainer"] h6 {
        color: #27453F !important;
    }
    .stApp [data-testid="stMarkdownContainer"] h1,
    .stApp [data-testid="stMarkdownContainer"] h2,
    .stApp [data-testid="stMarkdownContainer"] h3 {
        color: #183F3A !important;
    }

    .stApp .hero h1 { color: #FFFFFF !important; }
    .stApp .hero p { color: #E5F1ED !important; }
    .stApp .hero .eyebrow { color: #B7D6CC !important; }

    .stApp .highlight .result-name { color: #184F43 !important; }
    .stApp .highlight .fineprint { color: #48635A !important; }
    .stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #23483E !important; }
    .stApp [data-testid="stSidebar"] [data-testid="stAlert"] p { color: inherit !important; }
    .block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #183F3A; letter-spacing: -.02em; }
    .hero { background: linear-gradient(110deg, #103F35, #1B6555); border-radius: 14px;
            color: #FFFFFF; padding: 1.65rem 1.9rem; margin-bottom: 1.2rem; }
    .hero .eyebrow { font-size: .75rem; letter-spacing: .16em; color: #B7D6CC; margin-bottom: .5rem; }
    .hero h1 { color: #FFFFFF; margin: .2rem 0 .6rem 0; font-size: 2rem; }
    .hero p { color: #E5F1ED; margin: 0; line-height: 1.55; }
    .highlight { background: #EAF4EF; border: 1px solid #C9DED4; padding: 1rem 1.2rem;
                 border-radius: 10px; margin-bottom: 1rem; }
    .result-name { color: #184F43; font-size: 2.1rem; font-weight: 750; margin: 0; }
    .fineprint { color: #667B74; font-size: .85rem; line-height: 1.5; }
    [data-testid="stMetricValue"] { color: #1B5C4C; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return load_dataset()

@st.cache_resource(show_spinner="Cargando red neuronal entrenada…")
def load_model_bundle():
    # Se carga con TensorFlow/Keras, igual que en el notebook original.
    from tensorflow.keras.models import load_model

    info = read_info()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    if tuple(model.input_shape[1:]) != (7,) or model.output_shape[-1] != len(info["labels"]):
        raise ValueError("El archivo de red neuronal no coincide con sus metadatos.")
    return model, info


def result_csv(values: dict, results: list[dict]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(["Variable", "Valor"])
    for key in FEATURES:
        writer.writerow([FEATURE_NAMES[key], values[key]])
    writer.writerow([])
    writer.writerow(["Posición", "Cultivo", "Clase original", "Probabilidad del modelo (%)"])
    for i, row in enumerate(results, 1):
        writer.writerow([i, row["name_es"], row["label"], f'{row["probability"] * 100:.2f}'])
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def header():
    st.markdown(
        """
        <div class="hero">
          <h1>Recomendación de cultivos</h1>
          <p>Ingresa los datos del suelo y del clima para consultar las opciones.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_page(model, info, df):
    st.subheader("Consultar cultivo")
    st.write("Ingresa los valores del suelo y del clima, o selecciona un ejemplo.")
    presets = {
        "Ejemplo: arroz": "rice",
        "Ejemplo: maíz": "maize",
        "Ejemplo: plátano": "banana",
        "Introducir valores": None,
    }

    def reset_result():
        st.session_state.pop("last_result", None)

    preset = st.selectbox(
        "Valores de ejemplo", list(presets),
        on_change=reset_result, key="selected_preset",
        help="Cada ejemplo es un registro del archivo CSV.",
    )
    selected_label = presets[preset]
    if selected_label:
        sample = df.loc[df["label"] == selected_label, FEATURES].iloc[0]
    else:
        sample = df.loc[:, FEATURES].median()

    with st.form(key=f"crop_form_{preset}", clear_on_submit=False):
        st.markdown("#### Datos del suelo")
        a, b, c = st.columns(3)
        with a:
            n = st.number_input("Nitrógeno (N)", min_value=0.0, value=float(sample["n"]), step=1.0)
        with b:
            p = st.number_input("Fósforo (P)", min_value=0.0, value=float(sample["p"]), step=1.0)
        with c:
            k = st.number_input("Potasio (K)", min_value=0.0, value=float(sample["k"]), step=1.0)
        st.caption("Para N, P y K utiliza la misma escala que aparece en el archivo de datos.")
        st.markdown("#### Condiciones ambientales")
        a, b, c, d = st.columns(4)
        with a:
            temperature = st.number_input("Temperatura (°C)", value=float(sample["temperature"]), format="%.2f", step=0.5)
        with b:
            humidity = st.number_input("Humedad relativa (%)", min_value=0.0, max_value=100.0,
                                       value=float(sample["humidity"]), format="%.2f", step=1.0)
        with c:
            ph = st.number_input("pH del suelo", min_value=0.0, max_value=14.0,
                                 value=float(sample["ph"]), format="%.2f", step=0.1)
        with d:
            rainfall = st.number_input("Precipitación (mm)", min_value=0.0,
                                       value=float(sample["rainfall"]), format="%.2f", step=5.0)
        submitted = st.form_submit_button("Obtener recomendación", type="primary", use_container_width=True)

    if submitted:
        values = {
            "n": n, "p": p, "k": k, "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall,
        }
        try:
            results = predict_crop(model, info, values, top_k=3)
            warnings = data_range_warnings(values, info)
            st.session_state["last_result"] = {
                "values": values, "results": results, "warnings": warnings,
            }
        except (ValueError, RuntimeError) as exc:
            st.error(f"No fue posible procesar los datos: {exc}")
            st.session_state.pop("last_result", None)

    last = st.session_state.get("last_result")
    if last:
        st.divider()
        st.subheader("Resultado")
        if last["warnings"]:
            st.warning("Hay valores fuera de los rangos del conjunto de datos. Revisa lo siguiente:\n\n" +
                       "\n\n".join("• " + w for w in last["warnings"]))
        top = last["results"][0]
        left, right = st.columns([1.7, 1], gap="large")
        with left:
            st.markdown("<div class='highlight'><div class='fineprint'>Cultivo recomendado</div>"
                        f"<div class='result-name'>{top['name_es']}</div>"
                        f"<div class='fineprint'>Nombre en el archivo: {top['label'].upper()}</div></div>",
                        unsafe_allow_html=True)
            st.write("**Tres opciones**")
            for idx, item in enumerate(last["results"], 1):
                st.write(f"{idx}. **{item['name_es']}** · {item['probability'] * 100:.2f} %")
                st.progress(min(1.0, max(0.0, item["probability"])))
        with right:
            st.metric("Puntuación del modelo", f"{top['probability'] * 100:.2f} %")
            st.caption("Este porcentaje es una salida del modelo; no indica el éxito de una cosecha.")
            st.download_button(
                "Descargar resultado",
                data=result_csv(last["values"], last["results"]),
                file_name="recomendacion_cultivos.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.caption("Para decisiones de siembra, verifica los datos con un especialista agrícola.")
    else:
        st.info("Ingresa los datos para consultar un cultivo.")


def data_page(info, df):
    st.subheader("Datos del proyecto")
    a, b, c = st.columns(3)
    a.metric("Registros", f"{len(df):,}")
    b.metric("Cultivos", f"{df['label'].nunique()}")
    c.metric("Variables", str(len(FEATURES)))
    tab1, tab2, tab3 = st.tabs(["Distribución", "Variables", "Muestra del archivo"])
    with tab1:
        st.write("Cantidad de registros por cultivo")
        counts = df["label"].value_counts().sort_index()
        counts.index = [SPANISH_NAMES.get(k, k) for k in counts.index]
        st.bar_chart(counts.rename("Registros"), height=370)
    with tab2:
        feature = st.selectbox("Variable a comparar", FEATURES,
                               format_func=lambda v: FEATURE_NAMES[v])
        grouped = df.groupby("label", as_index=False)[feature].mean()
        grouped["cultivo"] = grouped["label"].map(lambda k: SPANISH_NAMES.get(k, k))
        st.write(f"Promedio de **{FEATURE_NAMES[feature]}** por cultivo")
        st.bar_chart(grouped.set_index("cultivo")[[feature]], height=360)
        st.write("Estadísticas del conjunto de datos")
        st.dataframe(df.loc[:, FEATURES].describe().round(2), use_container_width=True)
    with tab3:
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        st.caption("Se muestran los primeros 20 registros del archivo CSV.")


def model_page(info):
    st.subheader("Entrenamiento")
    st.write("Red neuronal con siete entradas y 22 cultivos posibles.")
    m = info.get("metrics", {})
    a, b, c = st.columns(3)
    a.metric("Exactitud · entrenamiento", f"{m.get('train_accuracy', 0) * 100:.2f} %")
    b.metric("Exactitud · validación", f"{m.get('val_accuracy', 0) * 100:.2f} %")
    c.metric("Exactitud · prueba independiente", f"{m.get('test_accuracy', 0) * 100:.2f} %")
    st.caption("Métricas del modelo que utiliza esta aplicación.")
    st.markdown("#### Arquitectura")
    st.code(info.get("architecture", ""), language=None)
    st.write(f"**Parámetros entrenables:** {info.get('parameters', '—'):,}")
    split = info.get("split", {})
    st.write("**Distribución de datos**")
    st.dataframe(
        pd.DataFrame([
            {"Conjunto": "Entrenamiento (70 %)", "Registros": split.get("train")},
            {"Conjunto": "Validación (15 %)", "Registros": split.get("validation")},
            {"Conjunto": "Prueba (15 %)", "Registros": split.get("test")},
        ]), hide_index=True, use_container_width=True,
    )
    hist = info.get("history", {})
    if "accuracy" in hist and "val_accuracy" in hist:
        st.markdown("#### Exactitud por época")
        frame = pd.DataFrame({
            "Entrenamiento": hist["accuracy"],
            "Validación": hist["val_accuracy"],
        }, index=range(1, len(hist["accuracy"]) + 1))
        frame.index.name = "Época"
        st.line_chart(frame, height=330)
    if "loss" in hist and "val_loss" in hist:
        st.markdown("#### Pérdida por época")
        frame = pd.DataFrame({
            "Entrenamiento": hist["loss"],
            "Validación": hist["val_loss"],
        }, index=range(1, len(hist["loss"]) + 1))
        frame.index.name = "Época"
        st.line_chart(frame, height=330)
    st.caption("La evaluación corresponde a registros del mismo conjunto de datos. Falta validación con datos de campo de México.")


def about_page(info, df):
    """Muestra la información general del proyecto sin repetir el encabezado."""
    st.title("Información del proyecto")
    st.write("Sistema de recomendación de cultivos mediante una red neuronal artificial.")
    st.divider()

    st.subheader("Datos generales")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Institución**")
            st.write("Universidad Autónoma de Nuevo León")
            st.write("Facultad de Ingeniería Mecánica y Eléctrica")
            st.markdown("**Materia**")
            st.write("Redes Neuronales Artificiales")
    with right:
        with st.container(border=True):
            st.markdown("**Equipo**")
            st.write("Equipo 6")
            st.markdown("**Tema del proyecto**")
            st.write("Clasificación de cultivos a partir de condiciones del suelo y del clima.")

    st.subheader("Descripción")
    st.write(
        "El proyecto utiliza una red neuronal para clasificar un registro agrícola "
        "de acuerdo con siete variables relacionadas con el suelo y el clima. "
        "La clasificación se realiza con base en los patrones presentes en el "
        "conjunto de datos Crop_recommendation.csv."
    )

    st.subheader("Datos utilizados")
    a, b, c = st.columns(3)
    a.metric("Registros", f"{len(df):,}")
    b.metric("Tipos de cultivo", f"{df['label'].nunique()}")
    c.metric("Variables de entrada", str(len(FEATURES)))

    variable_rows = [
        {"Variable": "Nitrógeno (N)", "Dato utilizado": "Contenido de nitrógeno del suelo"},
        {"Variable": "Fósforo (P)", "Dato utilizado": "Contenido de fósforo del suelo"},
        {"Variable": "Potasio (K)", "Dato utilizado": "Contenido de potasio del suelo"},
        {"Variable": "Temperatura", "Dato utilizado": "Temperatura en grados Celsius (°C)"},
        {"Variable": "Humedad", "Dato utilizado": "Humedad relativa (%)"},
        {"Variable": "pH", "Dato utilizado": "Nivel de acidez o alcalinidad del suelo"},
        {"Variable": "Precipitación", "Dato utilizado": "Cantidad de lluvia en milímetros (mm)"},
    ]
    st.dataframe(pd.DataFrame(variable_rows), use_container_width=True, hide_index=True)
    st.caption(
        "El archivo contiene 2,200 registros distribuidos en 22 cultivos, "
        "con 100 registros por clase. Para N, P y K se utiliza la misma escala numérica del dataset."
    )

    st.subheader("Funcionamiento")
    steps = [
        ("1. Captura de datos", "Se ingresan los siete valores del suelo y del clima."),
        ("2. Preparación", "Los valores se organizan y se escalan con el StandardScaler guardado durante el entrenamiento."),
        ("3. Clasificación", "La red neuronal procesa los datos y genera una puntuación para cada uno de los 22 cultivos."),
        ("4. Resultado", "La aplicación muestra las tres opciones con mayor puntuación."),
    ]
    for title, description in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    st.subheader("Estructura de la red neuronal")
    architecture_rows = [
        {"Etapa": "Entrada", "Configuración": "7 variables", "Función": "Recibe los datos del suelo y del clima."},
        {"Etapa": "Capa oculta 1", "Configuración": "64 neuronas · ReLU", "Función": "Procesa las relaciones entre las variables."},
        {"Etapa": "Regularización", "Configuración": "Dropout 0.20", "Función": "Desactiva el 20 % durante el entrenamiento."},
        {"Etapa": "Capa oculta 2", "Configuración": "32 neuronas · ReLU", "Función": "Continúa el procesamiento de los datos."},
        {"Etapa": "Salida", "Configuración": "22 neuronas · Softmax", "Función": "Entrega una puntuación para cada cultivo."},
    ]
    st.dataframe(pd.DataFrame(architecture_rows), use_container_width=True, hide_index=True)
    st.write(
        f"**Parámetros entrenables:** {info.get('parameters', 3318):,}. "
        "La configuración de entrenamiento utiliza Adam, "
        "sparse_categorical_crossentropy, 50 épocas y lotes de 32 registros."
    )

    st.subheader("Resultados de evaluación")
    metrics = info.get("metrics", {})
    current_cols = st.columns(3)
    current_cols[0].metric("Entrenamiento", f"{metrics.get('train_accuracy', 0) * 100:.2f} %")
    current_cols[1].metric("Validación", f"{metrics.get('val_accuracy', 0) * 100:.2f} %")
    current_cols[2].metric("Prueba", f"{metrics.get('test_accuracy', 0) * 100:.2f} %")
    st.caption("Resultados del modelo incluido en esta versión. La prueba se realizó con una partición separada del entrenamiento.")
    with st.expander("Referencia de la primera entrega"):
        st.write(
            "En la Fase 1 se reportaron aproximadamente 96 % de exactitud en entrenamiento "
            "y 95 % en validación. En esa versión se utilizó una división 80/20. "
            "Los resultados actuales corresponden a una nueva división 70/15/15, "
            "por lo que no deben compararse como si procedieran de la misma ejecución."
        )

    st.subheader("Archivos principales")
    files_rows = [
        {"Archivo": "app.py", "Uso": "Interfaz de la aplicación."},
        {"Archivo": "ml_core.py", "Uso": "Carga de datos, preparación y predicción."},
        {"Archivo": "entrenar_modelo.py", "Uso": "Entrenamiento y guardado de la red neuronal."},
        {"Archivo": "redNeuronal.ipynb", "Uso": "Notebook original de la primera entrega."},
        {"Archivo": "Crop_recommendation.csv", "Uso": "Conjunto de datos agrícola."},
        {"Archivo": "modelos/", "Uso": "Modelo entrenado y sus metadatos."},
        {"Archivo": "documentacion/", "Uso": "Documentos sobre variables, preprocesamiento, arquitectura y uso."},
    ]
    st.dataframe(pd.DataFrame(files_rows), use_container_width=True, hide_index=True)

    st.subheader("Alcance y consideraciones")
    st.write(
        "La aplicación clasifica los registros de acuerdo con los patrones aprendidos del dataset. "
        "No calcula el rendimiento ni la rentabilidad de una cosecha y no sustituye un análisis de suelo "
        "o la asesoría de un especialista agrícola."
    )
    st.warning(
        "El conjunto de datos está relacionado con condiciones agrícolas de India. "
        "Antes de utilizar las recomendaciones en México, se requiere validarlas con datos y mediciones de campo de la región."
    )


def main():
    try:
        df = cached_dataset()
        model, info = load_model_bundle()
    except (FileNotFoundError, ValueError, ImportError, OSError, RuntimeError) as exc:
        st.error("No se pudo cargar el sistema.")
        st.info("Revisa que estén instaladas las dependencias y que la carpeta modelos contenga los dos archivos del modelo.")
        with st.expander("Detalle del error"):
            st.code(str(exc))
        st.stop()
    page = st.sidebar.radio(
        "Secciones",
        ("Recomendación", "Datos", "Entrenamiento", "Información"),
    )
    if DATA_PATH.exists() and dataset_sha256() != info.get("dataset_sha256"):
        st.sidebar.warning("El archivo CSV cambió. Para actualizar el modelo, ejecuta ENTRENAR_MODELO.bat.")
    if page != "Información":
        header()
    if page == "Recomendación":
        recommendation_page(model, info, df)
    elif page == "Datos":
        data_page(info, df)
    elif page == "Entrenamiento":
        model_page(info)
    else:
        about_page(info, df)


if __name__ == "__main__":
    main()
