"""Funciones compartidas para el entrenamiento y la aplicación de cultivos.

La inferencia usa estadísticas guardadas en JSON. No necesita volver a ajustar
el escalador ni volver a entrenar el modelo para consultar un cultivo.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"
MODELS_DIR = BASE_DIR / "modelos"
MODEL_PATH = MODELS_DIR / "red_neuronal_cultivos.keras"
INFO_PATH = MODELS_DIR / "modelo_info.json"

FEATURES = ("n", "p", "k", "temperature", "humidity", "ph", "rainfall")
TARGET = "label"
SPANISH_NAMES = {
    "apple": "Manzana", "banana": "Plátano", "blackgram": "Frijol urad",
    "chickpea": "Garbanzo", "coconut": "Coco", "coffee": "Café",
    "cotton": "Algodón", "grapes": "Uva", "jute": "Yute",
    "kidneybeans": "Frijol rojo", "lentil": "Lenteja", "maize": "Maíz",
    "mango": "Mango", "mothbeans": "Frijol moth", "mungbean": "Frijol mungo",
    "muskmelon": "Melón", "orange": "Naranja", "papaya": "Papaya",
    "pigeonpeas": "Gandul", "pomegranate": "Granada", "rice": "Arroz",
    "watermelon": "Sandía",
}
FEATURE_NAMES = {
    "n": "Nitrógeno (N)", "p": "Fósforo (P)", "k": "Potasio (K)",
    "temperature": "Temperatura (°C)", "humidity": "Humedad relativa (%)",
    "ph": "pH del suelo", "rainfall": "Precipitación (mm)",
}


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga los datos con columnas en minúscula y en el orden esperado."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    required = list(FEATURES) + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {', '.join(missing)}")
    data = data.loc[:, required].dropna().copy()
    if data.empty:
        raise ValueError("El archivo CSV no contiene registros válidos.")
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="raise")
    if not np.isfinite(data.loc[:, FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("El dataset contiene valores numéricos no finitos.")
    data[TARGET] = data[TARGET].astype(str).str.strip()
    if (data[TARGET] == "").any():
        raise ValueError("Existen nombres de cultivo vacíos.")
    return data


def dataset_sha256(path: Path = DATA_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_info(path: Path = INFO_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            "No se encontró el archivo del modelo. Ejecuta ENTRENAR_MODELO.bat "
            "o python entrenar_modelo.py."
        )
    with path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    if info.get("features") != list(FEATURES):
        raise ValueError("El modelo guardado no coincide con las siete variables esperadas.")
    if len(info.get("labels", [])) != 22:
        raise ValueError("El modelo guardado no tiene las 22 etiquetas esperadas.")
    for key in ("mean", "scale", "ranges"):
        if key not in info:
            raise ValueError(f"Información del modelo incompleta: {key}")
    return info


def validate_input(values: dict) -> list[str]:
    """Valida la estructura y rangos físicos; los rangos del dataset se advierten aparte."""
    missing = [feat for feat in FEATURES if feat not in values]
    if missing:
        raise ValueError("Faltan valores: " + ", ".join(missing))
    for key in FEATURES:
        try:
            val = float(values[key])
        except (TypeError, ValueError) as e:
            raise ValueError(f"{FEATURE_NAMES[key]} debe ser numérico.") from e
        if not np.isfinite(val):
            raise ValueError(f"{FEATURE_NAMES[key]} debe ser finito.")
        if key in ("n", "p", "k", "rainfall") and val < 0:
            raise ValueError(f"{FEATURE_NAMES[key]} no puede ser negativo.")
        if key == "humidity" and not 0 <= val <= 100:
            raise ValueError("La humedad debe estar entre 0 y 100 %.")
        if key == "ph" and not 0 <= val <= 14:
            raise ValueError("El pH debe estar entre 0 y 14.")
    return []


def data_range_warnings(values: dict, info: dict) -> list[str]:
    """Advierte sobre valores fuera del intervalo observado al entrenar."""
    warnings = []
    for key in FEATURES:
        lo, hi = info["ranges"][key]
        val = float(values[key])
        if val < lo or val > hi:
            warnings.append(
                f"{FEATURE_NAMES[key]}: {val:g} está fuera del intervalo del dataset "
                f"({lo:.2f} a {hi:.2f})."
            )
    return warnings


def predict_crop(model, info: dict, values: dict, top_k: int = 3) -> list[dict]:
    """Aplica el mismo escalamiento aprendido y devuelve clases ordenadas."""
    validate_input(values)
    raw = np.array([[float(values[k]) for k in FEATURES]], dtype=np.float32)
    means = np.asarray(info["mean"], dtype=np.float32).reshape(1, -1)
    scales = np.asarray(info["scale"], dtype=np.float32).reshape(1, -1)
    if raw.shape != (1, 7) or means.shape != (1, 7) or scales.shape != (1, 7):
        raise ValueError("La entrada o la configuración de escalamiento son inválidas.")
    if np.any(scales <= 0):
        raise ValueError("El escalador guardado contiene una escala inválida.")
    scaled = (raw - means) / scales
    probs = np.asarray(model.predict(scaled, verbose=0), dtype=float).reshape(-1)
    if len(probs) != len(info["labels"]) or not np.isfinite(probs).all():
        raise ValueError("La red devolvió una salida incompatible con las etiquetas.")
    if np.any(probs < -1e-7) or not np.isclose(probs.sum(), 1, atol=1e-3):
        raise ValueError("Las probabilidades devueltas por el modelo no son válidas.")
    ranking = np.argsort(-probs)[: min(max(int(top_k), 1), len(probs))]
    return [
        {
            "label": str(info["labels"][int(i)]),
            "name_es": SPANISH_NAMES.get(str(info["labels"][int(i)]), str(info["labels"][int(i)])),
            "probability": float(probs[int(i)]),
        }
        for i in ranking
    ]
