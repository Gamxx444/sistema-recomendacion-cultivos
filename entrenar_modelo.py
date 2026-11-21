"""Entrena la red neuronal original y guarda modelo + preprocesamiento.

Ejecutar desde la carpeta del proyecto:
    python entrenar_modelo.py
Opcional: python entrenar_modelo.py --epochs 50
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml_core import (
    DATA_PATH, FEATURES, INFO_PATH, MODEL_PATH, MODELS_DIR,
    dataset_sha256, load_dataset,
)


def build_model(num_features: int, num_classes: int):
    """Arquitectura original: 64 ReLU -> Dropout 0.20 -> 32 ReLU -> Softmax."""
    model = keras.Sequential(
        [
            keras.Input(shape=(num_features,), name="variables_suelo_clima"),
            layers.Dense(64, activation="relu", name="capa_oculta_64"),
            layers.Dropout(0.20, name="regularizacion_dropout"),
            layers.Dense(32, activation="relu", name="capa_oculta_32"),
            layers.Dense(num_classes, activation="softmax", name="cultivos_22"),
        ],
        name="modelo_recomendacion_cultivos",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(epochs: int = 50, batch_size: int = 32, seed: int = 42) -> dict:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs y batch_size deben ser enteros positivos.")
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    data = load_dataset()
    X = data.loc[:, FEATURES].copy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["label"])
    classes = encoder.classes_.tolist()
    if len(classes) != 22:
        raise ValueError(f"Se esperaban 22 cultivos, se encontraron {len(classes)}.")

    # 70 % entrenamiento, 15 % validación, 15 % prueba sin reutilizar test.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=seed, stratify=y_rest,
    )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train).astype("float32")
    val_scaled = scaler.transform(X_val).astype("float32")
    test_scaled = scaler.transform(X_test).astype("float32")

    model = build_model(len(FEATURES), len(classes))
    print(f"Dataset: {len(data)} filas; {len(classes)} cultivos.")
    print(f"Entrenamiento={len(X_train)}, Validación={len(X_val)}, Prueba={len(X_test)}")
    print(f"Parámetros entrenables: {model.count_params()}")
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__}")
    print(f"Entrenamiento: {epochs} épocas, lote {batch_size}.")
    history = model.fit(
        train_scaled, y_train,
        validation_data=(val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    # Evaluación *única* de prueba después del entrenamiento.
    test_loss, test_accuracy = model.evaluate(test_scaled, y_test, verbose=0)
    train_loss, train_accuracy = model.evaluate(train_scaled, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(val_scaled, y_val, verbose=0)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    info = {
        "schema_version": 1,
        "project": "Sistema inteligente para la recomendación de cultivos",
        "model_file": MODEL_PATH.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": list(FEATURES),
        "labels": classes,
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "ranges": {
            col: [float(data[col].min()), float(data[col].max())]
            for col in FEATURES
        },
        "dataset_sha256": dataset_sha256(DATA_PATH),
        "dataset_rows": len(data),
        "dataset_labels": len(classes),
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "architecture": "7 entradas -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU) -> Dense(22, Softmax)",
        "parameters": model.count_params(),
        "training": {"epochs": epochs, "batch_size": batch_size, "seed": seed},
        "metrics": {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        },
        "history": {key: [float(value) for value in vals] for key, vals in history.history.items()},
        "environment": {
            "python": platform.python_version(),
            "keras": keras.__version__,
            "keras_backend": "tensorflow",
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    with INFO_PATH.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("\nENTRENAMIENTO COMPLETADO")
    print(f"Exactitud en prueba independiente: {test_accuracy:.2%}")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Metadatos: {INFO_PATH}")
    return info


def main():
    parser = argparse.ArgumentParser(description="Entrena y guarda el modelo de recomendación de cultivos.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
