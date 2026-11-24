
import os
import numpy as np
from tensorflow.keras.models import load_model

from backend.inference_preprocessing_fast import preprocess_fast
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "src",
    "model",
    "lstm_model"
)

MODEL_PATHS = {
    0.10: os.path.join(MODEL_DIR, "lstm_10_model.keras"),
    0.25: os.path.join(MODEL_DIR, "lstm_25_model.keras"),
    0.50: os.path.join(MODEL_DIR, "lstm_50_model.keras"),
    0.75: os.path.join(MODEL_DIR, "lstm_75_model.keras")
}


def load_models():

    models = {}

    for fraction, model_path in MODEL_PATHS.items():

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        models[fraction] = load_model(model_path)

    return models


def predict_execution(
    spark,
    execution_path,
    out_root,
    artifacts_dir
):

    prefixes = preprocess_fast(
    execution_path=execution_path,
    artifacts_dir=artifacts_dir
    )

    models = load_models()

    results = {}

    for fraction, model in models.items():

        X = prefixes[fraction]

        probability = float(
            model.predict(X, verbose=0)[0][0]
        )

        prediction = int(probability >= 0.5)

        results[fraction] = {
            "probability": probability,
            "prediction": prediction
        }

    return results
