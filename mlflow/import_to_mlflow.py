import mlflow
import keras
import os
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("RanS-MAP")

for model_type in ["gru", "lstm"]:
    for size in [10, 25, 50, 75]:
        path = os.path.join(BASE_DIR, "src", "model", f"{model_type}_model", f"{model_type}_{size}_model.keras")
        model = keras.models.load_model(path)
        
        with mlflow.start_run(run_name=f"{model_type.upper()}-{size}"):
            mlflow.set_tags({"model_type": model_type.upper(), "origin": "imported"})
            mlflow.log_param("variant", size)
            mlflow.log_param("n_params", model.count_params())
            mlflow.log_artifact(path, artifact_path="model")
        print(f"OK : {path}")