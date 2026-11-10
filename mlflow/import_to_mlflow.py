import mlflow
import keras

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("RanS-MAP")

for model_type in ["gru", "lstm"]:
    for size in [10, 25, 50, 75]:
        path = f"src/model/{model_type}_model/{model_type}_{size}_model.keras"
        model = keras.models.load_model(path)

        with mlflow.start_run(run_name=f"{model_type.upper()}-{size}"):
            mlflow.set_tags({"model_type": model_type.upper(), "origin": "imported"})
            mlflow.log_param("variant", size)
            mlflow.log_param("n_params", model.count_params())
            mlflow.log_artifact(path, artifact_path="model")
        print(f"OK : {path}")