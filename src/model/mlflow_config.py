import mlflow
import yaml
from pathlib import Path

def setup_mlflow():
    """Load MLflow config from params.yaml and set tracking URI."""
    params_path = Path("params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")

    return mlflow