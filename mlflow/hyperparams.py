import keras
import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
exp = client.get_experiment_by_name("RanS-MAP")
ROOT = Path(__file__).resolve().parent.parent

for model_type in ["gru", "lstm"]:
    for size in [10, 25, 50, 75]:
        path = ROOT / "src" / "model" / f"{model_type}_model" / f"{model_type}_{size}_model.keras"
        model = keras.models.load_model(str(path))

        # 1. Dictionnaire des hyperparamètres
        params = {"n_layers": len(model.layers)}

        for i, layer in enumerate(model.layers):
            cfg = layer.get_config()
            params[f"layer{i}_type"] = layer.__class__.__name__
            for key in ["units", "dropout", "recurrent_dropout", "rate",
                        "activation", "recurrent_activation", "return_sequences"]:
                if key in cfg:
                    params[f"layer{i}_{key}"] = cfg[key]

        # 2. Entrée du modèle et optimiseur
        params["input_shape"] = str(model.input_shape)
        if model.optimizer is not None:
            opt = model.optimizer.get_config()
            params["optimizer"] = model.optimizer.__class__.__name__
            params["learning_rate"] = opt.get("learning_rate")
        params["loss"] = str(getattr(model, "loss", None))

        # 3. Retrouver le run correspondant
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.model_type = '{model_type.upper()}' and params.variant = '{size}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            print(f"Run introuvable : {model_type} {size}")
            continue
        run_id = runs[0].info.run_id

        # 4. Envoyer chaque paramètre
        for k, v in params.items():
            try:
                client.log_param(run_id, k, v)
            except Exception as e:
                print(f"  ignoré {k} : {e}")
        print(f"OK : {model_type.upper()}-{size} ({len(params)} paramètres)")