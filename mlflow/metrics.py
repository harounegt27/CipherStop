import mlflow
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "RanS-MAP"   # doit être IDENTIQUE à import_to_mlflow.py
CSV_PATH = Path(__file__).resolve().parent.parent / "src" / "results" / "results_gru_lstm.csv"

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    names = [e.name for e in client.search_experiments()]
    raise SystemExit(
        f"Experiment '{EXPERIMENT_NAME}' introuvable. Experiments existants : {names}"
    )

all_df = pd.read_csv(CSV_PATH)
print("Colonnes :", all_df.columns.tolist())
print("Lignes :", len(all_df))

for model_type in ["GRU", "LSTM"]:
    df = all_df[all_df["Model"].astype(str).str.upper() == model_type]

    for _, row in df.iterrows():
        prefix = int(str(row["Prefix"]).replace("%", ""))

        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=(
                f"tags.model_type = '{model_type}' "
                f"and params.variant = '{prefix}'"
            ),
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            print(f"Run introuvable : {model_type} {prefix}%")
            continue

        run_id = runs[0].info.run_id

        client.log_metric(run_id, "accuracy",  float(row["Accuracy"]))
        client.log_metric(run_id, "precision", float(row["Precision"]))
        client.log_metric(run_id, "recall",    float(row["Recall"]))
        client.log_metric(run_id, "f1_score",  float(row["F1-score"]))

        try:
            client.log_param(run_id, "prefix_pct", prefix)
        except Exception:
            pass

        client.log_artifact(run_id, str(CSV_PATH), artifact_path="results")
        print(f"OK : {model_type} {prefix}%")