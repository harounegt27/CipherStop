import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "RanS-MAP"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found. Run import_to_mlflow.py first.")

# Find GRU-25 run
runs = client.search_runs(
    [exp.experiment_id],
    filter_string="tags.model_type = 'GRU' and params.variant = '25'",
    order_by=["start_time DESC"],
    max_results=1,
)

if not runs:
    raise SystemExit("GRU-25 run not found. Run import_to_mlflow.py and metrics.py first.")

best_run = runs[0]
run_id = best_run.info.run_id

print(f"Best run found: GRU-25")
print(f"Run ID       : {run_id}")
print(f"F1-score     : {best_run.data.metrics.get('f1_score', 'not logged yet')}")
print(f"Accuracy     : {best_run.data.metrics.get('accuracy', 'not logged yet')}")

# Apply tags
client.set_tag(run_id, "stage", "Staging")
client.set_tag(run_id, "candidate", "production")
client.set_tag(run_id, "selected_by", "Haroun Mezned")
client.set_tag(run_id, "selection_reason",
    "Best tradeoff: F1=1.0 at 25% sequence, fewer params (17345) vs LSTM (22849)")

print("\n✅ Tags applied successfully.")
print(f"View at: {MLFLOW_URI}")