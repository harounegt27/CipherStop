import mlflow
from mlflow.tracking import MlflowClient

import os
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "RanS-MAP"
MODEL_NAME      = "CipherStop-GRU"   # name in the Model Registry
BEST_MODEL_TYPE = "GRU"
BEST_VARIANT    = "25"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

# ── 1. Find the GRU-25 run ──────────────────────────────────────────────────
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' not found. Run import_to_mlflow.py first.")

runs = client.search_runs(
    [exp.experiment_id],
    filter_string=(
        f"tags.model_type = '{BEST_MODEL_TYPE}' "
        f"and params.variant = '{BEST_VARIANT}'"
    ),
    order_by=["start_time DESC"],
    max_results=1,
)

if not runs:
    raise SystemExit("GRU-25 run not found. Run import_to_mlflow.py and metrics.py first.")

best_run    = runs[0]
run_id      = best_run.info.run_id
artifact_uri = f"runs:/{run_id}/model"

print(f"✅ Best run found : GRU-25")
print(f"   Run ID         : {run_id}")
print(f"   F1-score       : {best_run.data.metrics.get('f1_score', 'not logged yet')}")
print(f"   Artifact URI   : {artifact_uri}")

# ── 2. Register the model ───────────────────────────────────────────────────
print(f"\n📦 Registering model as '{MODEL_NAME}' ...")
result = mlflow.register_model(artifact_uri, MODEL_NAME)
model_version = result.version
print(f"   Registered version : {model_version}")

# ── 3. Transition to Production (alias-based, MLflow 2.9+) ──────────────────
print(f"\n🚀 Setting alias 'production' ...")
client.set_registered_model_alias(
    name    = MODEL_NAME,
    alias   = "production",
    version = model_version,
)
# ── 4. Add description ──────────────────────────────────────────────────────
client.update_model_version(
    name        = MODEL_NAME,
    version     = model_version,
    description = (
        "GRU model trained on 25% of behavioral sequence. "
        "F1=1.0, 17345 params. Best tradeoff between earliness and accuracy. "
        "Selected for production by Haroun Mezned."
    ),
)

print(f"\n✅ Done.")
print(f"   Model     : {MODEL_NAME}")
print(f"   Version   : {model_version}")
print(f"   Stage     : Production")
print(f"   View at   : {MLFLOW_URI}/#/models/{MODEL_NAME}")