import mlflow
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://localhost:5000")

df = mlflow.search_runs(experiment_names=["RanS-MAP"])

df = df.dropna(subset=["metrics.recall"])

df = (df.sort_values("start_time", ascending=False)
        .drop_duplicates(subset=["tags.model_type", "params.variant"]))

table = df[["tags.model_type", "params.variant", "metrics.accuracy",
            "metrics.precision", "metrics.recall", "metrics.f1_score",
            "params.n_params"]].copy()
table.columns = ["modele", "prefixe_%", "accuracy", "precision",
                 "recall", "f1", "n_params"]
table["prefixe_%"] = table["prefixe_%"].astype(int)
table = table.sort_values(["prefixe_%", "modele"])
print(table.to_string(index=False))
table.to_csv("comparison_table.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ["recall", "f1"]):
    for name, g in table.groupby("modele"):
        g = g.sort_values("prefixe_%")
        ax.plot(g["prefixe_%"], g[metric], marker="o", label=name)
    ax.set_xlabel("Préfixe utilisé (%)")
    ax.set_ylabel(metric)
    ax.set_xticks([10, 25, 50, 75])
    ax.set_title(f"{metric} selon la part de séquence")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("comparison.png", dpi=150)
mlflow.set_experiment("RansMAP-comparison")
with mlflow.start_run(run_name="GRU-vs-LSTM"):
    mlflow.log_artifact("comparison_table.csv")
    mlflow.log_artifact("comparison.png")