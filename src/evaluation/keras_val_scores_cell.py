# Paste this as a NEW CELL at the end of GRU_LSTM.ipynb (Kaggle), after the training cells.
# It scores the VALIDATION split with the 4 LSTM models that the API serves
# (same scaler, same padding as in training) and writes val_scores_lstm.csv.
import numpy as np, pandas as pd
from tensorflow.keras.models import load_model

MODEL_NAME = "lstm"                      # the model family served by backend/inference.py
fam = manifest.set_index("rel_dir")["family"]
out = pd.DataFrame({"rel_dir": val_ids, "family": [fam[r] for r in val_ids], "label": y_val})
for fraction in FRACTIONS:
    pct = int(fraction * 100)
    m = load_model(f"{WORK}/{MODEL_NAME}_{pct}_model.keras")
    out[f"prob_{pct}"] = m.predict(padded_data[fraction]["val"], verbose=0).ravel()
out.to_csv(f"{WORK}/val_scores_lstm.csv", index=False)
print(out.groupby("label").size().to_dict(), "->", f"{WORK}/val_scores_lstm.csv")
print(out.groupby("label")[[f"prob_{int(f*100)}" for f in FRACTIONS]].describe().T.round(3).to_string())
