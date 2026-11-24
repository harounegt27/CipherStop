import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences


STREAMS = [
    "ata_read",
    "ata_write",
    "mem_exec",
    "mem_readwrite",
    "mem_read",
    "mem_write",
]

HAS_ENTROPY = {
    "ata_write",
    "mem_readwrite",
    "mem_write",
}

FRACTIONS = [0.10, 0.25, 0.50, 0.75]

WINDOW_S = 1.0

PAD_VALUE = -999.0

MODEL_INPUT_LENGTHS = {
    0.10: 32,
    0.25: 80,
    0.50: 161,
    0.75: 241,
}

COLUMNS = ["sec", "ns", "addr", "size", "entropy", "c6"]


def load_artifacts(artifacts_dir: str):
    scaler = joblib.load(
        os.path.join(artifacts_dir, "scaler.pkl")
    )
    with open(
        os.path.join(artifacts_dir, "feature_columns.json")
    ) as f:
        feature_columns = json.load(f)
    return scaler, feature_columns


def process_stream(csv_path: str, stream: str) -> pd.DataFrame:
    """
    Read one RanSMAP CSV, compute 1-second window features.
    Equivalent to what Spark does in feature_extraction.py
    but runs entirely in pandas — much faster for single files.
    """

    df = pd.read_csv(
        csv_path,
        header=None,
        names=COLUMNS,
        dtype={
            "sec": "Int64",
            "ns": float,
            "addr": "Int64",
            "size": "Int64",
            "entropy": float,
            "c6": float,
        },
    )

    # Keep only valid rows
    valid = (
        df["sec"].notna()
        & df["ns"].notna()
        & df["addr"].notna()
        & df["size"].notna()
        & (df["size"] >= 0)
    )
    df = df[valid].copy()

    if df.empty:
        return pd.DataFrame()

    # Compute 1-second window index
    df["win"] = np.floor(
        (df["sec"] + df["ns"] / 1_000_000_000.0) / WINDOW_S
    ).astype(int)

    # Aggregate per window
    agg = df.groupby("win").agg(
        **{
            f"{stream}_n":        ("size", "count"),
            f"{stream}_size_sum": ("size", "sum"),
            f"{stream}_addr_range": (
                "addr",
                lambda x: x.max() - x.min()
            ),
        }
    )

    if stream in HAS_ENTROPY:
        entropy_valid = df[df["entropy"].notna() & (df["entropy"] > 0)]
        if not entropy_valid.empty:
            entropy_agg = entropy_valid.groupby("win")["entropy"].agg(
                **{
                    f"{stream}_entropy_mean": "mean",
                    f"{stream}_entropy_max": "max",
                }
            )
            agg = agg.join(entropy_agg, how="left")
        else:
            agg[f"{stream}_entropy_mean"] = 0.0
            agg[f"{stream}_entropy_max"] = 0.0

    return agg.reset_index()


def merge_streams(execution_path: str) -> pd.DataFrame:
    """
    Process all 6 streams and merge them on the window index.
    """
    merged = None

    for stream in STREAMS:
        csv_path = os.path.join(
            execution_path, f"{stream}.csv"
        )

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Missing: {csv_path}"
            )

        df = process_stream(csv_path, stream)

        if df.empty:
            continue

        if merged is None:
            merged = df
        else:
            merged = pd.merge(
                merged, df,
                on="win",
                how="outer"
            )

    if merged is None or merged.empty:
        raise ValueError("No data extracted.")

    # Normalize window to start at 0
    merged = merged.sort_values("win")
    merged["step"] = merged["win"] - merged["win"].min()
    merged = merged.drop(columns=["win"])
    merged = merged.fillna(0)

    return merged


def build_sequence(
    merged: pd.DataFrame,
    feature_columns: list
) -> np.ndarray:
    """
    Build a 2D numpy array (steps × features)
    in the same order as training.
    """
    for col in feature_columns:
        if col not in merged.columns:
            merged[col] = 0.0

    merged = merged.sort_values("step")
    sequence = merged[feature_columns].to_numpy(
        dtype=np.float32
    )

    return sequence


def prepare_prefixes(
    sequence: np.ndarray,
    scaler
) -> dict:
    """
    Normalize and cut the sequence at 10/25/50/75%.
    Returns a dict of padded arrays ready for the model.
    """
    sequence = scaler.transform(sequence).astype(
        np.float32
    )

    prefixes = {}

    for fraction in FRACTIONS:

        prefix_len = max(
            1, int(len(sequence) * fraction)
        )
        prefix = sequence[:prefix_len]
        max_len = MODEL_INPUT_LENGTHS[fraction]

        padded = pad_sequences(
            [prefix],
            maxlen=max_len,
            padding="post",
            truncating="post",
            dtype="float32",
            value=PAD_VALUE,
        )

        prefixes[fraction] = padded

    return prefixes


def preprocess_fast(
    execution_path: str,
    artifacts_dir: str
) -> dict:
    """
    Full fast inference preprocessing pipeline.
    Drop-in replacement for preprocess_execution()
    without Spark or Parquet I/O.
    """
    scaler, feature_columns = load_artifacts(
        artifacts_dir
    )

    merged = merge_streams(execution_path)

    sequence = build_sequence(
        merged, feature_columns
    )

    prefixes = prepare_prefixes(sequence, scaler)

    return prefixes