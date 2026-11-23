
import os
import sys
import json
import joblib
import numpy as np

from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from tensorflow.keras.preprocessing.sequence import pad_sequences




PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import feature_extraction as fe


STREAMS = [
    "ata_read",
    "ata_write",
    "mem_exec",
    "mem_readwrite",
    "mem_read",
    "mem_write"
]

FRACTIONS = [0.10, 0.25, 0.50, 0.75]

PAD_VALUE = -999.0

MODEL_INPUT_LENGTHS = {
    0.10: 32,
    0.25: 80,
    0.50: 161,
    0.75: 241
}




def load_artifacts(artifacts_dir):

    scaler_path = os.path.join(
        artifacts_dir,
        "scaler.pkl"
    )

    features_path = os.path.join(
        artifacts_dir,
        "feature_columns.json"
    )

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}"
        )

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Feature columns not found: {features_path}"
        )

    scaler = joblib.load(scaler_path)

    with open(features_path, "r") as f:
        feature_columns = json.load(f)


    if scaler.n_features_in_ != len(feature_columns):
        raise ValueError(
            "Mismatch between scaler and feature columns!"
        )



    return scaler, feature_columns



def merge_streams(spark, out_root):

    merged = None

    for stream in STREAMS:

        stream_path = os.path.join(
            out_root,
            stream
        )

        if not os.path.exists(stream_path):
            raise FileNotFoundError(
                f"Extracted stream not found: {stream_path}"
            )

        df = spark.read.parquet(stream_path)

        df = df.filter(
            F.col("win").isNotNull()
        )

        rawcnt_col = f"{stream}_rawcnt"

        if rawcnt_col in df.columns:
            df = df.drop(rawcnt_col)

        if merged is None:
            merged = df

        else:
            merged = merged.join(
                df,
                on=["rel_dir", "win"],
                how="full_outer"
            )

    if merged is None:
        raise ValueError("No streams were merged.")

    numeric_cols = [
        name
        for name, dtype in merged.dtypes
        if dtype in [
            "bigint",
            "int",
            "double",
            "float",
            "smallint",
            "tinyint"
        ]
    ]

    merged = merged.fillna(
        0,
        subset=numeric_cols
    )

    window_spec = Window.partitionBy("rel_dir")

    merged = (
        merged
        .withColumn(
            "step",
            F.col("win") - F.min("win").over(window_spec)
        )
        .drop("win")
    )

    return merged.toPandas()




def create_execution_sequence(df, feature_columns):

    if df.empty:
        raise ValueError("No features extracted.")

    missing = [
        col
        for col in feature_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing training features: {missing}"
        )

        df = df.sort_values("step")

    max_step = int(df["step"].max())

    df = (
        df
        .set_index("step")[feature_columns]
        .reindex(
            range(max_step + 1),
            fill_value=0
        )
    )

    sequence = df.to_numpy(
        dtype=np.float32
    )




    return sequence




def normalize_sequence(sequence, scaler):

    normalized_sequence = scaler.transform(
        sequence
    )

    normalized_sequence = normalized_sequence.astype(
        np.float32
    )



    return normalized_sequence




def prepare_prefixes(sequence):

    prefixes = {}


    for fraction in FRACTIONS:

        prefix_length = max(
            1,
            int(len(sequence) * fraction)
        )

        prefix = sequence[:prefix_length]

        max_len = MODEL_INPUT_LENGTHS[fraction]

        if len(prefix) > max_len:
            raise ValueError(
                f"Prefix {fraction} has {len(prefix)} steps, "
                f"but model accepts only {max_len} steps."
            )

        padded = pad_sequences(
            [prefix],
            maxlen=max_len,
            padding="post",
            truncating="post",
            dtype="float32",
            value=PAD_VALUE
        )

        prefixes[fraction] = padded


    return prefixes


def preprocess_execution(
    spark,
    execution_path,
    out_root,
    artifacts_dir
):

 

    fe.ingest_all(
        spark=spark,
        raw_root=execution_path,
        out_root=out_root
    )

    scaler, feature_columns = load_artifacts(
        artifacts_dir
    )

    merged_df = merge_streams(
        spark=spark,
        out_root=out_root
    )

  

    sequence = create_execution_sequence(
        df=merged_df,
        feature_columns=feature_columns
    )


    normalized_sequence = normalize_sequence(
        sequence=sequence,
        scaler=scaler
    )

    prefixes = prepare_prefixes(
        normalized_sequence
    )


    return prefixes



if __name__ == "__main__":

    artifacts_dir = PROJECT_ROOT / "artifacts"

    scaler, feature_columns = load_artifacts(
        str(artifacts_dir)
    )
