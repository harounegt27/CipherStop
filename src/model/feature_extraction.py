import os
import time

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    DoubleType
)



RAW_ROOT = (
    "/kaggle/input/datasets/hiranomanabu/"
    "ransmap-2024-ransomware-behavioral-features/"
    "RanSMAP/dataset"
)

STREAMS = [
    "ata_read",
    "ata_write",
    "mem_exec",
    "mem_readwrite",
    "mem_read",
    "mem_write"
]

HAS_ENTROPY = {
    "ata_write",
    "mem_readwrite",
    "mem_write"
}

WINDOW_S = 1.0


EVENT_SCHEMA = StructType([
    StructField("sec", LongType(), True),
    StructField("ns", DoubleType(), True),
    StructField("addr", LongType(), True),
    StructField("size", LongType(), True),
    StructField("entropy", DoubleType(), True),
    StructField("c6", DoubleType(), True)
])




def get_rel_dirs(exec_df):

    if hasattr(exec_df, "columns") and "rel_dir" in exec_df.columns:

        # Pandas DataFrame
        if hasattr(exec_df, "to_dict"):
            return (
                exec_df["rel_dir"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

        # Spark DataFrame
        return [
            row["rel_dir"]
            for row in (
                exec_df
                .select("rel_dir")
                .distinct()
                .collect()
            )
        ]

    raise ValueError(
        "exec_df must contain a rel_dir column."
    )



def ingest_stream(stream, exec_df, spark, out_root):

    start_time = time.time()

    print(f"\nProcessing stream: {stream}")

    rel_dirs = get_rel_dirs(exec_df)

    if not rel_dirs:
        raise ValueError("No execution directories found.")


    paths = []
    missing = []

    for rel_dir in rel_dirs:

        csv_path = os.path.join(
            RAW_ROOT,
            rel_dir,
            f"{stream}.csv"
        )

        if os.path.isfile(csv_path):
            paths.append(csv_path)

        else:
            missing.append((rel_dir, csv_path))

    print("Total executions:", len(rel_dirs))
    print("Available files:", len(paths))
    print("Missing files:", len(missing))

    if missing:

        print(
            f"WARNING: {len(missing)} missing files "
            f"for stream {stream}."
        )

        for rel_dir, path in missing[:5]:
            print("Missing:", path)

    if not paths:

        raise FileNotFoundError(
            f"No CSV files found for stream {stream}."
        )



    raw = (
        spark.read
        .schema(EVENT_SCHEMA)
        .option("header", "false")
        .csv(paths)
        .withColumn(
            "_source",
            F.input_file_name()
        )
    )

    # Recover rel_dir from the source file path
    raw = raw.withColumn(
        "rel_dir",
        F.regexp_extract(
            F.col("_source"),
            r"/dataset/(.+)/[^/]+$",
            1
        )
    )



    valid_condition = (
        F.col("sec").isNotNull()
        & F.col("ns").isNotNull()
        & F.col("addr").isNotNull()
        & F.col("size").isNotNull()
        & (F.col("size") >= 0)
    )

    raw = raw.withColumn(
        "_valid",
        valid_condition
    )


    raw = raw.withColumn(
        "win",
        F.floor(
            (
                F.col("sec")
                + F.col("ns") / 1_000_000_000.0
            ) / WINDOW_S
        ).cast("long")
    )


    aggregations = [

        F.sum(
            F.when(F.col("_valid"), 1).otherwise(0)
        ).alias(f"{stream}_n"),

        F.count("*").alias(f"{stream}_rawcnt"),

        F.sum(
            F.when(
                F.col("_valid"),
                F.col("size")
            ).otherwise(0)
        ).alias(f"{stream}_size_sum"),

        (
            F.max(
                F.when(F.col("_valid"), F.col("addr"))
            )
            -
            F.min(
                F.when(F.col("_valid"), F.col("addr"))
            )
        ).alias(f"{stream}_addr_range")
    ]



    if stream in HAS_ENTROPY:

        entropy_condition = (
            F.col("_valid")
            & F.col("entropy").isNotNull()
            & (F.col("entropy") > 0)
        )

        aggregations.extend([

            F.avg(
                F.when(
                    entropy_condition,
                    F.col("entropy")
                )
            ).alias(f"{stream}_entropy_mean"),

            F.max(
                F.when(
                    entropy_condition,
                    F.col("entropy")
                )
            ).alias(f"{stream}_entropy_max")

        ])


    windows = (
        raw
        .groupBy("rel_dir", "win")
        .agg(*aggregations)
    )



    output_path = os.path.join(
        out_root,
        stream
    )

    (
        windows
        .write
        .mode("overwrite")
        .parquet(output_path)
    )


    row_count = windows.count()

    elapsed = time.time() - start_time

    report = {
        "stream": stream,
        "executions_requested": len(rel_dirs),
        "files_found": len(paths),
        "files_missing": len(missing),
        "windows": row_count,
        "output_path": output_path,
        "elapsed_seconds": round(elapsed, 2)
    }

    print(f"Completed {stream}")
    print("Windows:", row_count)
    print("Output:", output_path)
    print("Time:", round(elapsed, 2), "seconds")

    return report


def ingest_all(exec_df, spark, out_root):

    os.makedirs(out_root, exist_ok=True)

    report = {}

    for stream in STREAMS:

        report[stream] = ingest_stream(
            stream=stream,
            exec_df=exec_df,
            spark=spark,
            out_root=out_root
        )

    print("\nAll streams processed.")

    return report
