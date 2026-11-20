
import os
import time
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    DoubleType
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


def ingest_stream(stream, spark, raw_root, out_root, execution_id):

    start_time = time.time()

    print(f"\nProcessing stream: {stream}")

    csv_path = os.path.join(
        raw_root,
        f"{stream}.csv"
    )

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Missing CSV file: {csv_path}"
        )

    raw = (
        spark.read
        .schema(EVENT_SCHEMA)
        .option("header", "false")
        .csv(csv_path)
    )

    raw = raw.withColumn(
        "rel_dir",
        F.lit(execution_id)
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
        "execution_id": execution_id,
        "files_found": 1,
        "windows": row_count,
        "output_path": output_path,
        "elapsed_seconds": round(elapsed, 2)
    }

    print(f"Completed {stream}")
    print("Windows:", row_count)
    print("Output:", output_path)
    print("Time:", round(elapsed, 2), "seconds")

    return report


def ingest_all(spark, raw_root, out_root):

    if not os.path.isdir(raw_root):
        raise FileNotFoundError(
            f"Execution folder not found: {raw_root}"
        )

    execution_id = os.path.basename(
        os.path.normpath(raw_root)
    )

    os.makedirs(out_root, exist_ok=True)

    report = {}

    for stream in STREAMS:

        report[stream] = ingest_stream(
            stream=stream,
            spark=spark,
            raw_root=raw_root,
            out_root=out_root,
            execution_id=execution_id
        )

    print("\nAll streams processed.")

    return report
