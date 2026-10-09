import os
import sys

from pyspark.sql import SparkSession

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def get_spark(app_name: str = "cipherstop", driver_mem: str = "4g") -> SparkSession:
    """Return a shared local SparkSession configured for the CipherStop pipeline."""
    return (
        SparkSession.builder
        .master("local[*]")                              # use all CPU cores
        .appName(app_name)
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.shuffle.partitions", "16")    # default 200 is too many locally
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
