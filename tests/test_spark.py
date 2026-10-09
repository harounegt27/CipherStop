from src.common.spark_session import get_spark


def test_spark_runs():
    spark = get_spark("smoke")
    result = spark.range(1_000_000).selectExpr("sum(id) as s").first()["s"]
    assert result == 499999500000
    spark.stop()


def test_parquet_roundtrip(tmp_path):
    spark = get_spark("smoke_parquet")
    out = str(tmp_path / "out")
    spark.range(1000).write.mode("overwrite").parquet(out)
    assert spark.read.parquet(out).count() == 1000
    spark.stop()
