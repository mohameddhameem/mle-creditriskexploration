import os
from pyspark.sql import SparkSession
import config


def _bronze_path(bronze_dir: str, table_name: str) -> str:
    return os.path.join(bronze_dir, table_name)


def process_bronze_table(
        spark: SparkSession,
        table_name: str,
        source_path: str,
        bronze_dir: str,
        primary_key: str | None = None,
) -> int:
    """Reads a raw source file and writes it as Parquet to the Bronze layer."""
    df = spark.read.parquet(source_path)
    row_count = df.count()

    if row_count == 0:
        raise ValueError(f"[Bronze] {table_name}: source file is empty — aborting.")

    if primary_key:
        null_count = df.filter(df[primary_key].isNull()).count()
        if null_count > 0:
            raise ValueError(
                f"[Bronze] {table_name}: {null_count:,} null values in primary key "
                f"'{primary_key}' — aborting."
            )

    out_path = _bronze_path(bronze_dir, table_name)
    df.write.mode("overwrite").parquet(out_path)

    return row_count


def run_bronze_layer(spark: SparkSession, data_dir: str, bronze_dir: str) -> dict:
    """Ingests all source tables into the Bronze layer."""
    RAW_FILES = config.RAW_FILES

    primary_keys = {
        "application": "SK_ID_CURR",
        "bureau": "SK_ID_BUREAU",
        "bureau_balance": None,
        "previous_app": "SK_ID_PREV",
        "pos_cash": None,
        "installments": None,
        "credit_card": None,
    }

    results = {}
    for table_name, filename in RAW_FILES.items():
        source_path = os.path.join(data_dir, filename)
        if not os.path.exists(source_path):
            raise FileNotFoundError(
                f"[Bronze] Expected file not found: {source_path}\n"
                f"Ensure all 7 Parquet files are present in {data_dir}"
            )
        results[table_name] = process_bronze_table(
            spark=spark,
            table_name=table_name,
            source_path=source_path,
            bronze_dir=bronze_dir,
            primary_key=primary_keys[table_name],
        )

    return results
