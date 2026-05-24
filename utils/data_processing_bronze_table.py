# =============================================================================
# utils/data_processing_bronze_table.py
# Bronze layer: ingest raw CSVs → Parquet with zero business logic.
# One Parquet file per source table, partitioned into data/bronze/<table>/.
# Validations: row count, non-null primary key, schema snapshot written to log.
# =============================================================================

import os
import logging
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def _bronze_path(bronze_dir: str, table_name: str) -> str:
    return os.path.join(bronze_dir, table_name)


def process_bronze_table(
    spark: SparkSession,
    table_name: str,
    csv_path: str,
    bronze_dir: str,
    primary_key: str | None = None,
) -> int:
    """
    Read a raw CSV and write it as Parquet to the Bronze layer.

    Parameters
    ----------
    spark        : active SparkSession
    table_name   : logical name used as the output sub-directory
    csv_path     : absolute path to the source CSV file
    bronze_dir   : root Bronze output directory (data/bronze/)
    primary_key  : column name to validate for nulls (optional)

    Returns
    -------
    Row count of the written Parquet.
    """
    logger.info(f"[Bronze] Reading {table_name} from {csv_path}")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(csv_path)
    )

    row_count = df.count()
    col_count = len(df.columns)
    logger.info(f"[Bronze] {table_name}: {row_count:,} rows, {col_count} columns")

    # --- Validation 1: non-empty ---
    if row_count == 0:
        raise ValueError(f"[Bronze] {table_name}: source CSV is empty — aborting.")

    # --- Validation 2: primary key nulls ---
    if primary_key:
        null_count = df.filter(df[primary_key].isNull()).count()
        if null_count > 0:
            raise ValueError(
                f"[Bronze] {table_name}: {null_count:,} null values in primary key "
                f"'{primary_key}' — aborting."
            )
        logger.info(f"[Bronze] {table_name}: primary key '{primary_key}' — no nulls.")

    # --- Write Bronze Parquet (overwrite for idempotency) ---
    out_path = _bronze_path(bronze_dir, table_name)
    df.write.mode("overwrite").parquet(out_path)
    logger.info(f"[Bronze] {table_name}: written to {out_path}")

    # --- Schema snapshot to log ---
    schema_lines = [f"  {f.name}: {f.dataType}" for f in df.schema.fields]
    logger.info(f"[Bronze] {table_name} schema:\n" + "\n".join(schema_lines))

    return row_count


def run_bronze_layer(spark: SparkSession, data_dir: str, bronze_dir: str) -> dict:
    """
    Ingest all 7 source tables into the Bronze layer.

    Returns a dict of {table_name: row_count} for downstream logging.
    """
    import config
    RAW_FILES = config.RAW_FILES

    # Primary keys per table (None = no PK validation)
    primary_keys = {
        "application":    "SK_ID_CURR",
        "bureau":         "SK_ID_BUREAU",
        "bureau_balance": None,              # compound key SK_ID_BUREAU + MONTHS_BALANCE
        "previous_app":   "SK_ID_PREV",
        "pos_cash":       None,              # compound key SK_ID_PREV + MONTHS_BALANCE
        "installments":   None,              # compound key SK_ID_PREV + NUM_INSTALMENT_NUMBER
        "credit_card":    None,              # compound key SK_ID_PREV + MONTHS_BALANCE
    }

    results = {}
    for table_name, filename in RAW_FILES.items():
        csv_path = os.path.join(data_dir, filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[Bronze] Expected file not found: {csv_path}\n"
                f"Ensure all 7 CSVs are present in {data_dir}"
            )
        results[table_name] = process_bronze_table(
            spark=spark,
            table_name=table_name,
            csv_path=csv_path,
            bronze_dir=bronze_dir,
            primary_key=primary_keys[table_name],
        )

    logger.info("[Bronze] All tables ingested. Summary:")
    for name, count in results.items():
        logger.info(f"  {name}: {count:,} rows")

    return results
