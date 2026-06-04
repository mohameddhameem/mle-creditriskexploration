import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, IntegerType
import config

_DAYS_EMPLOYED_SENTINEL = 365_243

_MEDI_COLS = [
    "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI",
    "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
]

_MODE_TRIPLICATE_COLS = [
    "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE",
    "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE",
]

# Documents 2,4,5,7,9-21 have near-zero variance in the training population
# (submission rate < 1%) so they add noise without signal.
# Documents 3, 6, 8 are retained as they show meaningful submission rates
# and correlation with default in EDA.
_FLAG_DOCUMENT_DROP = [
    "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
]

# FLAG_MOBIL and FLAG_CONT_MOBILE are 1 for virtually every applicant (zero variance).
_ZERO_VARIANCE_FLAGS = [
    "FLAG_MOBIL",
    "FLAG_CONT_MOBILE",
]

_APPLICATION_DROP_COLS = (
        _MEDI_COLS
        + _MODE_TRIPLICATE_COLS
        + _FLAG_DOCUMENT_DROP
        + _ZERO_VARIANCE_FLAGS
)


def _silver_path(silver_dir: str, table_name: str) -> str:
    return os.path.join(silver_dir, table_name)


def _cast_days_columns(df: DataFrame) -> DataFrame:
    """Casts DAYS_* columns to IntegerType."""
    days_cols = [c for c in df.columns if c.startswith("DAYS_")]
    for col in days_cols:
        df = df.withColumn(col, F.col(col).cast(IntegerType()))
    return df


def _silver_application(df: DataFrame) -> DataFrame:
    """Cleans the application table."""
    existing_drop = [c for c in _APPLICATION_DROP_COLS if c in df.columns]
    df = df.drop(*existing_drop)

    df = df.withColumn(
        "DAYS_EMPLOYED_MISSING",
        F.when(F.col("DAYS_EMPLOYED") == _DAYS_EMPLOYED_SENTINEL, 1).otherwise(0).cast(IntegerType()),
    )
    df = df.withColumn(
        "DAYS_EMPLOYED",
        F.when(F.col("DAYS_EMPLOYED") == _DAYS_EMPLOYED_SENTINEL, None).otherwise(F.col("DAYS_EMPLOYED")),
    )

    if "application_date" in df.columns:
        df = df.withColumn("application_date", F.col("application_date").cast(DateType()))

    return df


def _silver_bureau(df: DataFrame) -> DataFrame:
    """Cleans the bureau table."""
    df = _cast_days_columns(df)
    return df


def _silver_bureau_balance(df: DataFrame) -> DataFrame:
    """Cleans the bureau_balance table."""
    df = df.withColumn("MONTHS_BALANCE", F.col("MONTHS_BALANCE").cast(IntegerType()))
    return df


def _silver_previous_app(df: DataFrame) -> DataFrame:
    """Cleans the previous_application table."""
    df = _cast_days_columns(df)
    return df


def _silver_pos_cash(df: DataFrame) -> DataFrame:
    """Cleans the POS_CASH_balance table."""
    df = df.withColumn("MONTHS_BALANCE", F.col("MONTHS_BALANCE").cast(IntegerType()))
    return df


def _silver_installments(df: DataFrame) -> DataFrame:
    """Cleans the installments_payments table."""
    df = _cast_days_columns(df)
    return df


def _silver_credit_card(df: DataFrame) -> DataFrame:
    """Cleans the credit_card_balance table."""
    df = df.withColumn("MONTHS_BALANCE", F.col("MONTHS_BALANCE").cast(IntegerType()))
    return df


def process_silver_table(
        spark: SparkSession,
        table_name: str,
        bronze_dir: str,
        silver_dir: str,
) -> int:
    """Reads a Bronze Parquet, applies Silver transforms, and writes a Silver Parquet."""
    bronze_path = os.path.join(bronze_dir, table_name)
    df = spark.read.parquet(bronze_path)

    if table_name == "application":
        df = _silver_application(df)
    elif table_name == "bureau":
        df = _silver_bureau(df)
    elif table_name == "bureau_balance":
        df = _silver_bureau_balance(df)
    elif table_name == "previous_app":
        df = _silver_previous_app(df)
    elif table_name == "pos_cash":
        df = _silver_pos_cash(df)
    elif table_name == "installments":
        df = _silver_installments(df)
    elif table_name == "credit_card":
        df = _silver_credit_card(df)

    row_count = df.count()
    out_path = _silver_path(silver_dir, table_name)
    df.write.mode("overwrite").parquet(out_path)

    return row_count


def run_silver_layer(spark: SparkSession, bronze_dir: str, silver_dir: str) -> dict:
    """Processes all Bronze tables into Silver."""
    RAW_FILES = config.RAW_FILES
    results = {}
    for table_name in RAW_FILES:
        results[table_name] = process_silver_table(
            spark=spark,
            table_name=table_name,
            bronze_dir=bronze_dir,
            silver_dir=silver_dir,
        )
    return results
