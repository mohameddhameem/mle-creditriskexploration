# =============================================================================
# utils/data_processing_silver_table.py
# Silver layer: clean, standardise, and reduce each Bronze Parquet.
# No joins (those happen in Gold). One Silver Parquet per source table.
#
# application silver treatment:
#   - Drop 28 primary cols (_MEDI x14, triplicate _MODE x14)
#   - Drop 19 secondary cols (17 low-variance FLAG_DOCUMENT + FLAG_MOBIL +
#     FLAG_CONT_MOBILE)
#   - Recode DAYS_EMPLOYED sentinel (365,243) → null
#   - Add DAYS_EMPLOYED_MISSING binary flag (1 = was sentinel, 0 = real value)
#     Purpose: XGBoost handles nulls natively and does not need this flag.
#     Logistic Regression requires imputation which destroys null signal;
#     the flag preserves that signal so it survives Gold-layer imputation.
#   - Cast application_date to DateType (already present in dated CSV)
#   - Result: 77 columns (123 - 28 - 19 + 1 flag)
#
# All other tables: type-cast numeric columns and write as-is — no column drops.
# =============================================================================

import os
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, IntegerType

# ---------------------------------------------------------------------------
# Silver layer constants — application table
# ---------------------------------------------------------------------------

# Sentinel value in DAYS_EMPLOYED that encodes retirees / non-employed (~1000 yrs)
_DAYS_EMPLOYED_SENTINEL = 365_243

# Primary drop: _MEDI and triplicate _MODE columns (14 each = 28 total)
# Kept: 5 standalone _MODE cols (FONDKAPREMONT, HOUSETYPE, TOTALAREA,
#        WALLSMATERIAL, EMERGENCYSTATE) which have no _AVG/_MEDI counterpart.
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

# Secondary drop: near-zero variance flag columns
# FLAG_DOCUMENT_3 (71%), _6 (8.81%), _8 (8.14%) retained — meaningful variance.
_FLAG_DOCUMENT_DROP = [
    "FLAG_DOCUMENT_2",  "FLAG_DOCUMENT_4",  "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_7",  "FLAG_DOCUMENT_9",  "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
]

_ZERO_VARIANCE_FLAGS = [
    "FLAG_MOBIL",       # 100% = 1
    "FLAG_CONT_MOBILE", # 99.8% = 1
]

# Combined drop list: 28 primary + 19 secondary = 47 columns
# Final silver_application column count: 123 - 47 + 1 (DAYS_EMPLOYED_MISSING flag) = 77
_APPLICATION_DROP_COLS = (
    _MEDI_COLS
    + _MODE_TRIPLICATE_COLS
    + _FLAG_DOCUMENT_DROP
    + _ZERO_VARIANCE_FLAGS
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silver_path(silver_dir: str, table_name: str) -> str:
    return os.path.join(silver_dir, table_name)


def _cast_days_columns(df: DataFrame) -> DataFrame:
    """
    All DAYS_* columns in supplementary tables are stored as integer offsets
    (negative = past, positive = future relative to application date).
    Ensure they are IntegerType so downstream arithmetic is clean.
    """
    days_cols = [c for c in df.columns if c.startswith("DAYS_")]
    for col in days_cols:
        df = df.withColumn(col, F.col(col).cast(IntegerType()))
    return df


# ---------------------------------------------------------------------------
# Per-table Silver transforms
# ---------------------------------------------------------------------------

def _silver_application(df: DataFrame) -> DataFrame:
    """Clean the application table to its 77-column Silver form."""

    # 1. Drop columns that exist in the DataFrame (guard against schema drift)
    existing_drop = [c for c in _APPLICATION_DROP_COLS if c in df.columns]
    missing_drop  = [c for c in _APPLICATION_DROP_COLS if c not in df.columns]
    if missing_drop:
        logger.warning(
            f"[Silver] application: {len(missing_drop)} expected drop columns not found "
            f"in source — skipping: {missing_drop}"
        )
    df = df.drop(*existing_drop)
    logger.info(f"[Silver] application: dropped {len(existing_drop)} columns.")

    # 2. DAYS_EMPLOYED sentinel → null + missing flag
    df = df.withColumn(
        "DAYS_EMPLOYED_MISSING",
        F.when(F.col("DAYS_EMPLOYED") == _DAYS_EMPLOYED_SENTINEL, 1).otherwise(0).cast(IntegerType()),
    )
    df = df.withColumn(
        "DAYS_EMPLOYED",
        F.when(F.col("DAYS_EMPLOYED") == _DAYS_EMPLOYED_SENTINEL, None).otherwise(F.col("DAYS_EMPLOYED")),
    )
    logger.info(
        f"[Silver] application: DAYS_EMPLOYED sentinel ({_DAYS_EMPLOYED_SENTINEL:,}) recoded to null; "
        "DAYS_EMPLOYED_MISSING flag added."
    )

    # 3. Ensure application_date is DateType
    if "application_date" in df.columns:
        df = df.withColumn("application_date", F.col("application_date").cast(DateType()))

    return df


def _silver_bureau(df: DataFrame) -> DataFrame:
    """Minimal Silver treatment for bureau table."""
    df = _cast_days_columns(df)
    return df


def _silver_bureau_balance(df: DataFrame) -> DataFrame:
    """Minimal Silver treatment for bureau_balance table."""
    # MONTHS_BALANCE is a negative integer offset — cast to int
    df = df.withColumn("MONTHS_BALANCE", F.col("MONTHS_BALANCE").cast(IntegerType()))
    return df


def _silver_previous_app(df: DataFrame) -> DataFrame:
    """Minimal Silver treatment for previous_application table."""
    df = _cast_days_columns(df)
    return df


def _silver_pos_cash(df: DataFrame) -> DataFrame:
    """Minimal Silver treatment for POS_CASH_balance table."""
    df = df.withColumn("MONTHS_BALANCE", F.col("MONTHS_BALANCE").cast(IntegerType()))
    return df


def _silver_installments(df: DataFrame) -> DataFrame:
    """Minimal Silver treatment for installments_payments table."""
    df = _cast_days_columns(df)
    return df


def _silver_credit_card(df: DataFrame) -> DataFrame:
    """Minimal Silver treatment for credit_card_balance table."""
    df = df.withColumn("MONTHS_BALANCE", F.col("MONTHS_BALANCE").cast(IntegerType()))
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def process_silver_table(
    spark: SparkSession,
    table_name: str,
    bronze_dir: str,
    silver_dir: str,
) -> int:
    """
    Read a Bronze Parquet, apply Silver transforms, write Silver Parquet.

    Returns row count of the Silver table.
    """
    bronze_path = os.path.join(bronze_dir, table_name)
    logger.info(f"[Silver] Processing {table_name} from {bronze_path}")

    df = spark.read.parquet(bronze_path)

    # Dispatch per-table transform
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
    else:
        logger.warning(f"[Silver] No specific transform for '{table_name}' — writing as-is.")

    row_count = df.count()
    col_count = len(df.columns)
    logger.info(f"[Silver] {table_name}: {row_count:,} rows, {col_count} columns")

    out_path = _silver_path(silver_dir, table_name)
    df.write.mode("overwrite").parquet(out_path)
    logger.info(f"[Silver] {table_name}: written to {out_path}")

    return row_count


def run_silver_layer(spark: SparkSession, bronze_dir: str, silver_dir: str) -> dict:
    """
    Process all 7 Bronze tables into Silver.

    Returns a dict of {table_name: row_count}.
    """
    import config
    RAW_FILES = config.RAW_FILES

    results = {}
    for table_name in RAW_FILES:
        results[table_name] = process_silver_table(
            spark=spark,
            table_name=table_name,
            bronze_dir=bronze_dir,
            silver_dir=silver_dir,
        )

    logger.info("[Silver] All tables processed. Summary:")
    for name, count in results.items():
        logger.info(f"  {name}: {count:,} rows")

    return results
