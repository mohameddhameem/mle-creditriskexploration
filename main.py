import os
import sys
import logging

import pyspark

import config
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# ---------------------------------------------------------------------------
# HuggingFace download (activate once application_train_dated.csv is uploaded):
# Uncomment the two lines below — download_dataset.py will populate data/
# automatically. No other changes needed in this file, config.py, or the utils.
# ---------------------------------------------------------------------------
import download_dataset

# Download files from HuggingFace into the canonical data directory
download_dataset.download_dataset(output_dir=config.DATA_DIR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("HomeCredit-Medallion-Pipeline") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Home Credit Medallion Pipeline — Bronze → Silver → Gold")
    logger.info("=" * 60)
    logger.info(f"  DATA_DIR   : {config.DATA_DIR}")
    logger.info(f"  BRONZE_DIR : {config.BRONZE_DIR}")
    logger.info(f"  SILVER_DIR : {config.SILVER_DIR}")
    logger.info(f"  GOLD_DIR   : {config.GOLD_DIR}")

    # Verify source directory exists before proceeding
    if not os.path.isdir(config.DATA_DIR):
        logger.error(
            f"DATA_DIR not found: {config.DATA_DIR}\n"
            "Update DATA_DIR in config.py and ensure the folder exists."
        )
        sys.exit(1)

    # Create output directories (datamart/bronze, datamart/silver, datamart/gold)
    os.makedirs(config.BRONZE_DIR, exist_ok=True)
    os.makedirs(config.SILVER_DIR, exist_ok=True)
    os.makedirs(config.GOLD_DIR, exist_ok=True)

    # Pre-ingest validation: ensure all expected RAW_FILES are present in DATA_DIR
    missing = [f for f in config.RAW_FILES.values() if not os.path.exists(os.path.join(config.DATA_DIR, f))]
    if missing:
        logger.error(
            f"Missing expected source CSV files in {config.DATA_DIR}: {missing}\n"
            "If you intended to download from HuggingFace, ensure internet access and retry."
        )
        sys.exit(2)

    # --- Bronze layer ---
    logger.info("\n[Pipeline] Starting Bronze layer...")
    bronze_results = utils.data_processing_bronze_table.run_bronze_layer(
        spark, config.DATA_DIR, config.BRONZE_DIR
    )

    # --- Silver layer ---
    logger.info("\n[Pipeline] Starting Silver layer...")
    silver_results = utils.data_processing_silver_table.run_silver_layer(
        spark, config.BRONZE_DIR, config.SILVER_DIR
    )

    # --- Gold layer ---
    logger.info("\n[Pipeline] Starting Gold layer...")
    gold_results = utils.data_processing_gold_table.run_gold_layer(
        spark=spark,
        silver_dir=config.SILVER_DIR,
        gold_dir=config.GOLD_DIR,
        oot_start=config.OOT_START,
        train_frac=config.SPLIT_TRAIN,
        val_frac=config.SPLIT_VAL,
        seed=config.RANDOM_SEED,
    )

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete. Row count summary:")
    logger.info(f"\n  {'Table':<20} {'Bronze':>10} {'Silver':>10}")
    logger.info(f"  {'-'*20} {'-'*10} {'-'*10}")
    for name in bronze_results:
        logger.info(
            f"  {name:<20} {bronze_results[name]:>10,} {silver_results[name]:>10,}"
        )
    logger.info(f"\n  {'Gold store/split':<30} {'Rows':>10}")
    logger.info(f"  {'-'*30} {'-'*10}")
    for store_split, cnt in sorted(gold_results.items()):
        logger.info(f"  {store_split:<30} {cnt:>10,}")
    logger.info("=" * 60)

    spark.stop()
