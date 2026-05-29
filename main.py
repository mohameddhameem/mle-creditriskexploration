import os
import sys
import logging

import pyspark

import config
import utils.data_processing_bronze_table
import utils.data_processing_silver_table

# ---------------------------------------------------------------------------
# HuggingFace download (activate once application_train_dated.csv is uploaded):
# Uncomment the two lines below — download_dataset.py will populate data/
# automatically. No other changes needed in this file, config.py, or the utils.
# ---------------------------------------------------------------------------
# import download_dataset
# download_dataset.download_dataset(output_dir="data")

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
    logger.info("Home Credit Medallion Pipeline — Bronze → Silver")
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

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete. Row count summary:")
    logger.info(f"  {'Table':<20} {'Bronze':>10} {'Silver':>10}")
    logger.info(f"  {'-'*20} {'-'*10} {'-'*10}")
    for name in bronze_results:
        logger.info(
            f"  {name:<20} {bronze_results[name]:>10,} {silver_results[name]:>10,}"
        )
    logger.info("=" * 60)

    spark.stop()
