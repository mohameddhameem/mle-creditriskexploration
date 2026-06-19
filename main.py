import logging
import os
import pyspark
import sys

import config
import download_dataset
import utils.data_processing_bronze_table
import utils.data_processing_gold_table
import utils.data_processing_silver_table

logger = logging.getLogger(__name__)
download_dataset.download_dataset(output_dir=config.DATA_DIR)

spark = pyspark.sql.SparkSession.builder \
    .appName("HomeCredit-Medallion-Pipeline") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

if __name__ == "__main__":
    if not os.path.isdir(config.DATA_DIR):
        sys.exit(1)

    os.makedirs(config.BRONZE_DIR, exist_ok=True)
    os.makedirs(config.SILVER_DIR, exist_ok=True)
    os.makedirs(config.GOLD_DIR, exist_ok=True)

    missing = [f for f in config.RAW_FILES.values() if not os.path.exists(os.path.join(config.DATA_DIR, f))]
    if missing:
        sys.exit(2)

    logger.info("[Pipeline] Starting Bronze layer...")
    utils.data_processing_bronze_table.run_bronze_layer(
        spark, config.DATA_DIR, config.BRONZE_DIR
    )

    logger.info("[Pipeline] Starting Silver layer...")
    utils.data_processing_silver_table.run_silver_layer(
        spark, config.BRONZE_DIR, config.SILVER_DIR
    )

    # --- Gold layer ---
    logger.info("[Pipeline] Starting Gold layer...")
    gold_results = utils.data_processing_gold_table.run_gold_layer(
        spark=spark,
        silver_dir=config.SILVER_DIR,
        gold_dir=config.GOLD_DIR,
        oot_start=config.OOT_START,
        train_frac=config.SPLIT_TRAIN,
        val_frac=config.SPLIT_VAL,
        seed=config.RANDOM_SEED,
    )

    spark.stop()
