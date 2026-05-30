import os
import sys
import pyspark
import config
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import download_dataset

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

    utils.data_processing_bronze_table.run_bronze_layer(
        spark, config.DATA_DIR, config.BRONZE_DIR
    )

    utils.data_processing_silver_table.run_silver_layer(
        spark, config.BRONZE_DIR, config.SILVER_DIR
    )

    spark.stop()
