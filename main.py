import os
from datetime import datetime

import pyspark

import download_dataset
# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")



if __name__ == "__main__":
    # Download dataset before proceeding
    download_dataset.download_dataset(output_dir="data")

