import argparse
import logging
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import pyspark
from pyspark.sql import functions as F

# Add repo root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    select_champion,
    load_artefacts,
    build_train_medians,
    production_preprocess,
    write_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--model", default="champion")
    args = parser.parse_args()

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("HomeCredit-Monthly-Inference")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    snapshotdate_str = args.snapshotdate
    snapshotdate = datetime.strptime(snapshotdate_str, "%Y-%m-%d")

    # Define the monthly booking window
    start_date = snapshotdate.replace(day=1)
    end_date = start_date + relativedelta(months=1)
    logger.info(f"Filtering applications between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")

    # Load Silver application table to filter by date
    silver_app_path = "datamart/silver/application"
    if not os.path.exists(silver_app_path):
        logger.error(f"Silver application table not found at {silver_app_path}")
        spark.stop()
        sys.exit(1)

    silver_app = spark.read.parquet(silver_app_path)
    monthly_app = silver_app.filter(
        (F.col("application_date") >= F.lit(start_date.date().strftime("%Y-%m-%d"))) &
        (F.col("application_date") < F.lit(end_date.date().strftime("%Y-%m-%d")))
    ).select("SK_ID_CURR")

    # Load all Gold feature stores
    gold_dir = "datamart/gold/feature_store"
    splits = ["train", "val", "test", "oot"]
    paths = [os.path.join(gold_dir, s) for s in splits if os.path.exists(os.path.join(gold_dir, s))]

    if not paths:
        logger.error(f"No Gold feature stores found under {gold_dir}")
        spark.stop()
        sys.exit(1)

    features_sdf = spark.read.parquet(*paths)

    # Join features with monthly app IDs
    monthly_features_sdf = features_sdf.join(monthly_app, on="SK_ID_CURR", how="inner")
    df_score = monthly_features_sdf.toPandas()

    if df_score.empty:
        logger.warning(f"No bookings found for the month of {snapshotdate_str}")
        spark.stop()
        return

    logger.info(f"Scoring {len(df_score)} records for month {snapshotdate_str}")

    # Load champion model
    model_name, ts, metadata = select_champion(args.model)
    model, encoders, feature_names, metadata = load_artefacts(model_name, ts)

    # Prepare data for scoring
    train_medians = build_train_medians(spark, encoders, feature_names)
    X, ids = production_preprocess(df_score, encoders, train_medians, feature_names)

    # Predict
    probs = model.predict_proba(X)[:, 1]
    logger.info(f"Score distribution: mean={probs.mean():.4f} min={probs.min():.4f} max={probs.max():.4f}")

    # Write predictions
    write_predictions(
        spark=spark,
        ids=ids,
        probs=probs,
        model_name=model_name,
        ts=ts,
        feature_path=gold_dir,
        run_date=snapshotdate_str,
    )

    spark.stop()
    logger.info("Monthly inference execution completed.")

if __name__ == "__main__":
    main()
