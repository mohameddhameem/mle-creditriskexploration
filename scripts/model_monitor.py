import argparse
import logging
import os
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import functions as F
import pyspark.sql.types as T
from sklearn.metrics import roc_auc_score

# Add repo root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import (
    select_champion,
    load_artefacts,
    build_train_medians,
    production_preprocess,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def calculate_psi(expected, actual, num_bins=10):
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)
    
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)
    
    if len(bins) < 2:
        return 0.0
        
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)
    
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--model", default="champion")
    args = parser.parse_args()

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("HomeCredit-Model-Monitoring")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    snapshotdate_str = args.snapshotdate
    snapshotdate = datetime.strptime(snapshotdate_str, "%Y-%m-%d")
    splits = ["train", "val", "test", "oot"]

    # 1. Performance Monitoring (AUC-ROC) with 6-month lookback
    prediction_date = snapshotdate - relativedelta(months=6)
    prediction_date_str = prediction_date.strftime("%Y-%m-%d")
    pred_path = f"datamart/predictions/{prediction_date_str}"
    
    auc = None
    if os.path.exists(pred_path):
        logger.info(f"Predictions found for {prediction_date_str} (6 months prior). Computing performance...")
        preds_sdf = spark.read.parquet(pred_path)
        preds_pdf = preds_sdf.toPandas()

        # Load labels
        gold_label_dir = "datamart/gold/label_store"
        label_paths = [os.path.join(gold_label_dir, s) for s in splits if os.path.exists(os.path.join(gold_label_dir, s))]
        
        if label_paths:
            labels_sdf = spark.read.parquet(*label_paths)
            labels_pdf = labels_sdf.toPandas()
            merged = preds_pdf.merge(labels_pdf, on="SK_ID_CURR", how="inner")
            if not merged.empty:
                auc = roc_auc_score(merged['TARGET'], merged['default_prob'])
                logger.info(f"Performance for predictions of {prediction_date_str} evaluated at {snapshotdate_str}: AUC = {auc:.4f}")
            else:
                logger.warning("Merged predictions and labels dataframe is empty.")
    else:
        logger.warning(f"No predictions found for {prediction_date_str} (6 months prior to monitoring date). Skipping performance (AUC) calculation.")

    # 2. Drift Detection (PSI) for current month's scored data
    model_name, ts, metadata = select_champion(args.model)
    model, encoders, feature_names, metadata = load_artefacts(model_name, ts)
    train_medians = build_train_medians(spark, encoders, feature_names)

    # Determine top 5 features to monitor from champion model metadata
    if model_name == "xgboost":
        top_feats = list(metadata["xgb_top10_features"].keys())[:5]
    else:
        top_feats = list(metadata["lr_top10_features"].keys())[:5]
    logger.info(f"Top 5 champion features selected for drift monitoring: {top_feats}")

    # Load baseline training features & score them to get baseline predictions
    logger.info("Computing baseline predictions and feature distributions from train split...")
    train_features_sdf = spark.read.parquet("datamart/gold/feature_store/train")
    train_features_df = train_features_sdf.toPandas()
    X_train, _ = production_preprocess(train_features_df, encoders, train_medians, feature_names)
    baseline_preds = model.predict_proba(X_train)[:, 1]

    # Load current month predictions
    current_pred_path = f"datamart/predictions/{snapshotdate_str}"
    current_preds_df = pd.DataFrame()
    if os.path.exists(current_pred_path):
        current_preds_df = spark.read.parquet(current_pred_path).toPandas()

    # Load current month features
    start_date = snapshotdate.replace(day=1)
    end_date = start_date + relativedelta(months=1)
    silver_app_path = "datamart/silver/application"
    
    current_features_df = pd.DataFrame()
    if os.path.exists(silver_app_path) and not current_preds_df.empty:
        silver_app = spark.read.parquet(silver_app_path)
        current_month_app = silver_app.filter(
            (F.col("application_date") >= F.lit(start_date.date().strftime("%Y-%m-%d"))) &
            (F.col("application_date") < F.lit(end_date.date().strftime("%Y-%m-%d")))
        ).select("SK_ID_CURR")
        
        gold_feat_dir = "datamart/gold/feature_store"
        feat_paths = [os.path.join(gold_feat_dir, s) for s in splits if os.path.exists(os.path.join(gold_feat_dir, s))]
        if feat_paths:
            features_sdf = spark.read.parquet(*feat_paths)
            current_features_df = features_sdf.join(current_month_app, on="SK_ID_CURR", how="inner").toPandas()

    prediction_psi = None
    feature_psis = [None] * len(top_feats)

    if not current_preds_df.empty and not current_features_df.empty:
        logger.info(f"Computing drift metrics for month {snapshotdate_str}...")
        # Prediction PSI
        prediction_psi = calculate_psi(baseline_preds, current_preds_df['default_prob'].values)
        logger.info(f"Prediction PSI: {prediction_psi:.4f}")

        # Feature PSI
        X_current, _ = production_preprocess(current_features_df, encoders, train_medians, feature_names)
        for i, feat in enumerate(top_feats):
            if feat in X_train.columns and feat in X_current.columns:
                feature_psis[i] = calculate_psi(X_train[feat].values, X_current[feat].values)
                logger.info(f"Feature Drift PSI - {feat}: {feature_psis[i]:.4f}")
    else:
        logger.warning(f"No current predictions or features found for {snapshotdate_str}. Skipping PSI drift metrics.")

    # Save monitoring metrics
    monitor_results = {
        'snapshot_date': snapshotdate_str,
        'auc': float(auc) if auc is not None else None,
        'avg_prediction': float(current_preds_df['default_prob'].mean()) if not current_preds_df.empty else None,
        'prediction_psi': prediction_psi,
    }
    for i, feat in enumerate(top_feats):
        monitor_results[f'drift_{feat}'] = feature_psis[i]

    monitor_dir = "datamart/gold/model_monitoring"
    os.makedirs(monitor_dir, exist_ok=True)
    output_path = os.path.join(monitor_dir, f"monitor_{snapshotdate_str.replace('-', '_')}.parquet")

    # Define schema dynamically
    fields = [
        T.StructField("snapshot_date", T.StringType(), False),
        T.StructField("auc", T.DoubleType(), True),
        T.StructField("avg_prediction", T.DoubleType(), True),
        T.StructField("prediction_psi", T.DoubleType(), True),
    ]
    for feat in top_feats:
        fields.append(T.StructField(f"drift_{feat}", T.DoubleType(), True))
        
    schema = T.StructType(fields)
    spark.createDataFrame(pd.DataFrame([monitor_results]), schema=schema).write.mode("overwrite").parquet(output_path)
    logger.info(f"Monitoring results successfully saved to {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()
