import logging
import os
import pickle
import json
from datetime import datetime
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pyspark
import pyspark.sql.functions as F
 
import config
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
 
CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE", "prev_last_status",
]
 
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_split(spark, split: str) -> pd.DataFrame:
    features = spark.read.parquet(os.path.join(config.GOLD_DIR, "feature_store", split))
    labels   = spark.read.parquet(os.path.join(config.GOLD_DIR, "label_store",   split))
    df = features.join(labels, on=ID_COL, how="inner")
    return df.toPandas()
 
 
def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = False):
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("missing")
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "missing")
            df[col] = le.transform(df[col])
    return df, encoders
 
 
def fill_nulls(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna("missing")
        else:
            median_val = X[col].median()
            X[col] = X[col].fillna(0 if pd.isna(median_val) else median_val)
    # Final safety net — catches anything remaining
    X = X.fillna(0)
    return X
 
 
def prepare(df: pd.DataFrame, encoders: dict = None, fit: bool = False):
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)
    y = df[TARGET_COL].values
    X = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore")
    X = fill_nulls(X)
    assert not X.isnull().any().any(), "NaNs still present after fill!"
    return X, y, encoders
 
 
def evaluate(name: str, model, X, y, split_name: str):
    prob = model.predict_proba(X)[:, 1]
    auc  = roc_auc_score(y, prob)
    pred = (prob >= 0.5).astype(int)
    logger.info(f"[{name}] {split_name} AUC-ROC: {auc:.4f}")
    logger.info(f"\n{classification_report(y, pred, digits=4)}")
    return auc
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("HomeCredit-Training")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
 
    logger.info("Loading train split...")
    train_df = load_split(spark, "train")
    logger.info(f"Train shape: {train_df.shape}  |  default rate: {train_df[TARGET_COL].mean():.3f}")
 
    logger.info("Loading val split...")
    val_df = load_split(spark, "val")
 
    logger.info("Loading test split...")
    test_df = load_split(spark, "test")
 
    logger.info("Loading OOT split...")
    oot_df = load_split(spark, "oot")
 
    X_train, y_train, encoders = prepare(train_df, fit=True)
    X_val,   y_val,   _        = prepare(val_df,   encoders=encoders)
    X_test,  y_test,  _        = prepare(test_df,  encoders=encoders)
    X_oot,   y_oot,   _        = prepare(oot_df,   encoders=encoders)
 
    feature_names = list(X_train.columns)
    results = {}
 
    # -------------------------------------------------------------------
    # Model 1: Logistic Regression
    # -------------------------------------------------------------------
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED, n_jobs=-1)
    lr.fit(X_train, y_train)
 
    results["logistic_regression"] = {
        "val_auc":  evaluate("LR", lr, X_val,  y_val,  "val"),
        "test_auc": evaluate("LR", lr, X_test, y_test, "test"),
        "oot_auc":  evaluate("LR", lr, X_oot,  y_oot,  "OOT"),
    }
 
    lr_importance = pd.Series(np.abs(lr.coef_[0]), index=feature_names).sort_values(ascending=False)
    logger.info(f"[LR] Top 10 features:\n{lr_importance.head(10)}")
 
    # -------------------------------------------------------------------
    # Model 2: XGBoost
    # -------------------------------------------------------------------
    logger.info("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
 
    results["xgboost"] = {
        "val_auc":  evaluate("XGB", xgb_model, X_val,  y_val,  "val"),
        "test_auc": evaluate("XGB", xgb_model, X_test, y_test, "test"),
        "oot_auc":  evaluate("XGB", xgb_model, X_oot,  y_oot,  "OOT"),
    }
 
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    logger.info(f"[XGB] Top 10 features:\n{xgb_importance.head(10)}")
 
    # -------------------------------------------------------------------
    # Save artefacts
    # -------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    lr_path   = os.path.join(MODEL_DIR, f"logistic_regression_{ts}.pkl")
    xgb_path  = os.path.join(MODEL_DIR, f"xgboost_{ts}.pkl")
    enc_path  = os.path.join(MODEL_DIR, f"encoders_{ts}.pkl")
    meta_path = os.path.join(MODEL_DIR, f"metadata_{ts}.json")
 
    with open(lr_path,  "wb") as f: pickle.dump(lr,        f)
    with open(xgb_path, "wb") as f: pickle.dump(xgb_model, f)
    with open(enc_path, "wb") as f: pickle.dump(encoders,   f)
 
    metadata = {
        "timestamp": ts,
        "feature_names": feature_names,
        "results": results,
        "xgb_top10_features": xgb_importance.head(10).to_dict(),
        "lr_top10_features":  lr_importance.head(10).to_dict(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
 
    logger.info(f"Models saved to {MODEL_DIR}/")
    logger.info(f"  LR:       {lr_path}")
    logger.info(f"  XGBoost:  {xgb_path}")
    logger.info(f"  Encoders: {enc_path}")
    logger.info(f"  Metadata: {meta_path}")
 
    logger.info("\n===== RESULTS SUMMARY =====")
    for model_name, scores in results.items():
        logger.info(f"{model_name:25s}  val={scores['val_auc']:.4f}  test={scores['test_auc']:.4f}  oot={scores['oot_auc']:.4f}")
 
    spark.stop()
 
 
if __name__ == "__main__":
    main()