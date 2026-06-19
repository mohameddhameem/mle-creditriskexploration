import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import pyspark
import pyspark.sql.types as T

import config
from train import (
    encode_categoricals,
    prepare,
    load_split,
    CATEGORICAL_COLS,
    ID_COL,
    TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ---------------------------------------------------------------------------
# 1. Champion selection
# ---------------------------------------------------------------------------

def select_champion(model_arg: str) -> tuple:
    """
    Scans models/metadata_*.json. Selects the (model_name, ts) with the highest
    oot_auc for model_arg, or globally when model_arg == "champion".
    Writes models/champion.json. Returns (model_name, ts, metadata_dict).
    """
    meta_files = sorted(
        f for f in os.listdir(MODEL_DIR)
        if f.startswith("metadata_") and f.endswith(".json")
    )
    if not meta_files:
        logger.error("No metadata_*.json files found in %s", MODEL_DIR)
        sys.exit(1)

    best = None  # (oot_auc, model_name, ts, metadata_dict)
    for fname in meta_files:
        with open(os.path.join(MODEL_DIR, fname)) as fh:
            meta = json.load(fh)
        ts = meta["timestamp"]
        for mname, scores in meta["results"].items():
            if model_arg not in ("champion", mname):
                continue
            oot = scores.get("oot_auc", -1)
            if best is None or oot > best[0]:
                best = (oot, mname, ts, meta)

    if best is None:
        logger.error("No model matching --model '%s' found in any metadata file", model_arg)
        sys.exit(1)

    oot_auc, model_name, ts, metadata = best
    logger.info("Champion: %s @ %s  oot_auc=%.4f", model_name, ts, oot_auc)

    champion = {
        "model_name":       model_name,
        "model_version":    ts,
        "pkl_path":         os.path.join(MODEL_DIR, f"{model_name}_{ts}.pkl"),
        "encoders_path":    os.path.join(MODEL_DIR, f"encoders_{ts}.pkl"),
        "selection_metric": "oot_auc",
        "selected_value":   oot_auc,
    }
    with open(os.path.join(MODEL_DIR, "champion.json"), "w") as fh:
        json.dump(champion, fh, indent=2)

    return model_name, ts, metadata


# ---------------------------------------------------------------------------
# 2. Artefact loading
# ---------------------------------------------------------------------------

def load_artefacts(model_name: str, ts: str) -> tuple:
    """
    Loads and returns (model, encoders, feature_names, metadata) for the given
    (model_name, ts). Fails loudly if any file is missing.
    """
    paths = {
        "model":    os.path.join(MODEL_DIR, f"{model_name}_{ts}.pkl"),
        "encoders": os.path.join(MODEL_DIR, f"encoders_{ts}.pkl"),
        "metadata": os.path.join(MODEL_DIR, f"metadata_{ts}.json"),
    }
    for label, p in paths.items():
        if not os.path.exists(p):
            logger.error("Required artefact missing (%s): %s", label, p)
            sys.exit(1)

    with open(paths["model"], "rb") as fh:
        model = pickle.load(fh)
    with open(paths["encoders"], "rb") as fh:
        encoders = pickle.load(fh)
    with open(paths["metadata"]) as fh:
        metadata = json.load(fh)

    feature_names = metadata["feature_names"]
    logger.info(
        "Loaded artefacts: %s @ %s  (%d features)", model_name, ts, len(feature_names)
    )
    return model, encoders, feature_names, metadata


# ---------------------------------------------------------------------------
# 3. Feature loading (no label join)
# ---------------------------------------------------------------------------

def load_features(spark, path: str) -> pd.DataFrame:
    """Reads a feature_store parquet dir without joining label_store."""
    logger.info("Loading features from %s", path)
    return spark.read.parquet(path).toPandas()


# ---------------------------------------------------------------------------
# 4. Train-median reference
# ---------------------------------------------------------------------------

def build_train_medians(spark, encoders: dict, feature_names: list) -> dict:
    """
    Loads the train split, applies the champion encoders (fit=False), drops
    ID_COL + TARGET_COL, then mirrors fill_nulls' per-column median logic to
    produce the frozen train-split median map used for production imputation.
    Returns {col: float}.
    """
    logger.info("Building train-split median reference …")
    train_df = load_split(spark, "train")
    train_df, _ = encode_categoricals(train_df, encoders=encoders, fit=False)
    X = train_df.drop(columns=[ID_COL, TARGET_COL], errors="ignore")

    obj_cols = [c for c in X.columns if X[c].dtype == object]
    if obj_cols:
        logger.warning("Unexpected object columns after encoding train split: %s", obj_cols)

    medians: dict = {}
    for col in X.columns:
        m = X[col].median()
        medians[col] = 0.0 if pd.isna(m) else float(m)

    logger.info("Train medians computed for %d columns", len(medians))
    return medians


# ---------------------------------------------------------------------------
# 5. Production preprocessing
# ---------------------------------------------------------------------------

def production_preprocess(
    df: pd.DataFrame,
    encoders: dict,
    train_medians: dict,
    feature_names: list,
) -> tuple:
    """
    Replicates train.py preprocessing order with two correctness fixes:
      (i)  Categorical guard: for columns where 'missing' IS in classes_, the
           standard path (unknowns → 'missing') is preserved exactly as training
           did it. Only when 'missing' was never a training class does the guard
           activate, mapping unknowns to classes_[0] to prevent transform crash.
      (ii) Null imputation uses frozen train-split medians, not per-split medians.
    Returns (X: pd.DataFrame, ids: pd.Series).
    """
    df = df.copy()

    # (a) Categorical encoding
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("missing")
        le = encoders[col]
        known = set(le.classes_)

        if "missing" in known:
            # Standard path — identical to encode_categoricals in train.py.
            # Unknowns map to "missing", which is a known class, so transform is safe.
            df[col] = df[col].apply(lambda x, k=known: x if x in k else "missing")
        else:
            # Guard: "missing" was never in training classes for this column
            # (e.g. CODE_GENDER had no nulls in train). Using "missing" here would
            # crash transform, so fall back to classes_[0] for any unknown value.
            fallback = le.classes_[0]
            n_unknown = (~df[col].isin(known)).sum()
            if n_unknown > 0:
                logger.warning(
                    "Column %r: %d value(s) not in LabelEncoder classes_ and "
                    "'missing' is not a known class — mapping to fallback %r",
                    col, n_unknown, fallback,
                )
            df[col] = df[col].apply(lambda x, k=known, fb=fallback: x if x in k else fb)

        df[col] = le.transform(df[col])

    # (b) Stash ID; drop TARGET if present (labeled splits)
    ids = df[ID_COL].copy()
    drop_cols = [ID_COL] + ([TARGET_COL] if TARGET_COL in df.columns else [])
    X = df.drop(columns=drop_cols, errors="ignore")

    # (c) Fill nulls using frozen train medians — mirrors fill_nulls line-for-line
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna("missing")
        else:
            X[col] = X[col].fillna(train_medians.get(col, 0.0))
    X = X.fillna(0)  # safety net

    # (d) Reorder to feature_names and assert correctness
    missing_cols = [c for c in feature_names if c not in X.columns]
    if missing_cols:
        logger.error("Feature columns absent from scoring data: %s", missing_cols)
        sys.exit(1)
    X = X[feature_names]

    assert X.shape[1] == len(feature_names), (
        f"Column count mismatch: got {X.shape[1]}, expected {len(feature_names)}"
    )
    assert not X.isnull().any().any(), "NaNs present after production preprocessing"

    return X, ids


# ---------------------------------------------------------------------------
# 6a. Parity self-test
# ---------------------------------------------------------------------------

def run_parity(
    model,
    encoders: dict,
    spark,
    split: str,
    metadata: dict,
    model_name: str,
) -> float:
    """
    Calls prepare() verbatim (as train.py does) on the labeled split and compares
    the resulting AUC to the value recorded in metadata. Aborts if delta > 1e-3,
    which indicates wrong artefacts or re-materialized Gold.
    Returns the parity AUC.
    """
    logger.info("[parity] Loading '%s' split …", split)
    df = load_split(spark, split)
    X, y, _ = prepare(df, encoders=encoders, fit=False)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)

    expected = metadata["results"].get(model_name, {}).get(f"{split}_auc")
    if expected is None:
        logger.warning("[parity] No %s_auc in metadata for %s; skipping delta check", split, model_name)
        logger.info("[parity] %s AUC = %.4f", split, auc)
        return auc

    delta = abs(auc - expected)
    logger.info(
        "[parity] %s  computed=%.4f  metadata=%.4f  delta=%.6f",
        split, auc, expected, delta,
    )
    if delta > 1e-3:
        logger.error(
            "[parity] FAIL — AUC delta %.6f > 1e-3. "
            "Likely cause: wrong model/encoder pkl or Gold was re-materialised. Aborting.",
            delta,
        )
        sys.exit(1)
    logger.info("[parity] PASS")
    return auc


# ---------------------------------------------------------------------------
# 6b. Production evaluation (skew delta)
# ---------------------------------------------------------------------------

def run_evaluate(
    predictions_df: pd.DataFrame,
    spark,
    split: str,
    parity_auc: float | None,
    metadata: dict,
    model_name: str,
) -> None:
    """
    Joins predictions to label_store/{split} and computes the production AUC
    (train-median imputation path). Prints production_auc, parity_auc, and their
    delta — the quantified train-serve skew.
    """
    label_path = os.path.join(config.GOLD_DIR, "label_store", split)
    logger.info("[evaluate] Loading label_store/%s …", split)
    labels = spark.read.parquet(label_path).toPandas()

    merged = predictions_df[[ID_COL, "default_prob"]].merge(labels, on=ID_COL, how="inner")
    if merged.empty:
        logger.error("[evaluate] No rows after joining predictions to label_store/%s", split)
        sys.exit(1)

    prod_auc = roc_auc_score(merged[TARGET_COL], merged["default_prob"])
    meta_auc = metadata["results"].get(model_name, {}).get(f"{split}_auc")

    logger.info("[evaluate] %-42s %.4f", "Production AUC  (train-median imputation):", prod_auc)
    if parity_auc is not None:
        logger.info(
            "[evaluate] %-42s %.4f", "Parity AUC      (per-split-median imputation):", parity_auc
        )
        logger.info(
            "[evaluate] %-42s %+.4f", "Skew delta      (production − parity):", prod_auc - parity_auc
        )
    if meta_auc is not None:
        logger.info("[evaluate] %-42s %.4f", f"Metadata {split}_auc:", meta_auc)


# ---------------------------------------------------------------------------
# 7. Write predictions
# ---------------------------------------------------------------------------

def write_predictions(
    spark,
    ids: pd.Series,
    probs: np.ndarray,
    model_name: str,
    ts: str,
    feature_path: str,
    run_date: str,
) -> pd.DataFrame:
    """
    Assembles the predictions DataFrame and writes it as parquet (overwrite) to
    config.PREDICTIONS_DIR/{run_date}/. Returns the pandas DataFrame for --evaluate.
    """
    scored_at = datetime.now(timezone.utc)
    out_pd = pd.DataFrame({
        ID_COL:               ids.values,
        "default_prob":       probs,
        "model_name":         model_name,
        "model_version":      ts,
        "scored_at":          scored_at,
        "feature_store_path": feature_path,
    })

    schema = T.StructType([
        T.StructField(ID_COL,               T.LongType(),      False),
        T.StructField("default_prob",        T.DoubleType(),    False),
        T.StructField("model_name",          T.StringType(),    False),
        T.StructField("model_version",       T.StringType(),    False),
        T.StructField("scored_at",           T.TimestampType(), False),
        T.StructField("feature_store_path",  T.StringType(),    False),
    ])
    out_path = os.path.join(config.PREDICTIONS_DIR, run_date)
    os.makedirs(out_path, exist_ok=True)
    spark.createDataFrame(out_pd, schema=schema).write.mode("overwrite").parquet(out_path)
    logger.info("Predictions written → %s  (%d rows)", out_path, len(out_pd))
    return out_pd


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference — Home Credit default risk")
    parser.add_argument(
        "--model", default="champion",
        choices=["champion", "xgboost", "logistic_regression"],
        help="Model to use (default: champion by oot_auc)",
    )
    parser.add_argument(
        "--split", default="oot",
        choices=["train", "val", "test", "oot"],
        help="Gold split to score (default: oot); ignored when --feature-path is set",
    )
    parser.add_argument(
        "--feature-path", default=None, metavar="DIR",
        help="Path to a feature_store parquet dir for unlabeled scoring; "
             "disables --parity and --evaluate",
    )
    parser.add_argument(
        "--run-date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Output partition label YYYY-MM-DD (default: UTC today)",
    )
    parser.add_argument(
        "--parity", action="store_true",
        help="Run AUC parity self-test against metadata values (labeled splits only)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Compute production AUC and skew delta vs parity AUC (labeled splits only)",
    )
    args = parser.parse_args()

    if args.feature_path and (args.parity or args.evaluate):
        logger.warning(
            "--feature-path is set; --parity and --evaluate require labels and are disabled"
        )
        args.parity = False
        args.evaluate = False

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("HomeCredit-Inference")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Champion selection
    model_name, ts, _ = select_champion(args.model)

    # 2. Artefact loading
    model, encoders, feature_names, metadata = load_artefacts(model_name, ts)

    # 3. Feature loading
    feature_path = (
        args.feature_path
        if args.feature_path
        else os.path.join(config.GOLD_DIR, "feature_store", args.split)
    )
    df_score = load_features(spark, feature_path)
    logger.info("Scoring %d rows", len(df_score))

    # 4. Train-median reference
    train_medians = build_train_medians(spark, encoders, feature_names)

    # 5. Production preprocessing
    X, ids = production_preprocess(df_score, encoders, train_medians, feature_names)

    # 6. Predict
    probs = model.predict_proba(X)[:, 1]
    logger.info(
        "Score distribution: mean=%.4f  min=%.4f  max=%.4f",
        float(probs.mean()), float(probs.min()), float(probs.max()),
    )

    # 7. Write
    predictions_df = write_predictions(
        spark, ids, probs, model_name, ts, feature_path, args.run_date
    )

    # 8. Optional self-tests
    parity_auc = None
    if args.parity:
        parity_auc = run_parity(model, encoders, spark, args.split, metadata, model_name)

    if args.evaluate:
        run_evaluate(predictions_df, spark, args.split, parity_auc, metadata, model_name)

    spark.stop()
    logger.info("Done.")


if __name__ == "__main__":
    main()
