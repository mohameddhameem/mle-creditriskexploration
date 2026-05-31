# =============================================================================
# utils/data_processing_gold_table.py
# Gold layer: feature engineering + train/val/test/OOT split.
#
# Inputs  : datamart/silver/{application, bureau, bureau_balance,
#                            previous_app, pos_cash, installments, credit_card}
# Outputs : datamart/gold/
#               feature_store/{train, val, test, oot}/   — SK_ID_CURR + features
#               label_store/{train, val, test, oot}/      — SK_ID_CURR + TARGET
#
# Split logic
# -----------
#   OOT   : application_date >= 2020-01-01
#   Pool  : application_date in [2018-01-01, 2019-12-31]
#   Within pool: random 70 / 20 / 10 (seed 42) → train / val / test
#
# Column accounting
# -----------------
#   silver_application has 77 columns total:
#     SK_ID_CURR (join key) + application_date (split key) + TARGET (label) + 74 features
#
# Feature summary (100 features in feature_store, excluding SK_ID_CURR)
# ----------------------------------------------------------------------
#   Application pass-through : 74  (77 Silver cols minus SK_ID_CURR, TARGET, application_date)
#   Bureau                   :  9  bureau_active_count, bureau_closed_count,
#                                   bureau_total_debt, bureau_total_credit_limit,
#                                   bureau_debt_to_limit_ratio, bureau_max_overdue_amount,
#                                   bureau_worst_status, bureau_days_last_overdue,
#                                   bureau_days_since_newest_credit
#   Previous application     :  6  prev_total_count, prev_approved_ratio,
#                                   prev_refused_count, prev_avg_credit,
#                                   prev_last_credit, prev_last_status
#   Installments             :  3  inst_late_payment_count, inst_max_days_late,
#                                   inst_payment_shortfall_ratio
#   POS cash                 :  4  pos_active_loans, pos_max_dpd,
#                                   pos_recent_max_dpd, pos_completed_loans
#   Credit card              :  4  cc_utilisation, cc_recent_utilisation,
#                                   cc_recent_dpd, cc_balance_trend
#   Total features           : 100  (+ SK_ID_CURR as key = 101 cols in feature_store)
#
# Null handling
# -------------
#   Nulls arising from left joins (applicants with no supplementary history)
#   are left as-is.  Imputation is deferred to model training:
#     - XGBoost  : handles nulls natively
#     - Logistic Regression : median imputation on train split only; imputer
#                             artefact is stored and applied to val/test/OOT
# =============================================================================

import os
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import IntegerType, DoubleType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application pass-through columns
# silver_application = 77 cols: SK_ID_CURR + application_date + TARGET + 74 features
# This list contains SK_ID_CURR + the 74 features only.
# TARGET  → label_store (handled in run_gold_layer)
# application_date → split key only, dropped from both stores after splitting
# ---------------------------------------------------------------------------
_APP_PASSTHROUGH_COLS = [
    "SK_ID_CURR",
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE",
    "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL",
    "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY", "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START", "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY", "ORGANIZATION_TYPE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",
    "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE",
    "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_8",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "DAYS_EMPLOYED_MISSING",
]  # 75 entries: SK_ID_CURR + 74 features (application_date and TARGET excluded)


# ---------------------------------------------------------------------------
# Bureau features  (9 features)
# ---------------------------------------------------------------------------

def _build_bureau_features(
        silver_bureau: DataFrame,
        silver_bureau_balance: DataFrame,
) -> DataFrame:
    """
    Two-hop join: bureau_balance → bureau → application (via SK_ID_BUREAU).

    Features per SK_ID_CURR:
      bureau_active_count           : COUNT of CREDIT_ACTIVE == 'Active'
      bureau_closed_count           : COUNT of CREDIT_ACTIVE == 'Closed'
      bureau_total_debt             : SUM(AMT_CREDIT_SUM_DEBT)
      bureau_total_credit_limit     : SUM(AMT_CREDIT_SUM)
      bureau_debt_to_limit_ratio    : total_debt / total_credit_limit  (null if 0)
      bureau_max_overdue_amount     : MAX(AMT_CREDIT_MAX_OVERDUE)
      bureau_worst_status           : MAX numeric STATUS from bureau_balance
                                      (0=no DPD, 1=1-30d, 2=31-60d, 3=61-90d,
                                       4=91-120d, 5=120+d; C and X excluded)
      bureau_days_last_overdue      : MAX(CREDIT_DAY_OVERDUE) — current DPD days
                                      across all active facilities
      bureau_days_since_newest_credit : MIN(ABS(DAYS_CREDIT)) — days since the most
                                        recently opened external credit facility
    """
    # -- bureau_balance: derive worst numeric STATUS per SK_ID_BUREAU --
    # STATUS values: '0','1','2','3','4','5' = DPD bands; 'C'=paid off; 'X'=unknown
    # Cast numeric codes only; C and X map to null (ignored by MAX).
    bb_agg = (
        silver_bureau_balance
        .withColumn(
            "status_numeric",
            F.col("STATUS").cast(IntegerType()),  # C/X → null, '0'-'5' → 0-5
        )
        .groupBy("SK_ID_BUREAU")
        .agg(F.max("status_numeric").cast(IntegerType()).alias("bb_worst_status"))
    )

    # -- join bureau_balance agg onto bureau --
    bureau = silver_bureau.join(bb_agg, on="SK_ID_BUREAU", how="left")

    # -- aggregate bureau to applicant level --
    bureau_feats = (
        bureau
        .groupBy("SK_ID_CURR")
        .agg(
            F.sum(
                F.when(F.col("CREDIT_ACTIVE") == "Active", 1).otherwise(0)
            ).cast(IntegerType()).alias("bureau_active_count"),

            F.sum(
                F.when(F.col("CREDIT_ACTIVE") == "Closed", 1).otherwise(0)
            ).cast(IntegerType()).alias("bureau_closed_count"),

            F.sum("AMT_CREDIT_SUM_DEBT").cast(DoubleType()).alias("bureau_total_debt"),
            F.sum("AMT_CREDIT_SUM").cast(DoubleType()).alias("bureau_total_credit_limit"),
            F.max("AMT_CREDIT_MAX_OVERDUE").cast(DoubleType()).alias("bureau_max_overdue_amount"),
            F.max("CREDIT_DAY_OVERDUE").cast(IntegerType()).alias("bureau_days_last_overdue"),
            F.min(F.abs(F.col("DAYS_CREDIT"))).cast(IntegerType()).alias("bureau_days_since_newest_credit"),
            F.max("bb_worst_status").cast(IntegerType()).alias("bureau_worst_status"),
        )
        .withColumn(
            "bureau_debt_to_limit_ratio",
            F.when(
                F.col("bureau_total_credit_limit") > 0,
                F.col("bureau_total_debt") / F.col("bureau_total_credit_limit"),
            ).otherwise(None).cast(DoubleType()),
        )
    )
    return bureau_feats


# ---------------------------------------------------------------------------
# Previous application features  (6 features)
# ---------------------------------------------------------------------------

def _build_prev_app_features(silver_prev_app: DataFrame) -> DataFrame:
    """
    Features per SK_ID_CURR:
      prev_total_count    : total number of previous applications
      prev_approved_ratio : approved / total  (null if 0 applications)
      prev_refused_count  : COUNT where NAME_CONTRACT_STATUS == 'Refused'
      prev_avg_credit     : AVG(AMT_CREDIT) across all previous applications
      prev_last_credit    : AMT_CREDIT of most recent application (by DAYS_DECISION DESC)
      prev_last_status    : NAME_CONTRACT_STATUS of most recent application
    """
    # -- aggregate counts, ratio, avg credit --
    agg = (
        silver_prev_app
        .groupBy("SK_ID_CURR")
        .agg(
            F.count("*").cast(IntegerType()).alias("prev_total_count"),
            F.sum(
                F.when(F.col("NAME_CONTRACT_STATUS") == "Approved", 1).otherwise(0)
            ).cast(IntegerType()).alias("_approved_count"),
            F.sum(
                F.when(F.col("NAME_CONTRACT_STATUS") == "Refused", 1).otherwise(0)
            ).cast(IntegerType()).alias("prev_refused_count"),
            F.avg("AMT_CREDIT").cast(DoubleType()).alias("prev_avg_credit"),
        )
        .withColumn(
            "prev_approved_ratio",
            F.when(
                F.col("prev_total_count") > 0,
                F.col("_approved_count").cast(DoubleType()) / F.col("prev_total_count"),
            ).otherwise(None).cast(DoubleType()),
        )
        .drop("_approved_count")
    )

    # -- most-recent application: rank by DAYS_DECISION descending (0 = application date,
    #    negative = days before — so DESC puts the least-negative value first) --
    w_recent = Window.partitionBy("SK_ID_CURR").orderBy(F.col("DAYS_DECISION").desc())
    most_recent = (
        silver_prev_app
        .withColumn("_rank", F.row_number().over(w_recent))
        .filter(F.col("_rank") == 1)
        .select(
            "SK_ID_CURR",
            F.col("AMT_CREDIT").alias("prev_last_credit"),
            F.col("NAME_CONTRACT_STATUS").alias("prev_last_status"),
        )
    )

    return agg.join(most_recent, on="SK_ID_CURR", how="left")


# ---------------------------------------------------------------------------
# Installments features  (3 features)
# ---------------------------------------------------------------------------

def _build_installments_features(silver_installments: DataFrame) -> DataFrame:
    """
    Features per SK_ID_CURR:
      inst_late_payment_count     : COUNT of rows where DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
      inst_max_days_late          : MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) across late rows
                                    (0 if never late)
      inst_payment_shortfall_ratio: SUM(shortfall) / SUM(AMT_INSTALMENT)
                                    where shortfall = MAX(AMT_INSTALMENT - AMT_PAYMENT, 0)
                                    (null if total due = 0)
    """
    return (
        silver_installments
        .withColumn(
            "is_late",
            F.when(F.col("DAYS_ENTRY_PAYMENT") > F.col("DAYS_INSTALMENT"), 1).otherwise(0),
        )
        .withColumn(
            "days_late",
            F.when(
                F.col("DAYS_ENTRY_PAYMENT") > F.col("DAYS_INSTALMENT"),
                F.col("DAYS_ENTRY_PAYMENT") - F.col("DAYS_INSTALMENT"),
            ).otherwise(0).cast(IntegerType()),
        )
        .withColumn(
            "shortfall",
            F.when(
                F.col("AMT_INSTALMENT") > F.col("AMT_PAYMENT"),
                F.col("AMT_INSTALMENT") - F.col("AMT_PAYMENT"),
            ).otherwise(0).cast(DoubleType()),
        )
        .groupBy("SK_ID_CURR")
        .agg(
            F.sum("is_late").cast(IntegerType()).alias("inst_late_payment_count"),
            F.max("days_late").cast(IntegerType()).alias("inst_max_days_late"),
            F.sum("shortfall").alias("_total_shortfall"),
            F.sum("AMT_INSTALMENT").cast(DoubleType()).alias("_total_due"),
        )
        .withColumn(
            "inst_payment_shortfall_ratio",
            F.when(
                F.col("_total_due") > 0,
                F.col("_total_shortfall") / F.col("_total_due"),
            ).otherwise(None).cast(DoubleType()),
        )
        .drop("_total_shortfall", "_total_due")
    )


# ---------------------------------------------------------------------------
# POS cash features  (4 features)
# ---------------------------------------------------------------------------

def _build_pos_cash_features(silver_pos_cash: DataFrame) -> DataFrame:
    """
    Features per SK_ID_CURR:
      pos_active_loans   : COUNT of contracts with NAME_CONTRACT_STATUS == 'Active'
                           at the most recent snapshot (MONTHS_BALANCE = 0)
      pos_max_dpd        : MAX(SK_DPD) across all months
      pos_recent_max_dpd : MAX(SK_DPD) over the most recent 3 months
                           (MONTHS_BALANCE >= -2, i.e. {-2, -1, 0})
      pos_completed_loans: COUNT of contracts with NAME_CONTRACT_STATUS == 'Completed'
                           at the most recent snapshot (MONTHS_BALANCE = 0)
    """
    # most recent snapshot per SK_ID_PREV = MONTHS_BALANCE == 0
    latest = silver_pos_cash.filter(F.col("MONTHS_BALANCE") == 0)

    latest_agg = (
        latest
        .groupBy("SK_ID_CURR")
        .agg(
            F.sum(
                F.when(F.col("NAME_CONTRACT_STATUS") == "Active", 1).otherwise(0)
            ).cast(IntegerType()).alias("pos_active_loans"),
            F.sum(
                F.when(F.col("NAME_CONTRACT_STATUS") == "Completed", 1).otherwise(0)
            ).cast(IntegerType()).alias("pos_completed_loans"),
        )
    )  

    # max DPD across all months
    all_agg = (
        silver_pos_cash
        .groupBy("SK_ID_CURR")
        .agg(F.max("SK_DPD").cast(IntegerType()).alias("pos_max_dpd"))
    )

    # recent max DPD: MONTHS_BALANCE in {-2, -1, 0}
    recent_max = (
        silver_pos_cash
        .filter(F.col("MONTHS_BALANCE") >= -2)
        .groupBy("SK_ID_CURR")
        .agg(F.max("SK_DPD").cast(IntegerType()).alias("pos_recent_max_dpd"))
    )

    return (
        all_agg
        .join(recent_max, on="SK_ID_CURR", how="left")
        .join(latest_agg, on="SK_ID_CURR", how="left")
    )


# ---------------------------------------------------------------------------
# Credit card features  (4 features)
# ---------------------------------------------------------------------------

def _build_credit_card_features(silver_credit_card: DataFrame) -> DataFrame:
    """
    Features per SK_ID_CURR:
      cc_utilisation         : AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL at the most recent
                               snapshot (MAX MONTHS_BALANCE per SK_ID_CURR)
      cc_recent_utilisation  : mean utilisation over the most recent 3 months
                               (MONTHS_BALANCE >= most_recent - 2)
      cc_recent_dpd          : MAX(SK_DPD) over the most recent 3 months
      cc_balance_trend       : OLS slope of AMT_BALANCE over MONTHS_BALANCE across all months
                               (positive slope = growing balance = risk signal)
    """
    # -- identify most recent MONTHS_BALANCE per applicant --
    w_max = Window.partitionBy("SK_ID_CURR")
    cc = silver_credit_card.withColumn(
        "_max_month", F.max("MONTHS_BALANCE").over(w_max)
    )

    # -- cc_utilisation: at most recent snapshot --
    utilisation_latest = (
        cc
        .filter(F.col("MONTHS_BALANCE") == F.col("_max_month"))
        .withColumn(
            "cc_utilisation",
            F.when(
                F.col("AMT_CREDIT_LIMIT_ACTUAL") > 0,
                F.col("AMT_BALANCE") / F.col("AMT_CREDIT_LIMIT_ACTUAL"),
            ).otherwise(None).cast(DoubleType()),
        )
        # an applicant may have multiple cards — average across them at same snapshot month
        .groupBy("SK_ID_CURR")
        .agg(F.avg("cc_utilisation").cast(DoubleType()).alias("cc_utilisation"))
    )

    # -- cc_recent_utilisation and cc_recent_dpd: last 3 months --
    recent = (
        cc
        .filter(F.col("MONTHS_BALANCE") >= F.col("_max_month") - 2)
        .withColumn(
            "utilisation",
            F.when(
                F.col("AMT_CREDIT_LIMIT_ACTUAL") > 0,
                F.col("AMT_BALANCE") / F.col("AMT_CREDIT_LIMIT_ACTUAL"),
            ).otherwise(None).cast(DoubleType()),
        )
        .groupBy("SK_ID_CURR")
        .agg(
            F.avg("utilisation").cast(DoubleType()).alias("cc_recent_utilisation"),
            F.max("SK_DPD").cast(IntegerType()).alias("cc_recent_dpd"),
        )
    )

    # -- cc_balance_trend: OLS slope via regr_slope(y=AMT_BALANCE, x=MONTHS_BALANCE) --
    # regr_slope is available in Spark 3.3+; returns null when < 2 data points.
    trend = (
        silver_credit_card
        .groupBy("SK_ID_CURR")
        .agg(
            F.regr_slope(
                F.col("AMT_BALANCE").cast(DoubleType()),
                F.col("MONTHS_BALANCE").cast(DoubleType()),
            ).alias("cc_balance_trend")
        )
    )

    return (
        utilisation_latest
        .join(recent, on="SK_ID_CURR", how="left")
        .join(trend, on="SK_ID_CURR", how="left")
    )


# ---------------------------------------------------------------------------
# Train / val / test / OOT split
# ---------------------------------------------------------------------------

def _split_dataset(
        df: DataFrame,
        oot_start: str,
        train_frac: float,
        val_frac: float,
        seed: int,
) -> dict:
    """
    Split a full Gold DataFrame into 4 subsets.

    OOT  : application_date >= oot_start
    Pool : the remainder (train/val/test drawn from here)

    Returns dict: {'train': df, 'val': df, 'test': df, 'oot': df}
    """
    oot = df.filter(F.col("application_date") >= oot_start)
    pool = df.filter(F.col("application_date") < oot_start)

    test_frac = round(1.0 - train_frac - val_frac, 10)
    train_df, val_df, test_df = pool.randomSplit(
        [train_frac, val_frac, test_frac], seed=seed
    )

    return {"train": train_df, "val": val_df, "test": test_df, "oot": oot}


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_split(df: DataFrame, base_dir: str, split_name: str, store: str) -> int:
    """
    Write one split of one store (feature_store or label_store).

    Returns row count.
    """
    path = os.path.join(base_dir, store, split_name)
    df.write.mode("overwrite").parquet(path)
    cnt = df.count()
    logger.info(f"[Gold] {store}/{split_name}: {cnt:,} rows → {path}")
    return cnt


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_gold_layer(
        spark: SparkSession,
        silver_dir: str,
        gold_dir: str,
        oot_start: str,
        train_frac: float,
        val_frac: float,
        seed: int,
) -> dict:
    """
    Build Gold feature_store and label_store from Silver tables.

    Parameters
    ----------
    spark       : active SparkSession
    silver_dir  : path to datamart/silver/
    gold_dir    : path to datamart/gold/
    oot_start   : ISO date string for OOT cut-off (e.g. '2020-01-01')
    train_frac  : fraction of non-OOT pool for training (e.g. 0.70)
    val_frac    : fraction of non-OOT pool for validation (e.g. 0.20)
    seed        : random seed for split

    Returns
    -------
    dict of {split/store: row_count}
    """
    logger.info("[Gold] Loading Silver tables...")

    # -- read Silver tables --
    silver_app = spark.read.parquet(os.path.join(silver_dir, "application"))
    silver_bur = spark.read.parquet(os.path.join(silver_dir, "bureau"))
    silver_bb = spark.read.parquet(os.path.join(silver_dir, "bureau_balance"))
    silver_prev = spark.read.parquet(os.path.join(silver_dir, "previous_app"))
    silver_pos = spark.read.parquet(os.path.join(silver_dir, "pos_cash"))
    silver_inst = spark.read.parquet(os.path.join(silver_dir, "installments"))
    silver_cc = spark.read.parquet(os.path.join(silver_dir, "credit_card"))

    # -- build supplementary feature tables --
    logger.info("[Gold] Building bureau features...")
    bureau_feats = _build_bureau_features(silver_bur, silver_bb)

    logger.info("[Gold] Building previous application features...")
    prev_feats = _build_prev_app_features(silver_prev)

    logger.info("[Gold] Building installments features...")
    inst_feats = _build_installments_features(silver_inst)

    logger.info("[Gold] Building POS cash features...")
    pos_feats = _build_pos_cash_features(silver_pos)

    logger.info("[Gold] Building credit card features...")
    cc_feats = _build_credit_card_features(silver_cc)

    # -- application pass-through (select only feature cols + split keys) --
    app_core = silver_app.select(
        _APP_PASSTHROUGH_COLS + ["application_date", "TARGET"]
    )

    # -- join all feature tables onto application (left join — preserves all 307,511 rows) --
    logger.info("[Gold] Joining all feature tables...")
    gold = (
        app_core
        .join(bureau_feats, on="SK_ID_CURR", how="left")
        .join(prev_feats, on="SK_ID_CURR", how="left")
        .join(inst_feats, on="SK_ID_CURR", how="left")
        .join(pos_feats, on="SK_ID_CURR", how="left")
        .join(cc_feats, on="SK_ID_CURR", how="left")
    )

    total_rows = gold.count()
    total_cols = len(gold.columns)
    logger.info(f"[Gold] Full Gold table: {total_rows:,} rows, {total_cols} columns")

    # -- split --
    logger.info("[Gold] Splitting into train / val / test / OOT...")
    splits = _split_dataset(
        gold,
        oot_start=oot_start,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )
    for name, sdf in splits.items():
        logger.info(f"[Gold]   {name}: {sdf.count():,} rows")

    # -- feature_store: SK_ID_CURR + all features (no TARGET, no application_date) --
    feature_cols = [c for c in gold.columns if c not in ("TARGET", "application_date")]

    # -- label_store: SK_ID_CURR + TARGET only --
    label_cols = ["SK_ID_CURR", "TARGET"]

    # -- write feature_store and label_store for each split --
    results = {}
    os.makedirs(gold_dir, exist_ok=True)

    for split_name, sdf in splits.items():
        feat_df = sdf.select(feature_cols)
        label_df = sdf.select(label_cols)

        results[f"feature_store/{split_name}"] = _write_split(
            feat_df, gold_dir, split_name, "feature_store"
        )
        results[f"label_store/{split_name}"] = _write_split(
            label_df, gold_dir, split_name, "label_store"
        )

    logger.info("[Gold] Gold layer complete.")
    return results
