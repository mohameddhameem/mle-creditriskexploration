import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Repo root is the same directory as this file
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw CSVs — inside the repo, gitignored
DATA_DIR = os.path.join(_REPO_ROOT, "data")

# Output directories for each medallion layer (inside repo, gitignored)
BRONZE_DIR = os.path.join(_REPO_ROOT, "datamart", "bronze")
SILVER_DIR = os.path.join(_REPO_ROOT, "datamart", "silver")
GOLD_DIR   = os.path.join(_REPO_ROOT, "datamart", "gold")

# ---------------------------------------------------------------------------
# Source file names
# ---------------------------------------------------------------------------
RAW_FILES = {
    "application":    "application_train.parquet",
    "bureau":         "bureau.parquet",
    "bureau_balance": "bureau_balance.parquet",
    "previous_app":   "previous_application.parquet",
    "pos_cash":       "POS_CASH_balance.parquet",
    "installments":   "installments_payments.parquet",
    "credit_card":    "credit_card_balance.parquet",
}

# ---------------------------------------------------------------------------
# Date constants (synthetic timeline for OOT split)
# Used in Gold layer splits.
# ---------------------------------------------------------------------------
TRAIN_START = "2018-01-01"
TRAIN_END   = "2019-12-31"   # inclusive upper bound for train/val/test pool
OOT_START   = "2020-01-01"
OOT_END     = "2020-12-31"

# Train / val / test proportions (applied within 2018-2019 data only)
SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.20
SPLIT_TEST  = 0.10

RANDOM_SEED = 42
