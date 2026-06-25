from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_FRAUD_CSV = RAW_DATA_DIR / "fraud.csv"

REPORTS_DIR = PROJECT_ROOT / "model" / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "model" / "artifacts"

TARGET = "isFraud"
DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]

TYPE_ENCODING = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}
TYPE_COLUMN = "type"

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
BOXPLOT_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

ENGINEERED_NUMERIC = [
    "errorBalanceOrig",
    "errorBalanceDest",
    "amount_log",
    "hour_of_day",
    "day_of_week",
]
ENGINEERED_FLAGS = [
    "orig_zero_after",
    "dest_zero_before",
    "dest_zero_after",
    "is_high_amount",
    "is_transfer_or_cashout",
]

USE_TEMPORAL_SPLIT = True
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
LABELED_PER_CLASS = 1000
RANDOM_SEED = 44

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}