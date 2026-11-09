"""Central configuration for the IEEE-CIS fraud detection project.

All paths derive from the repository root so the pipeline, dashboard and tests
behave identically regardless of the working directory.

Dataset: IEEE-CIS Fraud Detection (Vesta Corporation / Kaggle, 2019).
590,540 labelled transactions over 182 days, 20,663 frauds (3.499%).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IEEE_DIR = RAW_DIR / "ieee"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_TRANSACTION_CSV = IEEE_DIR / "train_transaction.csv"
RAW_IDENTITY_CSV = IEEE_DIR / "train_identity.csv"
PROCESSED_PARQUET = PROCESSED_DIR / "ieee_features.parquet"
QUALITY_REPORT = PROCESSED_DIR / "data_quality_report.json"
FEATURE_DICTIONARY = PROCESSED_DIR / "feature_dictionary.json"

# Artifact paths. Environment overrides let the test-suite use throwaway files so
# tests never mutate the artefacts produced by the real training run.
MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", MODELS_DIR / "fraud_model.joblib"))
THRESHOLD_CONFIG_PATH = Path(
    os.environ.get("FRAUD_THRESHOLD_PATH", MODELS_DIR / "risk_thresholds.json")
)
DB_PATH = Path(os.environ.get("FRAUD_DB_PATH", DATA_DIR / "fraud.db"))
FEATURE_ENCODER_PATH = Path(
    os.environ.get("FRAUD_ENCODER_PATH", MODELS_DIR / "feature_encoder.json")
)

MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_JSON = REPORTS_DIR / "model_comparison.json"
THRESHOLD_ANALYSIS_CSV = REPORTS_DIR / "threshold_analysis.csv"
TEST_METRICS_JSON = REPORTS_DIR / "test_metrics.json"
ERROR_ANALYSIS_JSON = REPORTS_DIR / "error_analysis.json"
FEATURE_ABLATION_JSON = REPORTS_DIR / "feature_set_ablation.json"
CALIBRATION_JSON = REPORTS_DIR / "calibration.json"
EDA_SUMMARY_JSON = REPORTS_DIR / "eda_summary.json"
SHAP_DIR = REPORTS_DIR / "shap"
PREDICTIONS_PARQUET = PROCESSED_DIR / "predictions.parquet"

RANDOM_STATE = 42
TARGET = "isFraud"

# --------------------------------------------------------------------- timeline
# TransactionDT is "seconds from a reference point". In the labelled training
# file it spans 86,400 .. 15,811,131 seconds, i.e. exactly 182 days, starting at
# day 1. Splits are strict forward slices of the timeline: 70% / 15% / 15%.
TXN_DT_MIN = 86_400
TXN_DT_MAX = 15_811_131
TXN_DT_SPAN = TXN_DT_MAX - TXN_DT_MIN  # 15,724,731 seconds

_SPLIT_1 = TXN_DT_MIN + int(round(0.70 * TXN_DT_SPAN))  # 11,093,712 -> day 128.4
_SPLIT_2 = TXN_DT_MIN + int(round(0.85 * TXN_DT_SPAN))  # 13,452,422 -> day 155.7

# Train: DT <= 11,093,712. Validation: up to 13,452,422. Test: after that.
TRAIN_END_DT = _SPLIT_1
VAL_END_DT = _SPLIT_2

MAX_TXN_HOUR = (TXN_DT_MAX - TXN_DT_MIN) // 3600  # 4,367 hours of history
MIN_TXN_HOUR = 0

SPLIT_NAMES = ("train", "validation", "test")

# ------------------------------------------------------------------- raw schema
# Identity frame: 24.42% of transactions have a matching row. Absence is itself
# informative, so we keep a has_identity flag instead of dropping those rows.
IDENTITY_KEY = "TransactionID"
# id_23 and id_27 are string fields (browser/proxy descriptors), not numbers.
ID_NUMERIC = (
    [f"id_{i:02d}" for i in range(1, 12)]        # id_01 .. id_11
    + ["id_13", "id_14"]
    + [f"id_{i:02d}" for i in range(17, 23)]     # id_17 .. id_22
    + ["id_24", "id_25", "id_26", "id_32"]
)

# Columns excluded from modelling:
#   TransactionID  - row identifier, used for joins only
#   TransactionDT  - raw timestamp, replaced by derived calendar features
DROP_COLUMNS = ("TransactionID", "TransactionDT")

C_COLUMNS = [f"C{i}" for i in range(1, 15)]
D_COLUMNS = [f"D{i}" for i in range(1, 16)]
V_COLUMNS = [f"V{i}" for i in range(1, 340)]
M_COLUMNS = [f"M{i}" for i in range(1, 10)]

# Low-cardinality categoricals are integer-coded and one-hot encoded for the
# linear baseline. High-cardinality ones are integer-coded for the tree models
# and frequency-encoded (train-only) for the linear baseline.
LOW_CARD_CATEGORICALS = [
    "ProductCD",
    "card4",
    "card6",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
]
HIGH_CARD_CATEGORICALS = [
    "P_emaildomain",
    "R_emaildomain",
    "id_30",
    "id_31",
    "id_33",
    "DeviceInfo",
]
CATEGORICAL_RAW = LOW_CARD_CATEGORICALS + HIGH_CARD_CATEGORICALS
CATEGORICAL_CODE_FEATURES = [f"cat_{c}" for c in CATEGORICAL_RAW]

# Features computed row-by-row from the transaction record itself.
ROW_FEATURES = [
    # amount shape
    "TransactionAmt",
    "log_amount",
    "amount_cents",
    "amount_is_round",
    "amount_is_whole",
    # calendar
    "txn_hour",
    "hour_of_day",
    "day_of_week",
    "day_index",
    "is_night",
    "is_weekend",
    # card / address
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    # match flags (T/F -> 1/0)
    *M_COLUMNS,
    # identity presence and completeness
    "has_identity",
    "identity_missing_count",
    "core_missing_count",
    # Vesta count / timedelta / engineered blocks
    *C_COLUMNS,
    *D_COLUMNS,
    *V_COLUMNS,
    *ID_NUMERIC,
    # train-only frequency encodings
    "card1_freq",
    "addr1_freq",
    "P_emaildomain_freq",
    "DeviceInfo_freq",
]

# Past-only behavioural features (computed from strictly earlier transactions).
HISTORY_FEATURES = [
    # past-only, see src/features/engineering.py
    "card1_prior_count",
    "card1_prior_amt_mean",
    "card1_prior_amt_std",
    "amount_vs_card1_mean",
    "card1_secs_since_prev",
    "card1_prior_count_24h",
    "card1_prior_count_7d",
    "uid_prior_count",
    "uid_prior_amt_mean",
    "amount_vs_uid_mean",
]

NUMERIC_FEATURES = ROW_FEATURES + HISTORY_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_CODE_FEATURES

# Feature-set ablation groups (validation only, documented in the report).
TRANSACTION_ONLY_BLOCK = (
    ROW_FEATURES
    + CATEGORICAL_CODE_FEATURES
)
NO_IDENTITY_COLUMNS = [
    c
    for c in FEATURE_COLUMNS
    if c not in {"has_identity", "identity_missing_count"}
    and not (
        c.startswith("cat_id_")
        or c in ID_NUMERIC
        or c in {"DeviceInfo_freq", "cat_DeviceInfo", "cat_DeviceType", "cat_id_33", "cat_id_31", "cat_id_30"}
    )
]
CORE_TRANSACTION_COLUMNS = [
    c
    for c in FEATURE_COLUMNS
    if c
    not in set(
        NO_IDENTITY_COLUMNS
        + HISTORY_FEATURES
        + [f"cat_{c}" for c in HIGH_CARD_CATEGORICALS]
        + ["card1_freq", "addr1_freq", "P_emaildomain_freq", "DeviceInfo_freq"]
    )
]
PRODUCTION_FEATURES = FEATURE_COLUMNS

# Neutral behavioural values used when a transaction is scored on its own, with
# no stored history for the card. Documented as a limitation of the endpoint.
HISTORY_DEFAULTS = {
    "card1_prior_count": 0.0,
    "card1_prior_amt_mean": 0.0,
    "card1_prior_amt_std": 0.0,
    "amount_vs_card1_mean": 1.0,
    "card1_secs_since_prev": -1.0,
    "card1_prior_count_24h": 0.0,
    "card1_prior_count_7d": 0.0,
    "uid_prior_count": 0.0,
    "uid_prior_amt_mean": 0.0,
    "amount_vs_uid_mean": 1.0,
}

# Illustrative operating policy (not a real banking rule set).
DEFAULT_INVESTIGATION_COST = 4.0  # USD per manual review
DEFAULT_MAX_FPR = 0.01  # review capacity: at most 1% of legitimate transactions
RISK_BAND_LABELS = ("Low Risk", "Medium Risk", "High Risk")

HIGH_RISK_ACTION = "Block / manual investigation"
MEDIUM_RISK_ACTION = "Step-up verification (OTP / call-back)"
LOW_RISK_ACTION = "Auto-approve"

# Display labels for the dashboard.
PRODUCT_LABELS = {
    "W": "W - wallet / stored value",
    "C": "C - credit / card-not-present",
    "R": "R - recurring / subscription",
    "H": "H - high-value goods",
    "S": "S - services",
}
