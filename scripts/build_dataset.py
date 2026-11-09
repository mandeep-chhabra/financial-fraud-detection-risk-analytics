"""Build the model-ready feature table from the raw IEEE-CIS files.

Outputs
-------
data/processed/ieee_features.parquet        : engineered features + split label
data/processed/data_quality_report.json     : audit used in the documentation
data/processed/feature_dictionary.json      : feature groups, rationale, leakage notes
models/feature_encoder.json                 : category/frequency encodings for scoring
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Adding ~460 columns one at a time makes pandas warn about fragmentation. The
# frame is written straight to parquet afterwards, so the warning is noise here.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from src.config import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_DICTIONARY,
    FEATURE_ENCODER_PATH,
    PROCESSED_PARQUET,
    QUALITY_REPORT,
    RANDOM_STATE,
)
from src.data.load import (  # noqa: E402
    assign_split,
    data_quality_report,
    dataset_fingerprint,
    load_raw,
    split_summary,
)
from src.features.engineering import (  # noqa: E402
    add_category_codes,
    add_frequency_features,
    add_history_features,
    add_row_features,
    compute_frequency_maps,
    feature_dictionary,
)
from src.utils import ensure_dir, get_logger, save_json, set_seed  # noqa: E402

logger = get_logger("build_dataset")

META_COLUMNS = [
    "txn_id", "transaction_dt", "txn_hour", "day_index", "amount", "product_cd",
    "card4", "card6", "card1", "addr1", "p_emaildomain", "has_identity", "isFraud", "split",
]


def build(force: bool = False) -> pd.DataFrame:
    ensure_dir(PROCESSED_PARQUET.parent)
    if PROCESSED_PARQUET.exists() and not force:
        logger.info("Feature table already exists at %s (use --force to rebuild)", PROCESSED_PARQUET)
        return pd.read_parquet(PROCESSED_PARQUET, columns=META_COLUMNS)

    set_seed(RANDOM_STATE)
    started = time.time()

    logger.info("Loading raw transaction + identity files")
    df = load_raw()
    logger.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])

    # Audit the raw frame before any feature work so the report describes the data
    # as published, not the engineered version.
    logger.info("Running data-quality audit")
    report = data_quality_report(df)

    # assign_split reads the original TransactionDT column, so split first.
    df["split"] = assign_split(df)
    df = df.rename(columns={"TransactionID": "txn_id", "TransactionDT": "transaction_dt"})

    logger.info("Encoding categorical columns")
    df, mappings = add_category_codes(df)

    logger.info("Frequency encodings fitted on the training period only")
    train_mask = (df["split"] == "train").to_numpy()
    frequency_maps = compute_frequency_maps(df.loc[train_mask])
    df = add_frequency_features(df, frequency_maps)

    logger.info("Row-level features")
    df = add_row_features(df)

    logger.info("Past-only history features (slowest step)")
    df = add_history_features(df)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Feature engineering did not produce {missing}")

    df["amount"] = df["TransactionAmt"].astype("float32")
    df["product_cd"] = df["ProductCD"].astype("string")
    # Display/analysis aliases used by the SQL layer and the dashboard.
    df["p_emaildomain"] = df["P_emaildomain"].astype("string")
    for column in ("card4", "card6"):
        df[column] = df[column].astype("string")

    keep = list(
        dict.fromkeys(
            [
                "txn_id",
                "transaction_dt",
                "txn_hour",
                "day_index",
                "amount",
                "product_cd",
                "card4",
                "card6",
                "card1",
                "addr1",
                "p_emaildomain",
                "has_identity",
                "isFraud",
                "split",
                *FEATURE_COLUMNS,
            ]
        )
    )
    df = df[keep]
    df["split"] = df["split"].astype("category")

    logger.info("Writing %s", PROCESSED_PARQUET)
    df.to_parquet(PROCESSED_PARQUET, index=False, compression="snappy")

    report["fingerprint"] = dataset_fingerprint(
        df.rename(columns={"txn_id": "TransactionID", "transaction_dt": "TransactionDT"})
    )
    report["split_summary"] = split_summary(
        df.rename(columns={"txn_id": "TransactionID", "transaction_dt": "TransactionDT"})
    ).to_dict(orient="records")
    report["engineered_features"] = len(FEATURE_COLUMNS)
    report["build_seconds"] = round(time.time() - started, 1)
    save_json(report, QUALITY_REPORT)
    save_json(feature_dictionary(), FEATURE_DICTIONARY)
    save_json(
        {"category_mappings": mappings, "frequency_maps": frequency_maps},
        FEATURE_ENCODER_PATH,
    )

    logger.info(
        "Done in %.1fs. Rows=%s fraud=%s features=%s",
        time.time() - started,
        f"{len(df):,}",
        f"{int(df.isFraud.sum()):,}",
        len(FEATURE_COLUMNS),
    )
    return df


if __name__ == "__main__":
    build(force="--force" in sys.argv)
