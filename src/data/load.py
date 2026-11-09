"""IEEE-CIS loading, schema validation, quality auditing and temporal splitting.

Two raw files are joined on ``TransactionID``:

* ``train_transaction.csv`` - 590,540 rows, 394 columns (amount, cards, address,
  Vesta C/D/V blocks, match flags).
* ``train_identity.csv``   - 144,233 rows, 41 columns (device, OS, browser,
  identity numbers). Only 24.42% of transactions have a match; the join is a
  LEFT join and the absence of identity data is kept as a feature.

The competition *test* files ship without labels, so they are not used: the
labelled file is split strictly by time instead (70/15/15).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    C_COLUMNS,
    D_COLUMNS,
    ID_NUMERIC,
    IDENTITY_KEY,
    RAW_IDENTITY_CSV,
    RAW_TRANSACTION_CSV,
    TARGET,
    TRAIN_END_DT,
    TXN_DT_MAX,
    TXN_DT_MIN,
    VAL_END_DT,
    V_COLUMNS,
)

REQUIRED_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "addr1",
]

# float32 halves the resident size of the 394-column transaction frame (~1.9 GB
# -> ~0.95 GB) without losing information: every value fits comfortably in
# float32's 7 significant digits at these magnitudes.
_FLOAT32_COLUMNS = (
    ["TransactionAmt", "card1", "card2", "card3", "card5",
     "addr1", "addr2", "dist1", "dist2"]
    + C_COLUMNS
    + D_COLUMNS
    + V_COLUMNS
    + ID_NUMERIC
)
TRANSACTION_DTYPES = {c: "float32" for c in _FLOAT32_COLUMNS}
TRANSACTION_DTYPES.update({"TransactionDT": "int32", "isFraud": "int8"})
IDENTITY_DTYPES = {c: "float32" for c in ID_NUMERIC}


class SchemaError(ValueError):
    """Raised when the raw files do not match the expected IEEE-CIS schema."""


def load_raw(
    transaction_csv: str | Path = RAW_TRANSACTION_CSV,
    identity_csv: str | Path = RAW_IDENTITY_CSV,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load and join the raw files with memory-friendly dtypes.

    ``nrows`` exists for tests and quick experiments only; the pipeline always
    consumes the full labelled file.
    """
    transaction_csv = Path(transaction_csv)
    identity_csv = Path(identity_csv)
    if not transaction_csv.exists():
        raise FileNotFoundError(
            f"Raw transactions not found at {transaction_csv}. "
            "Place train_transaction.csv and train_identity.csv in data/raw/ieee/."
        )

    txn = pd.read_csv(transaction_csv, dtype=TRANSACTION_DTYPES, nrows=nrows)
    if identity_csv.exists():
        identity = pd.read_csv(identity_csv, dtype=IDENTITY_DTYPES)
        # ``validate='one_to_one'`` fails loudly if identity ever has duplicate
        # TransactionIDs, which would silently duplicate fraud labels.
        df = txn.merge(identity, on=IDENTITY_KEY, how="left", validate="one_to_one")
        del identity
    else:
        df = txn
    del txn
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate column presence, label domain, keys and basic value integrity."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    if df.empty:
        raise SchemaError("Dataset is empty")

    if not df[IDENTITY_KEY].is_unique:
        raise SchemaError("TransactionID is not unique")

    labels = set(pd.unique(df[TARGET].dropna()))
    if not labels.issubset({0, 1}):
        raise SchemaError(f"Target must be binary 0/1, found {sorted(labels)}")

    if (df["TransactionAmt"] <= 0).any():
        raise SchemaError("Non-positive transaction amounts found")

    dt = df["TransactionDT"]
    if dt.min() < TXN_DT_MIN or dt.max() > TXN_DT_MAX:
        raise SchemaError(
            f"TransactionDT outside the expected window "
            f"[{TXN_DT_MIN}, {TXN_DT_MAX}]: got [{dt.min()}, {dt.max()}]"
        )


def assign_split(df: pd.DataFrame) -> pd.Series:
    """Return a split label per row using strictly forward time boundaries."""
    dt = df["TransactionDT"]
    split = pd.Series("test", index=df.index, dtype="object")
    split[dt <= VAL_END_DT] = "validation"
    split[dt <= TRAIN_END_DT] = "train"
    return split


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    split = assign_split(df)
    return {name: df.loc[split == name] for name in ("train", "validation", "test")}


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split row counts, day ranges and fraud prevalence."""
    split = assign_split(df)
    rows = []
    for name in ("train", "validation", "test"):
        part = df.loc[split == name]
        frauds = int(part[TARGET].sum())
        rows.append(
            {
                "split": name,
                "transactions": int(len(part)),
                "frauds": frauds,
                "fraud_rate": frauds / len(part) if len(part) else 0.0,
                "day_start": round(float(part["TransactionDT"].min()) / 86400, 2),
                "day_end": round(float(part["TransactionDT"].max()) / 86400, 2),
                "amount_sum": float(part["TransactionAmt"].sum()),
                "fraud_amount_sum": float(part.loc[part[TARGET] == 1, "TransactionAmt"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _missing_summary(df: pd.DataFrame) -> tuple[dict, dict]:
    rate = df.isna().mean().sort_values(ascending=False)
    top = {c: float(v) for c, v in rate.head(25).items()}
    buckets = {
        "no_missing": int((rate == 0).sum()),
        "under_5pct": int(((rate > 0) & (rate < 0.05)).sum()),
        "5_to_25pct": int(((rate >= 0.05) & (rate < 0.25)).sum()),
        "25_to_75pct": int(((rate >= 0.25) & (rate < 0.75)).sum()),
        "over_75pct": int((rate >= 0.75).sum()),
    }
    return top, buckets


def data_quality_report(df: pd.DataFrame) -> dict:
    """Audit the joined frame: missingness, keys, imbalance, leakage candidates."""
    label = df[TARGET]
    frauds = int(label.sum())
    rows = int(len(df))
    has_identity = df["id_01"].notna() if "id_01" in df.columns else pd.Series(False, index=df.index)

    by_product = (
        df.groupby("ProductCD", observed=True)[TARGET]
        .agg(transactions="count", frauds="sum")
        .assign(fraud_rate=lambda d: d["frauds"] / d["transactions"])
        .reset_index()
        .rename(columns={"ProductCD": "product_cd"})
    )
    by_card4 = (
        df.groupby("card4", observed=True, dropna=False)[TARGET]
        .agg(transactions="count", frauds="sum")
        .assign(fraud_rate=lambda d: d["frauds"] / d["transactions"])
        .reset_index()
    )
    by_card6 = (
        df.groupby("card6", observed=True, dropna=False)[TARGET]
        .agg(transactions="count", frauds="sum")
        .assign(fraud_rate=lambda d: d["frauds"] / d["transactions"])
        .reset_index()
    )

    missing_top, missing_buckets = _missing_summary(df)
    amount = df["TransactionAmt"].astype("float64")

    report = {
        "dataset": "IEEE-CIS Fraud Detection (Vesta Corporation, Kaggle 2019)",
        "rows": rows,
        "columns": int(df.shape[1]),
        "days_covered": round((int(df["TransactionDT"].max()) - int(df["TransactionDT"].min())) / 86400, 2),
        "frauds": frauds,
        "legitimate": rows - frauds,
        "fraud_rate": frauds / rows,
        "imbalance_ratio": (rows - frauds) / max(frauds, 1),
        "missing_top": missing_top,
        "missing_buckets": missing_buckets,
        "duplicate_transaction_ids": int(df[IDENTITY_KEY].duplicated().sum()),
        "identity_rows": int(has_identity.sum()),
        "identity_coverage": float(has_identity.mean()),
        "fraud_rate_with_identity": float(df.loc[has_identity, TARGET].mean()),
        "fraud_rate_without_identity": float(df.loc[~has_identity, TARGET].mean()),
        "unique_cards": int(df["card1"].nunique()),
        "unique_addresses": int(df["addr1"].nunique()),
        "amount_stats": {
            "mean": float(amount.mean()),
            "median": float(amount.median()),
            "p99": float(amount.quantile(0.99)),
            "max": float(amount.max()),
            "total": float(amount.sum()),
        },
        "fraud_amount_stats": {
            "mean": float(amount[label == 1].mean()),
            "median": float(amount[label == 1].median()),
            "total": float(amount[label == 1].sum()),
        },
        "fraud_amount_share": float(amount[label == 1].sum() / amount.sum()),
        "by_product": by_product.to_dict(orient="records"),
        "by_card4": by_card4.fillna({"card4": "missing"}).to_dict(orient="records"),
        "by_card6": by_card6.fillna({"card6": "missing"}).to_dict(orient="records"),
        "fraud_product_counts": {
            str(k): int(v) for k, v in df.loc[label == 1, "ProductCD"].value_counts().items()
        },
        "split": split_summary(df).to_dict(orient="records"),
        "leakage_audit": {
            "approach": (
                "Every feature is either a field of the transaction record itself, "
                "a train-only frequency encoding, or an aggregate over strictly "
                "earlier transactions of the same card / uid. No target encoding, "
                "no full-sample aggregate and no post-outcome field is used."
            ),
            "excluded_columns": {
                "TransactionID": "row key, no predictive meaning",
                "TransactionDT": "raw timestamp; replaced by derived calendar features",
                "isFraud": "the label itself",
            },
            "identity_note": (
                "Identity fields are recorded at authorisation time, so they are "
                "legitimate model inputs. Identity *presence* correlates with fraud "
                f"({100 * df.loc[has_identity, TARGET].mean():.2f}% vs "
                f"{100 * df.loc[~has_identity, TARGET].mean():.2f}%), which is a "
                "real signal (card-not-present / new-device traffic), not leakage."
            ),
        },
        "known_limitations": [
            "Anonymised, feature-engineered data: Vesta's own V/C/D blocks cannot be "
            "reconstructed, so feature meaning is only partly interpretable.",
            "No merchant category, chargeback outcome, customer tenure or label "
            "timestamp is available; a transaction labelled legitimate may simply "
            "not have been charged back yet.",
            "Identity data exists for only 24.4% of transactions; the model must run "
            "with and without it.",
            "Fraud prevalence (3.5%) is far higher than in real card portfolios "
            "(typically <0.2%), which flatters precision at a fixed threshold.",
            "Rare Vesta features are near-duplicates of each other, so SHAP "
            "attribution is spread across correlated columns.",
        ],
    }
    return report


def describe_columns() -> dict[str, str]:
    """Human-readable meaning of the raw column groups."""
    return {
        "TransactionID": "Unique transaction identifier (join key).",
        "isFraud": "Target: 1 if the transaction is fraudulent.",
        "TransactionDT": "Seconds from a reference point; spans 182 days.",
        "TransactionAmt": "Transaction amount in USD.",
        "ProductCD": "Product code for the transaction (W, C, R, H, S).",
        "card1-card6": "Payment card attributes: card1/card2 network & bank codes, "
                       "card4 brand (visa/mastercard/...), card6 debit/credit.",
        "addr1, addr2": "Billing address region and country code.",
        "dist1, dist2": "Distances between buyer, card, merchant and address.",
        "P_emaildomain": "Purchaser email domain.",
        "R_emaildomain": "Recipient email domain.",
        "C1-C14": "Vesta counting features (e.g. how many addresses/cards are tied "
                  "to a payment instrument).",
        "D1-D15": "Timedeltas in days between this transaction and a previous "
                  "event for the same card, e.g. days since the first purchase.",
        "M1-M9": "Match flags (T/F) between the transaction and the cardholder "
                 "identity, e.g. name/address match.",
        "V1-V339": "Vesta's engineered ranking/aggregation features; semantics "
                   "undisclosed by the organisers.",
        "id_01-id_38": "Identity/device fields: network, device, browser, OS, "
                       "screen resolution, and match status.",
        "DeviceType / DeviceInfo": "Device class and device model string.",
    }


def dataset_fingerprint(df: pd.DataFrame) -> dict:
    """Cheap reproducibility fingerprint."""
    return {
        "rows": int(len(df)),
        "frauds": int(df[TARGET].sum()),
        "total_amount": float(df["TransactionAmt"].sum()),
        "dt_range": [int(df["TransactionDT"].min()), int(df["TransactionDT"].max())],
        "expected_dt_range": [TXN_DT_MIN, TXN_DT_MAX],
        "expected_rows": 590_540,
    }
