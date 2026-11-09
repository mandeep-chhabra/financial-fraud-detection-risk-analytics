"""Schema and quality inspection of the raw IEEE-CIS files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ieee"
OUT = ROOT / "reports" / "ieee_schema_report.json"


def main() -> None:
    txn = pd.read_csv(RAW / "train_transaction.csv")
    ide = pd.read_csv(RAW / "train_identity.csv")
    print(f"transaction: {txn.shape}  identity: {ide.shape}")
    print(f"frauds: {int(txn.isFraud.sum()):,}  rate: {100 * txn.isFraud.mean():.4f}%")
    print(f"TransactionID unique: {txn.TransactionID.is_unique}")
    print(f"TransactionDT min/max: {txn.TransactionDT.min():,} .. {txn.TransactionDT.max():,}")
    days = (txn.TransactionDT.max() - txn.TransactionDT.min()) / 86400
    print(f"time span: {days:.1f} days")

    merged = txn.merge(ide, on="TransactionID", how="left", validate="one_to_one")
    has_identity = merged["id_01"].notna()
    print(
        f"identity coverage: {100 * has_identity.mean():.2f}% of rows; "
        f"fraud rate with identity {100 * merged.loc[has_identity, 'isFraud'].mean():.4f}% "
        f"vs without {100 * merged.loc[~has_identity, 'isFraud'].mean():.4f}%"
    )

    dtypes = txn.dtypes.astype(str).value_counts().to_dict()
    print("dtypes:", dtypes)
    missing = txn.isna().mean()
    print(f"columns >90% missing: {int((missing > 0.9).sum())}")
    print("top 10 missing:\n", (100 * missing).sort_values(ascending=False).head(10).round(1))

    cat_cols = [c for c in txn.columns if txn[c].dtype == object]
    print("object columns (transaction):", cat_cols)
    for c in cat_cols:
        print(f"  {c}: {txn[c].nunique()} unique, {txn[c].isna().mean():.1%} missing")
    ide_cat = [c for c in ide.columns if ide[c].dtype == object]
    print("object columns (identity):")
    for c in ide_cat:
        print(f"  {c}: {ide[c].nunique()} unique, {ide[c].isna().mean():.1%} missing")

    print("ProductCD:\n", txn.ProductCD.value_counts(dropna=False))
    print("card4:\n", txn.card4.value_counts(dropna=False))
    print("card6:\n", txn.card6.value_counts(dropna=False))
    print("P_emaildomain nulls:", int(txn.P_emaildomain.isna().sum()))
    print("TransactionAmt describe:\n", txn.TransactionAmt.describe())
    print("fraud by ProductCD:\n", txn.groupby("ProductCD").isFraud.agg(["count", "sum", "mean"]))

    report = {
        "transaction_shape": list(txn.shape),
        "identity_shape": list(ide.shape),
        "frauds": int(txn.isFraud.sum()),
        "fraud_rate": float(txn.isFraud.mean()),
        "time_span_days": float(days),
        "transaction_dt_min": int(txn.TransactionDT.min()),
        "transaction_dt_max": int(txn.TransactionDT.max()),
        "identity_coverage": float(has_identity.mean()),
        "fraud_rate_with_identity": float(merged.loc[has_identity, "isFraud"].mean()),
        "fraud_rate_without_identity": float(merged.loc[~has_identity, "isFraud"].mean()),
        "object_columns_transaction": cat_cols,
        "object_columns_identity": ide_cat,
        "columns_over_90pct_missing": int((missing > 0.9).sum()),
        "product_cd_counts": txn.ProductCD.value_counts().to_dict(),
        "fraud_by_product_cd": txn.groupby("ProductCD").isFraud.mean().to_dict(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
