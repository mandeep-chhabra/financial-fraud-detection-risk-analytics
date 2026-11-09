"""Exploratory data analysis: figures plus a JSON of every number quoted in the docs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    EDA_SUMMARY_JSON,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    PROCESSED_PARQUET,
    QUALITY_REPORT,
    VAL_END_DT,
)
from src.evaluation.plots import (  # noqa: E402
    plot_amount_distribution,
    plot_class_distribution,
    plot_correlation,
    plot_fraud_by_card,
    plot_fraud_rate_by_type,
    plot_missing_values,
    plot_temporal,
)
from src.utils import ensure_dir, load_json, save_json  # noqa: E402


def run_eda(path: Path = PROCESSED_PARQUET) -> dict:
    ensure_dir(FIGURES_DIR)
    report = load_json(QUALITY_REPORT)

    amount = pd.read_parquet(path, columns=["amount", "isFraud"])
    legit_amounts = amount.loc[amount["isFraud"] == 0, "amount"].to_numpy()
    fraud_amounts = amount.loc[amount["isFraud"] == 1, "amount"].to_numpy()
    del amount

    plot_class_distribution(report, FIGURES_DIR / "class_distribution.png")
    plot_amount_distribution(legit_amounts, fraud_amounts, FIGURES_DIR / "amount_distribution.png")
    plot_missing_values(report, FIGURES_DIR / "missing_values.png")

    by_product = pd.DataFrame(report["by_product"])
    plot_fraud_rate_by_type(by_product, FIGURES_DIR / "fraud_rate_by_product.png")

    card = (
        pd.read_parquet(path, columns=["card4", "card6", "isFraud"])
        .assign(
            group=lambda d: d["card4"].fillna("unknown")
            + " / "
            + d["card6"].fillna("unknown")
        )
        .groupby("group")
        .agg(transactions=("isFraud", "size"), frauds=("isFraud", "sum"))
        .assign(fraud_rate=lambda d: d["frauds"] / d["transactions"])
        .query("transactions >= 2000")
        .sort_values("fraud_rate", ascending=False)
        .reset_index()
    )
    plot_fraud_by_card(card, FIGURES_DIR / "fraud_by_card.png")

    temporal = (
        pd.read_parquet(path, columns=["day_index", "isFraud"])
        .groupby("day_index")
        .agg(transactions=("isFraud", "size"), frauds=("isFraud", "sum"))
        .reset_index()
        .rename(columns={"day_index": "day"})
    )
    plot_temporal(temporal, FIGURES_DIR / "temporal_fraud.png")

    # ------------------------------------------------ univariate ranking power
    test = pd.read_parquet(
        path,
        columns=list(dict.fromkeys(FEATURE_COLUMNS + ["isFraud"])),
        filters=[("transaction_dt", ">", VAL_END_DT)],
    )
    from sklearn.metrics import roc_auc_score

    y_test = test["isFraud"].to_numpy()
    univariate = {}
    for column in FEATURE_COLUMNS:
        values = test[column].to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        # Skip columns that are constant or almost entirely missing; AUC is
        # undefined for them and numpy would warn about a zero-degree slice.
        if finite.size < 100 or np.std(finite) == 0:
            continue
        filled = np.nan_to_num(values, nan=np.median(finite))
        univariate[column] = float(roc_auc_score(y_test, filled))
    univariate_top = dict(
        sorted(univariate.items(), key=lambda item: abs(item[1] - 0.5), reverse=True)[:10]
    )

    # Correlation matrix of the strongest features (readable subset only).
    top_features = list(univariate_top)[:12]
    sample = test[top_features + ["isFraud"]].sample(
        n=min(200_000, len(test)), random_state=42
    )
    corr = sample.corr(numeric_only=True)[["isFraud"]].sort_values("isFraud", ascending=False)
    plot_correlation(corr, FIGURES_DIR / "correlation_with_fraud.png")

    by_type = by_product.set_index("product_cd")
    summary = {
        "transactions": report["rows"],
        "frauds": report["frauds"],
        "fraud_rate_pct": 100 * report["fraud_rate"],
        "imbalance_ratio": report["imbalance_ratio"],
        "fraud_value_share_pct": 100 * report["fraud_amount_share"],
        "days_covered": report["days_covered"],
        "legit_amount_mean": float(np.mean(legit_amounts)),
        "legit_amount_median": float(np.median(legit_amounts)),
        "fraud_amount_mean": float(np.mean(fraud_amounts)),
        "fraud_amount_median": float(np.median(fraud_amounts)),
        "fraud_amount_multiple_of_median": float(
            np.median(fraud_amounts) / max(np.median(legit_amounts), 1e-9)
        ),
        "by_product": report["by_product"],
        "by_card4": report["by_card4"],
        "by_card6": report["by_card6"],
        "identity_coverage_pct": 100 * report["identity_coverage"],
        "fraud_rate_with_identity_pct": 100 * report["fraud_rate_with_identity"],
        "fraud_rate_without_identity_pct": 100 * report["fraud_rate_without_identity"],
        "unique_cards": report["unique_cards"],
        "unique_addresses": report["unique_addresses"],
        "missing_buckets": report["missing_buckets"],
        "univariate_roc_auc_test": univariate_top,
        "univariate_note": (
            "ROC-AUC of each single feature on the test period, with NaN replaced by "
            "the median. Values below 0.5 mean the feature is inversely oriented; "
            "distance from 0.5 is the signal. Univariate power is not additive and "
            "says nothing about causal importance."
        ),
        "highest_fraud_product": by_type["fraud_rate"].idxmax(),
        "highest_fraud_product_rate_pct": 100 * float(by_type["fraud_rate"].max()),
        "lowest_fraud_product": by_type["fraud_rate"].idxmin(),
        "lowest_fraud_product_rate_pct": 100 * float(by_type["fraud_rate"].min()),
        "peak_fraud_day": int(temporal.loc[temporal["frauds"].idxmax(), "day"]),
        "max_frauds_in_a_day": int(temporal["frauds"].max()),
        "daily_fraud_rate_pct_min": float(
            100 * (temporal["frauds"] / temporal["transactions"]).min()
        ),
        "daily_fraud_rate_pct_max": float(
            100 * (temporal["frauds"] / temporal["transactions"]).max()
        ),
        "correlations": {
            k: float(v)
            for k, v in corr["isFraud"].drop("isFraud").head(8).to_dict().items()
        },
        "insights": [
            f"Fraud prevalence is {100 * report['fraud_rate']:.3f}% "
            f"({report['frauds']:,} of {report['rows']:,}), a 1:{report['imbalance_ratio']:.0f} imbalance.",
            f"Product code C carries the highest fraud rate "
            f"({100 * float(by_type.loc['C', 'fraud_rate']):.2f}%), W the lowest "
            f"({100 * float(by_type.loc['W', 'fraud_rate']):.2f}%).",
            f"Fraudulent transactions are larger on average "
            f"(${np.mean(fraud_amounts):,.0f} vs ${np.mean(legit_amounts):,.0f}).",
            f"Identity data exists for {100 * report['identity_coverage']:.1f}% of rows; "
            f"those rows are fraudulent {100 * report['fraud_rate_with_identity']:.2f}% "
            f"of the time versus {100 * report['fraud_rate_without_identity']:.2f}% "
            "without identity data.",
            "The fraud rate is stable across the 182 days "
            f"({100 * (temporal['frauds'] / temporal['transactions']).min():.2f}%-"
            f"{100 * (temporal['frauds'] / temporal['transactions']).max():.2f}% per day), "
            "so a temporal split does not create a prevalence shock.",
        ],
        "figures": [
            "class_distribution.png",
            "amount_distribution.png",
            "fraud_rate_by_product.png",
            "fraud_by_card.png",
            "missing_values.png",
            "temporal_fraud.png",
            "correlation_with_fraud.png",
        ],
    }
    save_json(summary, EDA_SUMMARY_JSON)
    return summary


if __name__ == "__main__":
    run_eda()
