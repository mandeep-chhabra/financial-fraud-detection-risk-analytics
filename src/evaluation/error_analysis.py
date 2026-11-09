"""Held-out error analysis: false positives, false negatives, segments, drift.

Every function receives the test split only. Nothing here feeds back into model
selection or threshold choice; this module explains the chosen operating point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics


def _mask_summary(mask: np.ndarray, amounts: np.ndarray, types: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"count": 0, "amount_total": 0.0, "amount_mean": 0.0, "amount_median": 0.0, "by_product": {}}
    amount = amounts[mask]
    by_product = pd.Series(types[mask]).value_counts().to_dict()
    return {
        "count": n,
        "amount_total": float(amount.sum()),
        "amount_mean": float(amount.mean()),
        "amount_median": float(np.median(amount)),
        "by_product": {str(k): int(v) for k, v in by_product.items()},
    }


def run_error_analysis(
    y_true,
    scores,
    amounts,
    types,
    days,
    threshold: float,
    medium_threshold: float,
    identity: np.ndarray | None = None,
    segment_column_name: str = "product_cd",
) -> dict:
    y_true = np.asarray(y_true).astype(np.int8)
    scores = np.asarray(scores, dtype="float64")
    amounts = np.asarray(amounts, dtype="float64")
    types = np.asarray(types).astype(str)
    days = np.asarray(days, dtype="int32")

    pred = scores >= threshold
    tp = pred & (y_true == 1)
    fp = pred & (y_true == 0)
    fn = (~pred) & (y_true == 1)
    tn = (~pred) & (y_true == 0)

    fp_summary = _mask_summary(fp, amounts, types)
    fn_summary = _mask_summary(fn, amounts, types)
    fp_summary["rate_per_1000_legit"] = float(1000 * fp.sum() / max((y_true == 0).sum(), 1))
    fn_summary["share_of_fraud_value_missed"] = float(
        amounts[fn].sum() / max(amounts[y_true == 1].sum(), 1)
    )

    frame = pd.DataFrame(
        {"y": y_true, "score": scores, "amount": amounts, "product_cd": types, "day": days}
    )
    frame["amount_bucket"] = pd.qcut(
        frame["amount"], q=4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"], duplicates="drop"
    ).astype(str)
    if identity is not None:
        frame["identity"] = np.where(np.asarray(identity).astype(bool), "identity present", "identity absent")

    segments = {}
    segment_columns = ["product_cd", "amount_bucket"]
    if "identity" in frame.columns:
        segment_columns.append("identity")

    for column in segment_columns:
        rows = []
        for value, part in frame.groupby(column, observed=True):
            metrics = compute_metrics(part["y"], part["score"], threshold, part["amount"])
            rows.append(
                {
                    "segment": str(value),
                    "transactions": int(len(part)),
                    "frauds": int(part["y"].sum()),
                    "fraud_rate": float(part["y"].mean()),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "fpr": metrics["fpr"],
                    "pr_auc": metrics["pr_auc"],
                    "fraud_value_captured_pct": metrics["fraud_value_captured_pct"],
                }
            )
        segments[column] = rows

    # ------------------------------------------------------- temporal stability
    daily = []
    for day, part in frame.groupby("day"):
        if len(part) == 0:
            continue
        metrics = compute_metrics(part["y"], part["score"], threshold, part["amount"])
        daily.append(
            {
                "day": int(day),
                "transactions": int(len(part)),
                "frauds": int(part["y"].sum()),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fpr": metrics["fpr"],
                "fraud_value_captured_pct": metrics["fraud_value_captured_pct"],
            }
        )

    half = max(len(daily) // 2, 1)
    recall_first = float(np.mean([d["recall"] for d in daily[:half]])) if daily else float("nan")
    recall_last = float(np.mean([d["recall"] for d in daily[-half:]])) if daily else float("nan")

    # --------------------------------------------------- threshold sensitivity
    sensitivity = []
    for factor in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        t = float(np.clip(threshold * factor, 0.001, 0.999))
        metrics = compute_metrics(y_true, scores, t, amounts)
        sensitivity.append(
            {
                "threshold": t,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fpr": metrics["fpr"],
                "alerts": metrics["alerts"],
                "fraud_value_captured_pct": metrics["fraud_value_captured_pct"],
            }
        )

    examples = frame.assign(predicted=pred)[["product_cd", "amount", "score", "y", "day"]]
    top_fp = examples[fp].sort_values("score", ascending=False).head(10).to_dict(orient="records")
    top_fn = examples[fn].sort_values("score", ascending=False).head(10).to_dict(orient="records")

    return {
        "threshold": float(threshold),
        "medium_threshold": float(medium_threshold),
        "segment_column": segment_column_name,
        "confusion": {
            "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "tn": int(tn.sum())
        },
        "false_positives": fp_summary,
        "false_negatives": fn_summary,
        "alert_composition": {
            "precision": float(tp.sum() / max(pred.sum(), 1)),
            "alerts": int(pred.sum()),
        },
        "segments": segments,
        "temporal": {
            "daily": daily,
            "mean_recall_first_half": recall_first,
            "mean_recall_second_half": recall_last,
            "interpretation": (
                "Recall is averaged over calendar days of the test period. A large "
                "drop between halves indicates temporal degradation; a moderate one "
                "is expected because the operating threshold is fixed while the fraud "
                "mix drifts."
            ),
        },
        "threshold_sensitivity": sensitivity,
        "examples": {"top_false_positives": top_fp, "top_false_negatives": top_fn},
        "notes": [
            "False-positive cost is an assumption ($4 per review in the operating "
            "policy), not a measured bank figure.",
            "Missed fraud is valued at the full transaction amount, an upper bound on "
            "real loss because chargebacks are often only partial.",
            "The dataset contains no label timestamp: a transaction labelled legitimate "
            "may still be under dispute.",
        ],
    }
