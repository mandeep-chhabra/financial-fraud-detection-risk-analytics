"""Evaluation metrics for rare-event fraud classification.

Everything in this module is deterministic and independent of the model library,
so the training script, the dashboard and the tests share one definition of every
metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_div(num, den):
    den = np.asarray(den, dtype="float64")
    num = np.asarray(num, dtype="float64")
    out = np.divide(num, den, out=np.zeros_like(num, dtype="float64"), where=den != 0)
    return out


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    amounts=None,
    cost_per_investigation: float | None = None,
) -> dict:
    """Full metric set at a single operating threshold.

    ``amounts`` enables the value-based metrics: how much of the total fraudulent
    value is captured, and the net benefit of alerting.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score, dtype="float64")
    y_pred = (y_score >= threshold).astype(np.int8)

    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    positives = tp + fn
    negatives = tn + fp

    out = {
        "threshold": float(threshold),
        **counts,
        "n": int(len(y_true)),
        "positives": int(positives),
        "negatives": int(negatives),
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(positives, 1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": float(fp / max(negatives, 1)),
        "fnr": float(fn / max(positives, 1)),
        "specificity": float(tn / max(negatives, 1)),
        "alerts": int(tp + fp),
        "alert_rate": float((tp + fp) / max(len(y_true), 1)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if positives and negatives else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if positives else float("nan"),
        "brier": float(brier_score_loss(y_true, y_score)) if len(y_true) else float("nan"),
    }

    if amounts is not None:
        amounts = np.asarray(amounts, dtype="float64")
        fraud_amount = amounts[y_true == 1].sum()
        captured = amounts[(y_true == 1) & (y_pred == 1)].sum()
        out["fraud_value_total"] = float(fraud_amount)
        out["fraud_value_captured"] = float(captured)
        out["fraud_value_captured_pct"] = float(captured / fraud_amount) if fraud_amount > 0 else 0.0
        out["legitimate_value_blocked"] = float(amounts[(y_true == 0) & (y_pred == 1)].sum())
        if cost_per_investigation is not None:
            out["investigation_cost"] = float((tp + fp) * cost_per_investigation)
            out["net_benefit"] = float(captured - (tp + fp) * cost_per_investigation)

    return out


def threshold_sweep(
    y_true,
    y_score,
    thresholds=None,
    amounts=None,
    cost_per_investigation: float | None = None,
) -> pd.DataFrame:
    """Vectorised precision/recall/FPR/value table across thresholds.

    Sorting the scores once and using cumulative sums makes this O(n log n)
    regardless of how many thresholds are requested, which matters on the 1M-row
    test split.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score, dtype="float64")
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 4)
    thresholds = np.asarray(thresholds, dtype="float64")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    cumulative_tp = np.concatenate([[0], np.cumsum(y_sorted)])
    cumulative_fp = np.concatenate([[0], np.cumsum(1 - y_sorted)])
    if amounts is not None:
        amt_sorted = np.asarray(amounts, dtype="float64")[order]
        cumulative_value = np.concatenate([[0], np.cumsum(amt_sorted * y_sorted)])

    # number of scores >= threshold
    idx = np.searchsorted(-s_sorted, -thresholds, side="right")
    tp = cumulative_tp[idx]
    fp = cumulative_fp[idx]
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    fn = positives - tp
    tn = negatives - fp

    frame = pd.DataFrame(
        {
            "threshold": thresholds,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "alerts": tp + fp,
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "fpr": _safe_div(fp, fp + tn),
            "fnr": _safe_div(fn, fn + tp),
            "alert_rate": _safe_div(tp + fp, len(y_true)),
        }
    )
    frame["f1"] = _safe_div(2 * frame["precision"] * frame["recall"], frame["precision"] + frame["recall"])

    if amounts is not None:
        captured = cumulative_value[idx]
        frame["fraud_value_captured"] = captured
        frame["fraud_value_captured_pct"] = _safe_div(
            captured, float(np.asarray(amounts, dtype="float64")[y_true == 1].sum())
        )
        if cost_per_investigation is not None:
            frame["investigation_cost"] = (tp + fp) * cost_per_investigation
            frame["net_benefit"] = captured - frame["investigation_cost"]
    return frame


def recall_at_fpr(y_true, y_score, max_fpr: float = 0.01) -> dict:
    """Recall achievable while alerting at most ``max_fpr`` of legitimate rows.

    Uses the ROC curve and linear interpolation at the FPR budget, then returns
    the corresponding threshold so the number is reproducible in the dashboard.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score, dtype="float64")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.searchsorted(fpr, max_fpr, side="right") - 1)
    idx = max(idx, 0)
    return {
        "max_fpr": float(max_fpr),
        "fpr": float(fpr[idx]),
        "recall": float(tpr[idx]),
        "threshold": float(thresholds[idx]),
    }


def select_operating_threshold(
    sweep: pd.DataFrame,
    cost_per_investigation: float = 10.0,
    max_fpr: float = 0.01,
    min_recall: float = 0.0,
    objective: str = "f1",
) -> dict:
    """Choose an operating threshold inside the manual-review budget.

    Objectives
    ----------
    ``f1`` (default)
        Maximise the harmonic mean of precision and recall subject to
        ``FPR <= max_fpr``. F1 is used because PaySim amounts are synthetic: a
        dollar objective (``net_benefit``) makes the review cost negligible next
        to the simulated fraud value and degenerates to "alert everything inside
        the budget", which is not a useful operating policy.
    ``net_benefit``
        Maximise captured fraudulent value minus investigation cost.

    If no threshold satisfies the constraints the constraint is relaxed and that
    is recorded in the return value.
    """
    feasible = sweep[(sweep["fpr"] <= max_fpr) & (sweep["recall"] >= min_recall)].copy()
    relaxed = False
    if feasible.empty:
        feasible = sweep.copy()
        relaxed = True

    if objective == "net_benefit" and "net_benefit" in feasible.columns:
        best = feasible.sort_values(
            ["net_benefit", "recall"], ascending=[False, False]
        ).iloc[0]
        objective_text = (
            "maximise captured fraudulent value minus investigation cost, "
            f"subject to FPR <= {max_fpr:.2%} on the validation period"
        )
    else:
        best = feasible.sort_values(["f1", "recall"], ascending=False).iloc[0]
        objective_text = (
            "maximise F1 (fraud caught vs alert precision) subject to "
            f"FPR <= {max_fpr:.2%} manual-review capacity on the validation period"
        )

    return {
        "threshold": float(best["threshold"]),
        "objective": objective_text,
        "objective_metric": objective,
        "cost_per_investigation": float(cost_per_investigation),
        "max_fpr": float(max_fpr),
        "constraint_relaxed": bool(relaxed),
        "validation": {
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "fpr": float(best["fpr"]),
            "alerts": int(best["alerts"]),
            **(
                {
                    "net_benefit": float(best["net_benefit"]),
                    "fraud_value_captured": float(best["fraud_value_captured"]),
                    "fraud_value_captured_pct": float(best["fraud_value_captured_pct"]),
                }
                if "net_benefit" in feasible.columns
                else {}
            ),
        },
    }


def choose_risk_bands(
    sweep: pd.DataFrame,
    high_threshold: float,
    medium_fpr: float = 0.05,
    medium_alert_rate: float | None = 0.05,
) -> dict:
    """Derive the medium-risk cutoff from a documented capacity budget.

    Medium risk is the highest score cutoff whose cumulative alert volume stays
    inside the step-up-verification capacity (5% of transactions). Capacity-based
    bands are used because the score distribution is strongly bimodal: an
    FPR-based cutoff collapsed to a 4-transaction band on validation. If the
    sweep is too coarse for the capacity target, the function falls back to the
    FPR budget and records which basis was used.
    """
    basis = "alert_rate"
    candidates = pd.DataFrame()
    if medium_alert_rate is not None and "alert_rate" in sweep.columns:
        candidates = sweep[
            (sweep["alert_rate"] <= medium_alert_rate)
            & (sweep["threshold"] < high_threshold)
        ]
    if candidates.empty:
        basis = "fpr"
        candidates = sweep[(sweep["fpr"] <= medium_fpr) & (sweep["threshold"] < high_threshold)]

    # The lowest cutoff that still satisfies the capacity budget is the widest
    # band the budget allows. Using the highest one would always land immediately
    # below the high-risk threshold and produce an empty medium band.
    medium = (
        float(candidates["threshold"].min())
        if not candidates.empty
        else float(high_threshold * 0.5)
    )
    return {
        "high_threshold": float(high_threshold),
        "medium_threshold": float(medium),
        "medium_band_basis": basis,
        "medium_alert_rate_budget": (
            float(medium_alert_rate) if medium_alert_rate is not None else None
        ),
        "medium_fpr_budget": float(medium_fpr),
        "rule": (
            "High risk: p >= high_threshold (validation-optimal threshold under a "
            "1% FPR review budget). Medium risk: p >= medium_threshold, the lowest "
            "score cutoff whose cumulative alert volume still fits the "
            f"{medium_alert_rate:.0%} step-up-verification capacity. Low risk: "
            "everything below. Bands are policy, not ground truth."
        ),
    }


def assign_risk_band(probabilities, high_threshold: float, medium_threshold: float) -> np.ndarray:
    """Map probabilities to the three documented risk bands."""
    p = np.asarray(probabilities, dtype="float64")
    bands = np.full(p.shape, "Low Risk", dtype=object)
    bands[p >= medium_threshold] = "Medium Risk"
    bands[p >= high_threshold] = "High Risk"
    return bands


def risk_band_summary(probabilities, high_threshold: float, medium_threshold: float) -> pd.DataFrame:
    p = np.asarray(probabilities, dtype="float64")
    bands = assign_risk_band(p, high_threshold, medium_threshold)
    frame = pd.DataFrame({"risk_band": bands, "probability": p})
    summary = frame.groupby("risk_band").agg(
        transactions=("probability", "size"), mean_score=("probability", "mean")
    )
    return summary.reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0).reset_index()


def calibration_report(y_true, y_score, n_bins: int = 10) -> dict:
    """Reliability curve + Brier score.

    ``prob_true``/``prob_pred`` are the coordinates of the reliability diagram;
    a well-calibrated model lies on the diagonal.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score, dtype="float64")
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    return {
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "n_bins": int(n_bins),
        "bin_strategy": "quantile",
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "mean_predicted": float(y_score.mean()),
        "observed_rate": float(y_true.mean()),
    }


def calibrate_and_report(y_val, val_score, y_test, test_score, n_bins: int = 10) -> dict:
    """Fit an isotonic calibrator on validation and compare it on test.

    The raw model score is used for ranking and thresholding (isotonic steps can
    create ties and slightly hurt PR-AUC), but the reliability of the raw
    probabilities is reported honestly here, together with the Brier score of the
    calibrated variant.
    """
    from sklearn.isotonic import IsotonicRegression

    y_val = np.asarray(y_val).astype(np.int8)
    y_test = np.asarray(y_test).astype(np.int8)
    val_score = np.asarray(val_score, dtype="float64")
    test_score = np.asarray(test_score, dtype="float64")

    raw_val = calibration_report(y_val, val_score, n_bins)
    raw_test = calibration_report(y_test, test_score, n_bins)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_score, y_val)
    calibrated_test = iso.predict(test_score)
    calibrated = calibration_report(y_test, calibrated_test, n_bins)

    improvement = raw_test["brier_score"] - calibrated["brier_score"]
    return {
        "validation_raw": raw_val,
        "test": raw_test,
        "test_isotonic": calibrated,
        "brier_improvement_isotonic": float(improvement),
        "decision": (
            "Raw scores are kept for ranking and thresholding. The isotonic "
            "mapping is reported as a diagnostic only; it is not used to "
            "recompute the operating threshold, so all reported test metrics "
            "refer to the raw model probability."
        ),
        "note": (
            "Brier score mixes calibration and discrimination; compare it with "
            "the no-skill reference equal to prevalence*(1-prevalence)."
        ),
    }


def pr_curve_points(y_true, y_score, max_points: int = 500) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # Downsample for a light JSON payload while keeping curve shape.
    if len(thresholds) > max_points:
        idx = np.linspace(0, len(thresholds) - 1, max_points).astype(int)
        precision, recall = precision[idx], recall[idx]
        thresholds = thresholds[idx]
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "threshold": thresholds.tolist(),
        "baseline": float(np.mean(y_true)),
    }


def roc_curve_points(y_true, y_score, max_points: int = 500) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.linspace(0, len(thresholds) - 1, min(max_points, len(thresholds))).astype(int)
    return {
        "fpr": fpr[idx].tolist(),
        "tpr": tpr[idx].tolist(),
        "threshold": thresholds[idx].tolist(),
    }
