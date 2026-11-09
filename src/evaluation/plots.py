"""Matplotlib figures for reports and the dashboard.

Every function only reads pre-computed results, so plots can be regenerated
without retraining.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {"legit": "#3B82F6", "fraud": "#DC2626", "accent": "#F59E0B", "muted": "#6B7280"}


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(report: dict, path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [report["legitimate"], report["frauds"]]
    labels = ["Legitimate", "Fraud"]
    bars = ax.bar(labels, values, color=[PALETTE["legit"], PALETTE["fraud"]])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Class distribution (%.3f%% fraud)" % (100 * report["fraud_rate"]))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    return _save(fig, path)


def plot_amount_distribution(amounts_legit, amounts_fraud, path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(0, np.log10(max(np.max(amounts_legit), np.max(amounts_fraud))), 50)
    ax.hist(np.clip(amounts_legit, 1, None), bins=bins, alpha=0.6, label="Legitimate", color=PALETTE["legit"], density=True)
    ax.hist(np.clip(amounts_fraud, 1, None), bins=bins, alpha=0.6, label="Fraud", color=PALETTE["fraud"], density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount distribution by class")
    ax.legend()
    return _save(fig, path)


def plot_fraud_rate_by_type(by_type: pd.DataFrame, path, column: str = "product_cd") -> Path:
    """Fraud rate per category, annotated with the absolute fraud count."""
    data = by_type.copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [PALETTE["fraud"] if r > 0 else PALETTE["muted"] for r in data["fraud_rate"]]
    bars = ax.bar(data[column].astype(str), 100 * data["fraud_rate"], color=colors)
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud prevalence by product code")
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{int(row['frauds']):,}\nof {int(row['transactions']):,}",
            ha="center", va="bottom", fontsize=8,
        )
    return _save(fig, path)


def plot_missing_values(report: dict, path) -> Path:
    """Top-25 most-missing columns. IEEE-CIS has hundreds of sparse features."""
    missing = pd.Series(report.get("missing_top", {})) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    if missing.empty or missing.sum() == 0:
        ax.text(0.5, 0.5, "No missing values in the reported columns", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        missing.sort_values().plot.barh(ax=ax, color=PALETTE["accent"])
        ax.set_xlabel("Missing (% of rows)")
    ax.set_title("Top 25 columns by missing rate")
    return _save(fig, path)


def plot_fraud_by_card(card_frame: pd.DataFrame, path) -> Path:
    """Fraud rate by card brand and card type (debit vs credit)."""
    data = card_frame.copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['group']}\n({int(r['transactions']):,})" for _, r in data.iterrows()]
    bars = ax.bar(labels, 100 * data["fraud_rate"], color=PALETTE["legit"])
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud prevalence by card brand and card type")
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(row['frauds']):,}", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    return _save(fig, path)


def plot_correlation(corr: pd.DataFrame, path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Correlation between engineered features and the fraud label")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_temporal(daily: pd.DataFrame, path) -> Path:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(daily["day"], daily["transactions"], color=PALETTE["muted"], label="Transactions")
    ax1.set_xlabel("Simulated day")
    ax1.set_ylabel("Transactions", color=PALETTE["muted"])
    ax2 = ax1.twinx()
    ax2.bar(daily["day"], daily["frauds"], color=PALETTE["fraud"], alpha=0.6, label="Frauds")
    ax2.set_ylabel("Frauds", color=PALETTE["fraud"])
    ax1.set_title("Transaction volume and fraud count over time")
    return _save(fig, path)


def plot_confusion(metrics: dict, path) -> Path:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i, j]}\n{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Predicted legit", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax.set_title(f"Confusion matrix @ threshold {metrics['threshold']:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_pr_curve(points: dict, path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(points["recall"], points["precision"], color=PALETTE["fraud"], lw=2, label="Model")
    ax.axhline(points["baseline"], ls="--", color=PALETTE["muted"], label=f"Prevalence = {points['baseline']:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_roc_curve(points: dict, path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(points["fpr"], points["tpr"], color=PALETTE["legit"], lw=2, label="Model")
    ax.plot([0, 1], [0, 1], ls="--", color=PALETTE["muted"], label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_threshold_curves(sweep: pd.DataFrame, chosen: float, path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision", color=PALETTE["legit"])
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall", color=PALETTE["fraud"])
    ax.plot(sweep["threshold"], sweep["fpr"], label="FPR", color=PALETTE["accent"])
    ax.axvline(chosen, ls="--", color="black", label=f"Chosen = {chosen:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / recall / FPR versus threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_calibration(report: dict, path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color=PALETTE["muted"], label="Perfect calibration")
    ax.plot(report["prob_pred"], report["prob_true"], marker="o", color=PALETTE["fraud"], label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title(f"Reliability diagram (Brier = {report['brier_score']:.5f})")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_model_comparison(leaderboard: pd.DataFrame, path) -> Path:
    data = leaderboard.sort_values("recall_at_1pct_fpr")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(data))
    ax.barh(y - 0.2, data["recall_at_1pct_fpr"], height=0.4, color=PALETTE["fraud"], label="Recall @ 1% FPR")
    ax.barh(y + 0.2, data["pr_auc"], height=0.4, color=PALETTE["legit"], label="PR-AUC")
    ax.set_yticks(y, data["model"], fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score (validation)")
    ax.set_title("Model comparison: rare-event metrics")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path, top_n: int = 15) -> Path:
    data = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(data["feature"], data["mean_abs_shap"], color=PALETTE["fraud"])
    ax.set_xlabel("mean(|SHAP value|) - impact on fraud log-odds")
    ax.set_title("Global SHAP feature importance")
    return _save(fig, path)
