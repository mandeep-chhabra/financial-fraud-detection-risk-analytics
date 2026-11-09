"""Page 3 - model comparison, curves, calibration and threshold exploration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (
    load_figure_bytes,
    load_json,
    number,
    percent,
    query,
    require,
    threshold_config,
)
from src.config import ERROR_ANALYSIS_JSON, FIGURES_DIR, MODEL_COMPARISON_CSV, TEST_METRICS_JSON


def _show_figure(name: str, caption: str) -> None:
    data = load_figure_bytes(str(Path(FIGURES_DIR) / name))
    if data:
        st.image(data, caption=caption, use_container_width=True)
    else:
        st.info(f"Missing figure {name}. Re-run `python scripts/run_training.py`.")


def render() -> None:
    st.title("Model Performance")
    st.caption("All headline numbers are from the held-out test period (last 15% of days).")
    if not require(["database", "test_metrics", "predictions"]):
        return

    metrics = load_json(str(TEST_METRICS_JSON))
    test = metrics["test_at_operating_threshold"]
    bands = threshold_config()

    cols = st.columns(6)
    cols[0].metric("Precision", f"{test['precision']:.4f}")
    cols[1].metric("Recall", f"{test['recall']:.4f}")
    cols[2].metric("F1", f"{test['f1']:.4f}")
    cols[3].metric("PR-AUC", f"{test['pr_auc']:.4f}")
    cols[4].metric("ROC-AUC", f"{test['roc_auc']:.4f}")
    cols[5].metric("FPR", percent(100 * test["fpr"], 3))

    tabs = st.tabs(["Comparison", "Curves & calibration", "Threshold explorer", "Error analysis"])

    with tabs[0]:
        st.subheader("Experiment leaderboard (validation period)")
        comparison_path = Path(MODEL_COMPARISON_CSV)
        if comparison_path.exists():
            import pandas as pd

            leaderboard = pd.read_csv(comparison_path)
            display = leaderboard[
                [
                    "rank", "model", "train_rows", "precision", "recall", "f1",
                    "pr_auc", "roc_auc", "recall_at_1pct_fpr", "fpr", "fit_seconds",
                ]
            ].rename(
                columns={
                    "model": "Model",
                    "train_rows": "Train rows",
                    "pr_auc": "PR-AUC",
                    "roc_auc": "ROC-AUC",
                    "recall_at_1pct_fpr": "Recall @ 1% FPR",
                    "fpr": "FPR @ 0.5",
                    "fit_seconds": "Fit (s)",
                }
            )
            st.dataframe(
                display.style.format(
                    {
                        "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}",
                        "PR-AUC": "{:.4f}", "ROC-AUC": "{:.4f}",
                        "Recall @ 1% FPR": "{:.4f}", "FPR @ 0.5": "{:.5f}",
                        "Train rows": "{:,.0f}", "Fit (s)": "{:.0f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.info(
                "Selection rule: highest recall at FPR <= 1%, tie-broken by PR-AUC and "
                "then by training-set size. Accuracy and ROC-AUC are reported but never "
                "used to pick the model. Resampling strategies were compared on one "
                "common 120k-row sample so the only thing that changes is the imbalance "
                "treatment."
            )
        _show_figure("model_comparison.png", "Recall at 1% FPR vs PR-AUC by experiment")

        st.subheader("Validation vs test")
        st.dataframe(
            {
                "metric": ["PR-AUC", "ROC-AUC", "Recall @ 1% FPR", "Precision", "Recall", "FPR"],
                "validation": [
                    metrics["validation"]["pr_auc"],
                    metrics["validation"]["roc_auc"],
                    metrics["validation"]["recall_at_1pct_fpr"],
                    metrics["validation"]["at_operating_threshold"]["precision"],
                    metrics["validation"]["at_operating_threshold"]["recall"],
                    metrics["validation"]["at_operating_threshold"]["fpr"],
                ],
                "test": [
                    test["pr_auc"], test["roc_auc"],
                    metrics["test_recall_at_1pct_fpr"]["recall"],
                    test["precision"], test["recall"], test["fpr"],
                ],
            },
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "The two periods have almost identical fraud prevalence, so precision and "
            "recall are directly comparable. The small gap between them is the honest "
            "estimate of how much the validation-tuned threshold overfits the "
            "validation period."
        )

    with tabs[1]:
        left, right = st.columns(2)
        with left:
            _show_figure("pr_curve_test.png", "Precision-recall curve (test)")
        with right:
            _show_figure("roc_curve_test.png", "ROC curve (test)")
        _show_figure("calibration_test.png", "Reliability diagram (test)")
        prevalence = test["positives"] / test["n"]
        st.markdown(
            f"**Brier score** {test['brier']:.6f}. Brier mixes calibration and "
            "discrimination; compare it with the no-skill reference "
            f"prevalence·(1-prevalence) = {prevalence * (1 - prevalence):.6f}. "
            "A PR-AUC baseline of prevalence "
            f"({prevalence:.4f}) corresponds to a model that ranks fraud at random."
        )
        _show_figure("confusion_matrix_test.png", "Confusion matrix at the operating threshold")

    with tabs[2]:
        st.subheader("Adjust the operating threshold")
        sweep = query("threshold_analysis")
        chosen = metrics["operating_threshold"]
        threshold = st.slider(
            "Decision threshold (probability of fraud)",
            min_value=0.005, max_value=0.95,
            value=float(np.clip(chosen, 0.005, 0.95)), step=0.005,
        )
        scenario = query("confusion_at_threshold", {"threshold": threshold}).iloc[0]
        tp, fp, fn, tn = int(scenario["tp"]), int(scenario["fp"]), int(scenario["fn"]), int(scenario["tn"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        c = st.columns(5)
        c[0].metric("Alerts", number(tp + fp))
        c[1].metric("Precision", f"{precision:.4f}")
        c[2].metric("Recall", f"{recall:.4f}")
        c[3].metric("FPR", percent(100 * fpr, 3))
        c[4].metric(
            "Fraud value captured",
            percent(100 * scenario["fraud_value_captured"] / max(scenario["total_fraud_amount"], 1e-9), 2),
        )
        st.caption("Counts come from a live SQL query over all scored validation + test rows.")

        fig = go.Figure()
        for column, label, color in (
            ("precision", "Precision", "#2563EB"),
            ("recall", "Recall", "#DC2626"),
            ("fpr", "FPR", "#F59E0B"),
        ):
            fig.add_scatter(x=sweep["threshold"], y=sweep[column], name=label, line=dict(color=color))
        fig.add_vline(x=chosen, line_dash="dash", line_color="black",
                      annotation_text="validation-optimal", annotation_position="top")
        fig.add_vline(x=threshold, line_dash="dot", line_color="#6B7280",
                      annotation_text="your threshold", annotation_position="bottom")
        fig.update_layout(height=420, xaxis_title="Threshold", yaxis_title="Score",
                          legend=dict(orientation="h"), margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Curve computed on the validation period; the slider re-scores the scored rows live.")
        st.markdown(
            f"**Risk bands** — high: p >= {bands.get('high_threshold', float('nan')):.4f}, "
            f"medium: p >= {bands.get('medium_threshold', float('nan')):.4f}. "
            f"{bands.get('risk_band_rule', '')}"
        )

    with tabs[3]:
        if not Path(ERROR_ANALYSIS_JSON).exists():
            st.info("Run training to generate the error analysis.")
            return
        errors = load_json(str(ERROR_ANALYSIS_JSON))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**False positives**")
            st.write(
                {
                    "count": errors["false_positives"]["count"],
                    "rate per 1,000 legitimate": round(errors["false_positives"]["rate_per_1000_legit"], 2),
                    "value": f"${errors['false_positives']['amount_total']:,.0f}",
                    "by product": errors["false_positives"]["by_product"],
                }
            )
        with col2:
            st.markdown("**False negatives**")
            st.write(
                {
                    "count": errors["false_negatives"]["count"],
                    "share of fraud value missed": percent(
                        100 * errors["false_negatives"]["share_of_fraud_value_missed"]
                    ),
                    "value": f"${errors['false_negatives']['amount_total']:,.0f}",
                    "by product": errors["false_negatives"]["by_product"],
                }
            )

        st.markdown("**Performance by product code**")
        st.dataframe(errors["segments"]["product_cd"], hide_index=True, use_container_width=True)
        st.markdown("**Performance by amount quartile**")
        st.dataframe(errors["segments"]["amount_bucket"], hide_index=True, use_container_width=True)
        if "identity" in errors["segments"]:
            st.markdown("**Performance with and without identity data**")
            st.dataframe(errors["segments"]["identity"], hide_index=True, use_container_width=True)

        sensitivity = errors["threshold_sensitivity"]
        fig = px.line(
            sensitivity, x="threshold", y=["precision", "recall", "fpr"], markers=True,
            labels={"value": "score", "variable": "metric"},
        )
        fig.update_layout(height=340, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.json(
            {
                "mean recall first half of test period": errors["temporal"]["mean_recall_first_half"],
                "mean recall second half of test period": errors["temporal"]["mean_recall_second_half"],
            }
        )
        st.caption(errors["temporal"]["interpretation"])
