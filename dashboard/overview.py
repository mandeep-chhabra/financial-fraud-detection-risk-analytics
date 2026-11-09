"""Page 1 - executive overview."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.common import (
    BAND_COLORS,
    load_json,
    metrics_table,
    money,
    number,
    percent,
    query,
    require,
)
from src.config import TEST_METRICS_JSON


def render() -> None:
    st.title("Financial Fraud Detection & Transaction Risk Analytics")
    st.caption(
        "IEEE-CIS Fraud Detection (Vesta) · 590,540 labelled transactions over 182 days · "
        "temporal train/validation/test split · LightGBM risk scoring under an explicit "
        "review-capacity constraint."
    )
    if not require(["database", "test_metrics"]):
        return

    overview = query("overview").iloc[0]
    scored = query("scored_overview").iloc[0]
    bands = query("risk_bands")
    metrics = load_json(str(TEST_METRICS_JSON))

    high_risk = (
        int(bands.loc[bands["risk_band"] == "High Risk", "transactions"].sum()) if len(bands) else 0
    )

    row1 = st.columns(5)
    with row1[0]:
        st.metric("Transactions analysed", number(overview["total_transactions"]), "labelled dataset")
    with row1[1]:
        st.metric(
            "Fraudulent transactions",
            number(overview["fraud_transactions"]),
            f"{overview['fraud_rate_pct']:.3f}% prevalence",
        )
    with row1[2]:
        st.metric("Legitimate transactions", number(overview["legitimate_transactions"]))
    with row1[3]:
        st.metric("High-risk transactions", number(high_risk), "scored validation + test periods")
    with row1[4]:
        st.metric(
            "Fraud value at risk",
            money(overview["fraud_value"]),
            f"{overview['fraud_value_pct']:.2f}% of total value",
        )

    st.divider()

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Risk bands on scored transactions")
        fig = px.bar(
            bands,
            x="risk_band",
            y="transactions",
            color="risk_band",
            color_discrete_map=BAND_COLORS,
            text="transactions",
            category_orders={"risk_band": ["Low Risk", "Medium Risk", "High Risk"]},
        )
        fig.update_traces(texttemplate="%{text:,}")
        fig.update_layout(showlegend=False, height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Bands come from the validation-tuned cutoffs in `models/risk_thresholds.json`; "
            "they are decision policy, not ground truth."
        )
    with right:
        st.subheader("Held-out model performance")
        test = metrics["test_at_operating_threshold"]
        rows = [
            ("Precision", percent(100 * test["precision"], 2)),
            ("Recall", percent(100 * test["recall"], 2)),
            ("F1", f"{test['f1']:.4f}"),
            ("PR-AUC", f"{test['pr_auc']:.4f}"),
            ("ROC-AUC", f"{test['roc_auc']:.4f}"),
            ("False positive rate", percent(100 * test["fpr"], 3)),
        ]
        st.dataframe(
            {"metric": [r[0] for r in rows], "value (test)": [r[1] for r in rows]},
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Operating threshold {metrics['operating_threshold']:.3f} · "
            f"{number(test['alerts'])} alerts on {number(metrics['splits']['test_rows'])} "
            "held-out transactions."
        )

    st.divider()
    st.subheader("Fraud by product code")
    by_product = query("fraud_by_product")
    left, right = st.columns([1, 1.2])
    with left:
        fig = px.bar(
            by_product,
            x="product_cd",
            y="frauds",
            text="frauds",
            color="fraud_rate_pct",
            color_continuous_scale="Reds",
            labels={"frauds": "fraudulent transactions", "fraud_rate_pct": "fraud rate (%)"},
        )
        fig.update_layout(height=330, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(
            by_product.rename(
                columns={
                    "product_cd": "Product",
                    "transactions": "Transactions",
                    "frauds": "Frauds",
                    "fraud_rate_pct": "Fraud rate (%)",
                    "avg_amount": "Avg amount",
                    "fraud_value": "Fraud value",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Product C (credit / card-not-present) carries the highest fraud rate, W "
            "(wallet) the lowest. Unlike PaySim's TRANSFER/CASH_OUT artefact, every "
            "product here contains fraud, so channel is not a free filter."
        )

    st.divider()
    st.subheader("Scored portfolio")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alerts raised", number(scored["alerts"]), f"{scored['alert_rate_pct']:.3f}% of scored")
    c2.metric("Alert precision", percent(scored["precision_pct"], 2))
    c3.metric("Fraud recall", percent(scored["recall_pct"], 2))
    c4.metric("False positives", number(scored["false_positives"]))
    st.caption(
        "Computed by SQL over every scored validation and test transaction at the "
        "validation-tuned threshold."
    )

    st.subheader("Model comparison (validation period)")
    comparison = metrics_table(
        ["precision", "recall", "f1", "pr_auc", "roc_auc", "recall_at_1pct_fpr"]
    )
    st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
    st.info(metrics["interpretation"])
