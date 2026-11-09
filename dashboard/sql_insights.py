"""Page 6 - SQL analytics: every table and chart is produced by a SQL query."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import query, require
from src.database.queries import QUERIES


def _chart(name: str, frame):
    if frame.empty:
        return
    if name == "fraud_by_product":
        fig = px.bar(frame, x="product_cd", y="frauds", color="fraud_rate_pct",
                     color_continuous_scale="Reds", labels={"frauds": "frauds"})
    elif name == "fraud_by_card":
        fig = px.bar(frame, x="card_brand", y="fraud_rate_pct", color="card_type",
                     barmode="group", labels={"fraud_rate_pct": "fraud rate (%)"})
    elif name == "identity_coverage":
        fig = px.pie(frame, names="identity_status", values="transactions", hole=0.45)
    elif name == "fraud_by_amount_band":
        fig = px.bar(frame, x="amount_band", y="fraud_rate_pct",
                     labels={"amount_band": "amount band", "fraud_rate_pct": "fraud rate (%)"})
    elif name == "risk_bands":
        fig = px.bar(frame, x="risk_band", y="transactions", color="risk_band",
                     category_orders={"risk_band": ["Low Risk", "Medium Risk", "High Risk"]})
        fig.update_layout(showlegend=False)
    elif name == "risk_band_fraud_value":
        fig = px.bar(frame, x="risk_band", y="share_of_fraud_value_pct", color="risk_band",
                     labels={"share_of_fraud_value_pct": "% of fraud value"})
        fig.update_layout(showlegend=False)
    elif name == "daily_trends":
        fig = go.Figure()
        fig.add_bar(x=frame["day"], y=frame["frauds"], name="Frauds", marker_color="#DC2626")
        fig.add_scatter(x=frame["day"], y=frame["transactions"], name="Transactions",
                        yaxis="y2", line=dict(color="#6B7280"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
    elif name == "threshold_analysis":
        fig = go.Figure()
        for column, color in (("precision", "#2563EB"), ("recall", "#DC2626"), ("fpr", "#F59E0B")):
            fig.add_scatter(x=frame["threshold"], y=frame[column], name=column, line=dict(color=color))
    elif name == "hourly_fraud":
        fig = px.line(frame, x="hour_of_day", y="fraud_rate_pct", markers=True)
    elif name == "error_breakdown":
        fig = px.pie(frame, names="outcome", values="transactions", hole=0.45)
    elif name == "errors_by_product":
        fig = px.bar(frame, x="product_cd",
                     y=["false_positives", "false_negatives", "true_positives"], barmode="group")
    elif name == "split_summary":
        fig = px.bar(frame, x="split", y="transactions", color="fraud_rate_pct",
                     labels={"fraud_rate_pct": "fraud rate (%)"})
    elif name == "card_concentration":
        fig = px.bar(frame, x="card1", y="frauds", color="fraud_rate_pct",
                     labels={"card1": "card1", "frauds": "frauds"})
    else:
        return
    fig.update_layout(height=380, margin=dict(t=20, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    st.title("SQL Insights")
    st.caption(
        "Every number on this page is produced by a parameterised SQL query executed "
        "against `data/fraud.db` (SQLite, 590,540 transaction rows)."
    )
    if not require(["database"]):
        return

    names = [
        "overview", "scored_overview", "fraud_by_product", "fraud_by_card",
        "identity_coverage", "fraud_by_amount_band", "risk_bands", "risk_band_fraud_value",
        "daily_trends", "hourly_fraud", "amount_statistics", "error_breakdown",
        "errors_by_product", "split_summary", "card_concentration", "model_metrics",
        "model_metadata", "threshold_analysis", "top_risk_transactions",
    ]
    groups = {
        "Portfolio": ["overview", "scored_overview"],
        "Segments": ["fraud_by_product", "fraud_by_card", "identity_coverage", "fraud_by_amount_band", "card_concentration"],
        "Risk": ["risk_bands", "risk_band_fraud_value", "top_risk_transactions", "confusion_at_threshold"],
        "Time": ["daily_trends", "hourly_fraud"],
        "Model": ["error_breakdown", "errors_by_product", "split_summary", "model_metrics", "model_metadata", "threshold_analysis"],
    }
    labels = {
        "overview": "Portfolio overview",
        "scored_overview": "Scored portfolio & alert performance",
        "fraud_by_product": "Fraud by product code",
        "fraud_by_card": "Fraud by card brand and type",
        "identity_coverage": "Identity-data coverage vs fraud",
        "fraud_by_amount_band": "Fraud rate by amount band",
        "card_concentration": "Highest-fraud cards (200+ transactions)",
        "risk_bands": "Risk band distribution",
        "risk_band_fraud_value": "Fraud value by risk band",
        "confusion_at_threshold": "Confusion counts at a chosen threshold",
        "daily_trends": "Daily transaction and fraud trend",
        "hourly_fraud": "Fraud rate by hour of day",
        "amount_statistics": "Amount statistics by class",
        "error_breakdown": "Prediction outcomes (TP/FP/FN/TN)",
        "errors_by_product": "Errors by product code",
        "split_summary": "Temporal split summary",
        "model_metrics": "Stored model metrics",
        "model_metadata": "Model metadata",
        "threshold_analysis": "Threshold sweep",
        "top_risk_transactions": "Top 25 highest-risk transactions",
    }

    group = st.selectbox("Analysis group", list(groups))
    choice = st.selectbox(
        "Choose an analysis", groups[group], format_func=lambda n: labels.get(n, n)
    )

    params: dict = {}
    if choice == "top_risk_transactions":
        params["limit"] = 25
    elif choice == "confusion_at_threshold":
        params["threshold"] = st.slider("Threshold", 0.01, 0.95, 0.5, 0.01)

    frame = query(choice, params or None)

    _chart(choice, frame)
    st.dataframe(frame, hide_index=True, use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.download_button(
            "Download result as CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{choice}.csv",
            mime="text/csv",
        )
    with right:
        st.caption(f"{len(frame):,} result rows")

    with st.expander("View the SQL query", expanded=False):
        st.code(QUERIES[choice], language="sql")

    st.divider()
    st.markdown("### Why SQL is part of the system")
    st.markdown(
        "- The dashboard does not recompute aggregates in pandas; it queries the same "
        "database the analytics layer uses.\n"
        "- `transactions` holds all 590,540 labelled rows with a `split` column, and "
        "prediction columns are populated only for the validation and test periods.\n"
        "- `threshold_analysis` and `model_metrics` persist the experiment outputs, so "
        "monitoring views survive a model retrain without code changes."
    )
