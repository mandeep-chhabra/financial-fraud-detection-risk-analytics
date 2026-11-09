"""Page 2 - exploratory analytics driven by SQL aggregates and saved EDA figures."""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import load_json, money, number, percent, query, require
from src.config import EDA_SUMMARY_JSON, FIGURES_DIR, QUALITY_REPORT


def _image(name: str, caption: str) -> None:
    path = Path(FIGURES_DIR) / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure {name} not generated yet. Run `python scripts/run_eda.py`.")


def render() -> None:
    st.title("Exploratory Analytics")
    st.caption("What the data looks like before any model exists.")
    if not require(["database", "quality_report"]):
        return

    report = load_json(str(QUALITY_REPORT))
    summary = load_json(str(EDA_SUMMARY_JSON)) if Path(EDA_SUMMARY_JSON).exists() else {}

    top = st.columns(5)
    top[0].metric("Transactions", number(report["rows"]))
    top[1].metric("Frauds", number(report["frauds"]))
    top[2].metric("Fraud prevalence", f"{100 * report['fraud_rate']:.3f}%")
    top[3].metric("Imbalance ratio", f"1 : {report['imbalance_ratio']:.0f}")
    top[4].metric("Days covered", f"{report['days_covered']:.0f}")

    st.divider()
    tabs = st.tabs(["Distribution", "Amounts", "Segments", "Time", "Identity", "Data quality"])

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            _image("class_distribution.png", "Class distribution (log scale)")
        with right:
            _image(
                "correlation_with_fraud.png",
                "Correlation of the strongest single features with the label",
            )
        st.info(
            "Prevalence is low enough that accuracy is meaningless: predicting "
            "'legitimate' for every row already scores "
            f"{100 * (1 - report['fraud_rate']):.2f}%. The project therefore optimises "
            "recall inside a false-positive budget and reports PR-AUC."
        )

    with tabs[1]:
        stats = query("amount_statistics")
        st.dataframe(
            stats.rename(
                columns={
                    "is_fraud": "Fraud",
                    "transactions": "Transactions",
                    "mean_amount": "Mean amount",
                    "min_amount": "Min",
                    "max_amount": "Max",
                    "total_amount": "Total value",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        _image("amount_distribution.png", "Transaction amount distribution by class (log scale)")
        bands = query("fraud_by_amount_band")
        fig = px.bar(
            bands, x="amount_band", y="fraud_rate_pct", text="fraud_rate_pct",
            labels={"amount_band": "Amount band", "fraud_rate_pct": "Fraud rate (%)"},
        )
        fig.update_traces(texttemplate="%{text:.2f}%")
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if summary:
            st.markdown(
                f"- Fraudulent transactions are larger on average "
                f"({money(summary['fraud_amount_mean'])} vs {money(summary['legit_amount_mean'])}).\n"
                f"- Median fraud amount is {summary['fraud_amount_multiple_of_median']:.1f}x the "
                "median legitimate amount.\n"
                "- Fraud rate rises with amount, but amount alone is not a classifier: the "
                "distributions overlap heavily."
            )

    with tabs[2]:
        by_product = query("fraud_by_product")
        fig = px.scatter(
            by_product,
            x="transactions",
            y="fraud_rate_pct",
            size="frauds",
            color="product_cd",
            text="product_cd",
            log_x=True,
            labels={"transactions": "transactions (log)", "fraud_rate_pct": "fraud rate (%)"},
        )
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_product, hide_index=True, use_container_width=True)

        st.markdown("**Card brand and type**")
        by_card = query("fraud_by_card")
        st.dataframe(by_card, hide_index=True, use_container_width=True)
        _image("fraud_by_card.png", "Fraud rate by card brand and card type")

    with tabs[3]:
        daily = query("daily_trends")
        fig = go.Figure()
        fig.add_bar(x=daily["day"], y=daily["frauds"], name="Frauds", marker_color="#DC2626")
        fig.add_scatter(
            x=daily["day"], y=daily["transactions"], name="Transactions", yaxis="y2",
            line=dict(color="#6B7280"),
        )
        fig.update_layout(
            height=380,
            yaxis=dict(title="Frauds"),
            yaxis2=dict(title="Transactions", overlaying="y", side="right"),
            margin=dict(t=20, b=10),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(
            daily, x="day", y="fraud_rate_pct", markers=False,
            labels={"day": "Day", "fraud_rate_pct": "Fraud rate (%)"},
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Fraud prevalence is stable across the 182 days, so the temporal split "
            "does not create a prevalence shock between periods."
        )

        hourly = query("hourly_fraud")
        fig = px.line(
            hourly, x="hour_of_day", y="fraud_rate_pct", markers=True,
            labels={"hour_of_day": "Hour of day (UTC offset)", "fraud_rate_pct": "Fraud rate (%)"},
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        identity = query("identity_coverage")
        st.dataframe(identity, hide_index=True, use_container_width=True)
        st.warning(
            "Identity data exists for only "
            f"{100 * report['identity_coverage']:.1f}% of transactions, and those rows "
            f"are fraudulent {100 * report['fraud_rate_with_identity']:.2f}% of the time "
            f"versus {100 * report['fraud_rate_without_identity']:.2f}% without it. The "
            "flag is legitimate at scoring time but partly encodes how the processor "
            "collected data, so a production model would need to be monitored for it."
        )
        st.json(report["leakage_audit"], expanded=False)

    with tabs[5]:
        st.markdown("**Missing data (top 25 columns)**")
        missing = report.get("missing_top", {})
        if missing:
            st.dataframe(
                {
                    "column": list(missing),
                    "missing %": [round(100 * v, 2) for v in missing.values()],
                },
                hide_index=True,
                use_container_width=True,
            )
        _image("missing_values.png", "Missing rate of the 25 sparsest columns")

        buckets = report.get("missing_buckets", {})
        if buckets:
            st.markdown("**Columns by missing-rate bucket**")
            st.write(buckets)

        st.markdown("**Keys and identifiers**")
        st.write(
            {
                "duplicate transaction ids": report["duplicate_transaction_ids"],
                "unique cards (card1)": report["unique_cards"],
                "unique billing regions (addr1)": report["unique_addresses"],
                "identity rows": report["identity_rows"],
            }
        )
        st.markdown("**Known limitations**")
        for item in report["known_limitations"]:
            st.markdown(f"- {item}")
