"""Page 5 - global and local SHAP explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import load_figure_bytes, load_shap_sample, require
from src.config import FIGURES_DIR, SHAP_DIR


def render() -> None:
    st.title("Explainability")
    st.caption("SHAP values for the shipped model. Explanations describe the model, not causation.")
    if not require(["shap"]):
        return

    importance = pd.read_csv(SHAP_DIR / "global_importance.csv")
    sample, shap_values, base_value, feature_columns = load_shap_sample()
    top_features = importance["feature"].head(4).tolist()

    tabs = st.tabs(["Global drivers", "Transaction-level explanation", "Case studies"])

    with tabs[0]:
        top = importance.head(15).iloc[::-1]
        fig = px.bar(
            top, x="mean_abs_shap", y="feature", orientation="h",
            labels={"mean_abs_shap": "mean |SHAP| (log-odds)", "feature": ""},
            color="mean_abs_shap", color_continuous_scale="Reds",
        )
        fig.update_layout(height=480, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            data = load_figure_bytes(str(Path(FIGURES_DIR) / "shap_summary.png"))
            if data:
                st.image(data, caption="Beeswarm summary: direction and spread of each effect", use_container_width=True)
        with right:
            dependent = Path(FIGURES_DIR) / f"shap_dependence_{top_features[0]}.png"
            data = load_figure_bytes(str(dependent))
            if data:
                st.image(data, caption=f"Dependence plot: {top_features[0]}", use_container_width=True)

        st.dataframe(
            importance.head(15).style.format(
                {"mean_abs_shap": "{:.5f}", "mean_shap": "{:.5f}", "mean_value": "{:.3f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.info(
            "The model leans on a small set of Vesta `V`/`C` columns and on "
            "`TransactionAmt`, `card1` history and identity availability. Those V-columns "
            "are anonymised aggregates whose exact definition is undisclosed, and many "
            "of them are correlated with each other, so their individual attribution is "
            "shared somewhat arbitrarily. The correct reading is 'the model relies on "
            "these signals', not 'these signals cause fraud'."
        )

    with tabs[1]:
        st.subheader("Why did this transaction get its score?")
        options = sample.sort_values("fraud_probability", ascending=False).head(200)
        options = options.assign(
            label=lambda d: d.apply(
                lambda r: f"txn {int(r['txn_id'])} · p={r['fraud_probability']:.4f} · "
                          f"{'FRAUD' if r['isFraud'] == 1 else 'legit'} · "
                          f"{r['product_cd']} · ${r['amount_txn']:,.0f}",
                axis=1,
            )
        )
        selection = st.selectbox("Scored test transactions (highest risk first)", options["label"].tolist())
        row = options[options["label"] == selection].iloc[0]
        index = int(row["explanation_index"])

        values = shap_values[index]
        contributions = (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "value": sample.iloc[index][feature_columns].astype(float).to_numpy(),
                    "shap_value": values,
                }
            )
            .assign(abs_value=lambda d: d["shap_value"].abs())
            .sort_values("abs_value", ascending=False)
            .head(12)
            .iloc[::-1]
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model probability", f"{row['fraud_probability']:.4f}")
        col2.metric("Actual label", "Fraud" if row["isFraud"] == 1 else "Legitimate")
        col3.metric("Amount", f"${row['amount_txn']:,.0f}")
        col4.metric("Product", str(row["product_cd"]))

        fig = go.Figure(
            go.Bar(
                x=contributions["shap_value"],
                y=contributions["feature"],
                orientation="h",
                marker_color=["#DC2626" if v > 0 else "#2563EB" for v in contributions["shap_value"]],
                customdata=contributions["value"],
                hovertemplate="%{y}<br>SHAP %{x:.4f}<br>value %{customdata:.4g}<extra></extra>",
            )
        )
        fig.update_layout(
            height=440, xaxis_title="SHAP contribution (log-odds)", yaxis_title="",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        log_odds = base_value + float(values.sum())
        st.caption(
            f"Base value {base_value:.4f} + sum of contributions {values.sum():.4f} = "
            f"{log_odds:.4f} log-odds, i.e. p = {1 / (1 + np.exp(-log_odds)):.4f}."
        )
        st.caption(
            "Red bars increase the fraud score relative to the average transaction; blue bars reduce it."
        )

        st.dataframe(
            contributions.iloc[::-1].style.format({"value": "{:.4g}", "shap_value": "{:+.4f}"}),
            hide_index=True,
            use_container_width=True,
        )

    with tabs[2]:
        st.subheader("Representative explanations")
        meta = pd.read_json(SHAP_DIR / "meta.json", typ="series")
        cases = [
            ("shap_waterfall_top_fraud.png", "Highest-scored fraud in the sample"),
            ("shap_waterfall_top_legitimate_alert.png", "Highest-scored legitimate transaction (false-positive candidate)"),
            ("shap_waterfall_lowest_scored_fraud.png", "Lowest-scored fraud (a miss)"),
        ]
        for name, caption in cases:
            data = load_figure_bytes(str(Path(FIGURES_DIR) / name))
            if data:
                st.image(data, caption=caption, use_container_width=True)
        st.caption(f"Sample size: {int(meta['n_samples'])} scored test transactions.")
        st.warning(
            "Do not read feature importance as a causal claim. SHAP attributes the "
            "model's output given correlated features; a large value for a V-column "
            "means the model relies on it, not that it causes fraud."
        )
