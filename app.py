"""Streamlit entry point: six-page fraud risk analytics dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from dashboard import (
    eda_page,
    explainability_page,
    model_performance,
    overview,
    risk_assessment,
    sql_insights,
)
from dashboard.common import artifacts_status, model_metadata

PAGES = {
    "Overview": overview.render,
    "Exploratory Analytics": eda_page.render,
    "Model Performance": model_performance.render,
    "Transaction Risk Assessment": risk_assessment.render,
    "Explainability": explainability_page.render,
    "SQL Insights": sql_insights.render,
}

PAGE_ICONS = {
    "Overview": "📊",
    "Exploratory Analytics": "🔍",
    "Model Performance": "📈",
    "Transaction Risk Assessment": "🛡️",
    "Explainability": "🧠",
    "SQL Insights": "🗄️",
}


def main() -> None:
    st.set_page_config(
        page_title="Fraud Risk Analytics",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("🛡️ Fraud Risk Analytics")
        st.caption("IEEE-CIS · 590,540 transactions · boosted-tree risk scoring")
        page = st.radio(
            "Navigation",
            list(PAGES),
            format_func=lambda name: f"{PAGE_ICONS[name]}  {name}",
            label_visibility="collapsed",
        )

        st.divider()
        status = artifacts_status()
        meta = model_metadata()
        if meta:
            st.markdown("**Selected model**")
            st.caption(meta.get("model", "n/a"))
            st.markdown(
                f"Operating threshold **{float(meta.get('operating_threshold', 0)):.4f}**  \n"
                f"High risk ≥ **{float(meta.get('high_risk_threshold', 0)):.4f}**  \n"
                f"Medium risk ≥ **{float(meta.get('medium_risk_threshold', 0)):.4f}**"
            )
            st.caption(
                f"Trained on {int(float(meta.get('train_rows', 0))):,} transactions; "
                f"evaluated on {int(float(meta.get('test_rows', 0))):,} held-out rows."
            )
        else:
            st.caption("Run the pipeline to populate model metadata.")

        missing = [k for k, ok in status.items() if not ok]
        if missing:
            st.warning("Missing: " + ", ".join(missing))

        st.divider()
        st.caption(
            "Illustrative decision policy, not banking advice. The $4 investigation "
            "cost and 1% review budget are assumptions, not bank figures."
        )

    try:
        PAGES[page]()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python scripts/run_all.py", language="bash")
    except Exception as exc:  # noqa: BLE001 - keep the dashboard alive
        st.error(f"Page failed to render: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


main()
