"""Page 4 - interactive transaction risk assessment."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (
    BAND_COLORS,
    get_scorer,
    money,
    percent,
    probability_contributions,
    query,
    require,
)
from src.config import MAX_TXN_HOUR, PRODUCT_LABELS, RISK_BAND_LABELS
from src.preprocessing.validation import (
    InputValidationError,
    VALID_CARD_BRANDS,
    VALID_CARD_TYPES,
    VALID_PRODUCT_CODES,
    validate_transaction,
    validate_upload,
)


def _optional(text: str, label: str, help_text: str = "") -> str | None:
    """Blank text means 'unknown'; the scorer then falls back to a default."""
    value = st.text_input(label, value="", help=help_text or "Leave blank if unknown")
    return value.strip() or None


def _score_form(scorer) -> None:
    st.subheader("Score a single transaction")
    st.caption(
        "Provide what the reviewer actually knows at authorisation time. The Vesta "
        "C/D/V blocks, the fine-grained identity numbers and the card's behavioural "
        "history are not available in this form, so they fall back to training-set "
        "medians / neutral values. The result is a partial-information estimate and "
        "the endpoint lists exactly which inputs were defaulted."
    )

    with st.form("score_transaction"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input(
                "Amount (USD)", min_value=0.01, max_value=100000.0, value=129.99,
                step=10.0, format="%.2f",
            )
            product = st.selectbox(
                "Product code", VALID_PRODUCT_CODES,
                index=VALID_PRODUCT_CODES.index("W"),
                format_func=lambda p: PRODUCT_LABELS.get(p, p),
            )
            txn_hour = st.number_input(
                "Hours since dataset start", min_value=0, max_value=MAX_TXN_HOUR,
                value=int(MAX_TXN_HOUR * 0.9), step=1,
                help="Controls the calendar features (hour of day, day of week).",
            )
        with col2:
            card4 = st.selectbox("Card brand", ("unknown",) + VALID_CARD_BRANDS)
            card6 = st.selectbox("Card type", ("unknown",) + VALID_CARD_TYPES)
            card1 = _optional("", "card1 (issuer / card identifier)", "Numeric code, e.g. 9500")
            addr1 = _optional("", "addr1 (billing region)")
        with col3:
            email = _optional("", "Purchaser email domain", "e.g. gmail.com")
            device_type = st.selectbox("Device type", ("unknown", "desktop", "mobile"))
            device_info = _optional("", "Device model", "e.g. windows, ios, trident")
            dist1 = _optional("", "dist1 (distance)", "Distance between buyer and merchant")

        has_identity = st.checkbox(
            "Identity / device data is available for this transaction", value=True
        )
        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        record = {
            "amount": amount,
            "product_cd": product,
            "txn_hour": txn_hour,
            "card4": None if card4 == "unknown" else card4,
            "card6": None if card6 == "unknown" else card6,
            "card1": card1,
            "addr1": addr1,
            "p_emaildomain": email,
            "device_type": None if device_type == "unknown" else device_type,
            "device_info": device_info,
            "dist1": dist1,
            "has_identity": int(has_identity),
        }
        try:
            result = scorer.score_transaction(record)
            st.session_state["last_result"] = result
        except InputValidationError as exc:
            st.error("The transaction was rejected:")
            for error in exc.errors:
                st.markdown(f"- {error}")

    result = st.session_state.get("last_result")
    if result:
        probability = result["fraud_probability"]
        band = result["risk_band"]

        st.divider()
        left, right = st.columns([1, 1.4])
        with left:
            thresholds = result["thresholds"]
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=100 * probability,
                    number={"suffix": "%"},
                    title={"text": "Fraud probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": BAND_COLORS[band]},
                        "steps": [
                            {"range": [0, 100 * thresholds["medium_threshold"]], "color": "#DCFCE7"},
                            {"range": [100 * thresholds["medium_threshold"], 100 * thresholds["high_threshold"]], "color": "#FEF3C7"},
                            {"range": [100 * thresholds["high_threshold"], 100], "color": "#FEE2E2"},
                        ],
                    },
                )
            )
            fig.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown(f"### {band}")
            st.markdown(f"**Recommended action:** {result['action']}")
            st.markdown(
                f"- Alert at the tuned operating threshold "
                f"({result['operating_threshold']:.4f}): "
                f"**{'yes' if result['alert'] else 'no'}**\n"
                f"- High-risk cutoff: {thresholds['high_threshold']:.4f}\n"
                f"- Medium-risk cutoff: {thresholds['medium_threshold']:.4f}"
            )
            if result.get("warning"):
                st.warning(result["warning"])
            if probability >= thresholds["high_threshold"]:
                st.error("High-risk decisions should be reviewed by a human before any account action.")
            st.caption(
                "Illustrative policy only: real deployment requires cost validation, "
                "regulatory review, explanation requirements and customer remediation."
            )

        with st.expander("Inputs that were defaulted", expanded=False):
            defaulted = result.get("defaulted_inputs") or []
            st.write(defaulted if defaulted else "None: all raw inputs were provided.")
            st.caption(
                "Defaulted inputs use training medians or neutral history values. Because "
                "the Vesta blocks dominate this model, a form-only score is less reliable "
                "than a score computed on a complete production feature vector."
            )

        engineered = result.get("engineered_features")
        if engineered is not None:
            with st.expander("Why did the model produce this score?", expanded=True):
                try:
                    explanation = probability_contributions(engineered)
                    contrib = explanation["contributions"].head(10).iloc[::-1]
                    fig = go.Figure(
                        go.Bar(
                            x=contrib["shap_value"],
                            y=contrib["feature"],
                            orientation="h",
                            marker_color=[
                                "#DC2626" if v > 0 else "#2563EB" for v in contrib["shap_value"]
                            ],
                        )
                    )
                    fig.update_layout(
                        height=340, xaxis_title="SHAP contribution (log-odds)",
                        yaxis_title="", margin=dict(t=10, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Base value {explanation['base_value']:.3f} in log-odds; positive bars "
                        "push the transaction toward fraud, negative bars toward legitimate. "
                        "SHAP explains the model, not causation."
                    )
                except Exception as exc:  # noqa: BLE001 - explanation is best effort
                    st.info(f"Local explanation unavailable for this model type ({exc}).")

    st.divider()
    st.subheader("Policy check at your threshold")
    threshold = st.slider(
        "Operational threshold used for the alert decision below",
        min_value=0.005, max_value=0.95,
        value=float(min(max(scorer.operating_threshold, 0.005), 0.95)), step=0.005,
    )
    scenario = query("confusion_at_threshold", {"threshold": threshold}).iloc[0]
    tp, fp, fn = int(scenario["tp"]), int(scenario["fp"]), int(scenario["fn"])
    st.write(
        {
            "alerts": tp + fp,
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "false positives": fp,
            "missed frauds": fn,
        }
    )
    st.caption("Computed with SQL over every scored validation and test transaction.")


def _batch_tab(scorer) -> None:
    st.subheader("Score a batch of transactions")
    st.caption(
        "Required columns: amount, product_cd, txn_hour. Optional: card1-card6, addr1, "
        "addr2, dist1, dist2, p_emaildomain, device_type, device_info, has_identity. "
        "Max 5,000 rows."
    )
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    if uploaded is None:
        template = pd.DataFrame(
            [
                {
                    "amount": 129.99, "product_cd": "W", "txn_hour": 4000,
                    "card4": "visa", "card6": "credit", "card1": 9500,
                    "addr1": 299, "p_emaildomain": "gmail.com",
                    "device_type": "desktop", "has_identity": 1,
                },
                {
                    "amount": 1250.0, "product_cd": "C", "txn_hour": 4001,
                    "card4": "mastercard", "card6": "debit", "card1": 1234,
                    "addr1": 325, "p_emaildomain": "yahoo.com",
                    "device_type": "mobile", "has_identity": 0,
                },
            ]
        )
        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="transaction_template.csv",
            mime="text/csv",
        )
        return

    try:
        raw = pd.read_csv(uploaded)
        clean, row_errors = validate_upload(raw)
    except InputValidationError as exc:
        st.error("Upload rejected:")
        for error in exc.errors[:20]:
            st.markdown(f"- {error}")
        return
    except Exception as exc:  # noqa: BLE001 - malformed CSV
        st.error(f"Could not read the file: {exc}")
        return

    if row_errors:
        st.warning(f"{len(row_errors)} row(s) were rejected; the rest were scored.")
        with st.expander("Show rejected rows"):
            for error in row_errors[:50]:
                st.markdown(f"- {error}")

    scored, defaulted = scorer.score_batch(clean)
    display = scored.copy()
    display["fraud_probability"] = display["fraud_probability"].round(6)
    st.dataframe(display, hide_index=True, use_container_width=True)
    if defaulted:
        st.caption("Inputs defaulted for at least one row: " + ", ".join(defaulted))
    st.download_button(
        "Download scored transactions",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )


def render() -> None:
    st.title("Transaction Risk Assessment")
    st.caption("Enter a transaction, get a probability, a risk band and the recommended action.")
    if not require(["model", "encoder", "thresholds", "database"]):
        return
    scorer = get_scorer()

    tabs = st.tabs(["Single transaction", "Batch scoring"])
    with tabs[0]:
        _score_form(scorer)
    with tabs[1]:
        _batch_tab(scorer)
