"""Shared dashboard helpers: cached data access, formatting and local SHAP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import (
    DB_PATH,
    FEATURE_ENCODER_PATH,
    MODEL_PATH,
    PREDICTIONS_PARQUET,
    QUALITY_REPORT,
    SHAP_DIR,
    TEST_METRICS_JSON,
    THRESHOLD_CONFIG_PATH,
)
from src.database.queries import QUERIES, run_query
from src.models.predict import FraudScorer

BAND_COLORS = {"High Risk": "#DC2626", "Medium Risk": "#F59E0B", "Low Risk": "#16A34A"}


@st.cache_data(ttl=600, show_spinner=False)
def query(name: str, params: dict | None = None) -> pd.DataFrame:
    return run_query(name, params or {})


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_figure_bytes(path: str) -> bytes | None:
    file = Path(path)
    return file.read_bytes() if file.exists() else None


@st.cache_resource(show_spinner="Loading fraud model...")
def get_scorer() -> FraudScorer:
    return FraudScorer.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_explainer():
    import joblib

    bundle = joblib.load(MODEL_PATH)
    return shap.TreeExplainer(bundle["model"]), bundle


@st.cache_data(show_spinner=False)
def load_shap_sample() -> tuple[pd.DataFrame, np.ndarray, float, list[str]]:
    meta = load_json(str(SHAP_DIR / "meta.json"))
    frame = pd.read_parquet(SHAP_DIR / "sample.parquet")
    values = np.load(SHAP_DIR / "shap_values.npy")
    return frame, values, float(meta["base_value"]), list(meta["feature_columns"])


def artifacts_status() -> dict[str, bool]:
    return {
        "database": DB_PATH.exists(),
        "model": MODEL_PATH.exists(),
        "encoder": FEATURE_ENCODER_PATH.exists(),
        "thresholds": THRESHOLD_CONFIG_PATH.exists(),
        "predictions": PREDICTIONS_PARQUET.exists(),
        "shap": (SHAP_DIR / "shap_values.npy").exists(),
        "quality_report": QUALITY_REPORT.exists(),
        "test_metrics": TEST_METRICS_JSON.exists(),
    }


def require(keys: list[str]) -> bool:
    """Show a friendly setup message and return False when artefacts are missing."""
    status = artifacts_status()
    missing = [k for k in keys if not status.get(k)]
    if missing:
        st.warning(
            "Missing artefacts: **" + ", ".join(missing) + "**. Run `python scripts/run_all.py` "
            "to build the dataset, models, database and SHAP explanations."
        )
        return False
    return True


def money(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.1f}K"
    return f"${value:,.0f}"


def number(value, decimals: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{decimals}f}"


def percent(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}%"


def kpi(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def probability_contributions(feature_frame: pd.DataFrame) -> dict:
    """Top SHAP contributions for one engineered feature row.

    Used by the interactive scorer so a user can see *why* the form's transaction
    received its risk score. Only tree models are supported; the shipped model is
    LightGBM, and the caller checks the explainer type.
    """
    explainer, bundle = get_explainer()
    model = bundle["model"]
    X = feature_frame[bundle["feature_columns"]].astype("float32")
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        values = np.asarray(raw[1])[0]
        expected = np.ravel(explainer.expected_value)[1]
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, 1]
        values = values[0]
        expected = np.ravel(explainer.expected_value)
        expected = float(expected[1] if expected.size > 1 else expected[0])

    contributions = (
        pd.DataFrame(
            {
                "feature": bundle["feature_columns"],
                "value": X.iloc[0].to_numpy(),
                "shap_value": values.astype(float),
            }
        )
        .assign(abs_value=lambda d: d["shap_value"].abs())
        .sort_values("abs_value", ascending=False)
        .drop(columns="abs_value")
    )
    return {"base_value": float(expected), "contributions": contributions.reset_index(drop=True)}


def model_metadata() -> dict:
    try:
        frame = query("model_metadata")
        return dict(zip(frame["key"], frame["value"]))
    except Exception:
        return {}


def threshold_config() -> dict:
    if THRESHOLD_CONFIG_PATH.exists():
        return load_json(str(THRESHOLD_CONFIG_PATH))
    return {}


def metrics_table(metric_names: list[str]) -> pd.DataFrame:
    frame = query("model_metrics")
    frame = frame[frame["metric"].isin(metric_names)]
    return frame.pivot_table(index="metric", columns="eval_split", values="value", aggfunc="first")


__all__ = [
    "BAND_COLORS",
    "QUERIES",
    "artifacts_status",
    "get_explainer",
    "get_scorer",
    "kpi",
    "load_figure_bytes",
    "load_json",
    "load_shap_sample",
    "metrics_table",
    "model_metadata",
    "money",
    "number",
    "percent",
    "probability_contributions",
    "query",
    "require",
    "threshold_config",
]
