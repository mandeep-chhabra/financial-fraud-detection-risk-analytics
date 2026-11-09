"""Metric, threshold and risk-band tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    assign_risk_band,
    choose_risk_bands,
    compute_metrics,
    recall_at_fpr,
    select_operating_threshold,
    threshold_sweep,
)

Y = np.array([1, 0, 1, 0, 0])
SCORES = np.array([0.9, 0.08, 0.2, 0.1, 0.05])


def test_confusion_and_derived_metrics():
    metrics = compute_metrics(Y, SCORES, threshold=0.5)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["tn"] == 3
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["fpr"] == pytest.approx(0.0)
    assert metrics["fnr"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.8)


def test_value_metrics():
    amounts = np.array([100.0, 50.0, 20.0, 10.0, 5.0])
    metrics = compute_metrics(Y, SCORES, threshold=0.5, amounts=amounts, cost_per_investigation=2.0)
    assert metrics["fraud_value_total"] == pytest.approx(120.0)
    assert metrics["fraud_value_captured"] == pytest.approx(100.0)
    assert metrics["fraud_value_captured_pct"] == pytest.approx(100 / 120)
    assert metrics["net_benefit"] == pytest.approx(100.0 - 2.0)


def test_threshold_sweep_is_monotone_in_recall():
    rng = np.random.default_rng(0)
    y = (rng.random(2_000) < 0.1).astype(int)
    scores = np.clip(y * 0.4 + rng.random(2_000) * 0.6, 0, 1)
    sweep = threshold_sweep(y, scores, thresholds=np.round(np.arange(0.1, 0.9, 0.1), 3))
    assert sweep["recall"].is_monotonic_decreasing
    assert (sweep["precision"].between(0, 1)).all()


def test_recall_at_fpr_respects_budget():
    rng = np.random.default_rng(1)
    y = (rng.random(5_000) < 0.15).astype(int)
    scores = np.clip(y * 0.35 + rng.random(5_000) * 0.65, 0, 1)
    result = recall_at_fpr(y, scores, max_fpr=0.02)
    assert 0 <= result["fpr"] <= 0.021
    assert 0 <= result["recall"] <= 1


def test_select_operating_threshold_respects_fpr_budget():
    rng = np.random.default_rng(2)
    y = (rng.random(3_000) < 0.2).astype(int)
    scores = np.clip(y * 0.3 + rng.random(3_000) * 0.7, 0, 1)
    sweep = threshold_sweep(y, scores, thresholds=np.round(np.arange(0.05, 0.95, 0.05), 3))
    choice = select_operating_threshold(sweep, max_fpr=0.05, objective="f1")
    assert choice["threshold"] > 0
    assert choice["validation"]["fpr"] <= 0.051
    assert choice["constraint_relaxed"] is False


def test_select_operating_threshold_relaxes_when_infeasible():
    sweep = pd.DataFrame(
        {
            "threshold": [0.1, 0.2],
            "precision": [0.1, 0.2],
            "recall": [0.9, 0.8],
            "f1": [0.18, 0.32],
            "fpr": [0.5, 0.4],
            "alerts": [100, 90],
        }
    )
    choice = select_operating_threshold(sweep, max_fpr=0.01)
    assert choice["constraint_relaxed"] is True


def test_risk_bands_are_ordered_and_complete():
    sweep = pd.DataFrame(
        {
            "threshold": np.round(np.arange(0.05, 0.95, 0.05), 3),
            "alert_rate": np.linspace(0.9, 0.01, 18),
            "fpr": np.linspace(0.5, 0.001, 18),
            "precision": np.linspace(0.1, 0.9, 18),
            "recall": np.linspace(0.99, 0.1, 18),
            "f1": np.linspace(0.2, 0.2, 18),
            "alerts": np.linspace(900, 10, 18).astype(int),
        }
    )
    bands = choose_risk_bands(sweep, high_threshold=0.7, medium_alert_rate=0.05)
    assert bands["medium_threshold"] < bands["high_threshold"]
    assert bands["medium_band_basis"] in {"alert_rate", "fpr"}

    labels = assign_risk_band([0.01, 0.5, 0.99], bands["high_threshold"], bands["medium_threshold"])
    assert labels[0] == "Low Risk"
    assert labels[2] == "High Risk"
