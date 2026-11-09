"""End-to-end scoring tests: raw record -> engineering -> probability -> band."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CATEGORICAL_RAW,
    FEATURE_COLUMNS,
    ID_NUMERIC,
    MAX_TXN_HOUR,
    M_COLUMNS,
    TXN_DT_MIN,
)
from src.features.engineering import (
    add_category_codes,
    add_frequency_features,
    add_history_features,
    add_row_features,
    compute_frequency_maps,
    feature_dictionary,
)
from src.models.predict import FraudScorer
from src.models.registry import build_lightgbm
from src.models.train import _defaults_from_training
from src.preprocessing.validation import InputValidationError

RECORD = {
    "amount": 249.99,
    "product_cd": "C",
    "txn_hour": 4000,
    "card4": "visa",
    "card6": "credit",
    "card1": "9500",
    "addr1": "299",
    "p_emaildomain": "gmail.com",
    "device_type": "desktop",
}


def _synthetic_raw(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = np.arange(n)
    data: dict[str, np.ndarray] = {
        "TransactionDT": TXN_DT_MIN + index * 3_600,
        "TransactionAmt": rng.uniform(5, 600, n),
        "card1": rng.integers(1000, 1010, n),
        "card2": rng.integers(100, 110, n),
        "card3": rng.integers(150, 160, n),
        "card5": rng.integers(200, 210, n),
        "addr1": rng.integers(100, 105, n),
        "addr2": rng.integers(80, 90, n),
        "dist1": rng.uniform(0, 500, n),
        "dist2": rng.uniform(0, 500, n),
    }
    for column in M_COLUMNS:
        data[column] = rng.choice(["T", "F"], n)
    for i in range(1, 15):
        data[f"C{i}"] = rng.integers(0, 20, n).astype(float)
    for i in range(1, 16):
        data[f"D{i}"] = rng.integers(0, 400, n).astype(float)
    for i in range(1, 340):
        data[f"V{i}"] = rng.normal(0, 1, n)
    for column in ID_NUMERIC:
        data[column] = rng.normal(0, 1, n)
    data["ProductCD"] = rng.choice(["W", "C", "R"], n)
    data["card4"] = rng.choice(["visa", "mastercard"], n)
    data["card6"] = rng.choice(["debit", "credit"], n)
    data["P_emaildomain"] = rng.choice(["gmail.com", "yahoo.com"], n)
    data["R_emaildomain"] = np.nan
    data["DeviceType"] = rng.choice(["desktop", "mobile"], n)
    data["DeviceInfo"] = rng.choice(["windows", "ios"], n)
    for column in ("id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
                   "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"):
        data[column] = rng.choice(["a", "b"], n)
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def scorer(tmp_path_factory) -> FraudScorer:
    tmp_path = tmp_path_factory.mktemp("scorer")
    raw = _synthetic_raw()
    raw, mappings = add_category_codes(raw)
    maps = compute_frequency_maps(raw)
    raw = add_frequency_features(raw, maps)
    raw = add_row_features(raw)
    raw = add_history_features(raw)

    y = (raw["TransactionAmt"] > 550).astype(int).to_numpy()
    X = raw[FEATURE_COLUMNS]
    model = build_lightgbm(n_estimators=25, learning_rate=0.2, num_leaves=15)
    model.fit(X, y, categorical_feature=[c for c in FEATURE_COLUMNS if c.startswith("cat_")])

    raw_defaults, feature_defaults = _defaults_from_training(X, n=len(X))
    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "thresholds": {"high_threshold": 0.6, "medium_threshold": 0.1},
        "operating_threshold": 0.6,
        "model_name": "test-lightgbm",
        "raw_defaults": raw_defaults,
        "feature_defaults": feature_defaults,
        "cost_per_investigation": 4.0,
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(bundle, model_path)
    encoder_path = tmp_path / "encoder.json"
    encoder_path.write_text(
        json.dumps({"category_mappings": mappings, "frequency_maps": maps}), encoding="utf-8"
    )
    return FraudScorer.load(model_path, encoder_path)


def test_prepare_features_shape(scorer):
    from src.preprocessing.validation import records_to_frame, validate_transaction

    frame = records_to_frame([validate_transaction(RECORD)])
    engineered, defaulted = scorer.prepare_features(frame)
    assert engineered.shape == (1, len(FEATURE_COLUMNS))
    assert list(engineered.columns) == FEATURE_COLUMNS
    # Vesta C/D/V blocks cannot come from the form, so they must be disclosed.
    assert len(defaulted) > 0


def test_score_transaction_returns_valid_decision(scorer):
    result = scorer.score_transaction(RECORD)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_band"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert result["action"]
    assert result["model_name"] == "test-lightgbm"
    engineered = result["engineered_features"]
    assert engineered.shape == (1, len(FEATURE_COLUMNS))


def test_risk_band_follows_thresholds(scorer):
    result = scorer.score_transaction(RECORD)
    probability = result["fraud_probability"]
    if probability >= 0.6:
        assert result["risk_band"] == "High Risk"
    elif probability >= 0.1:
        assert result["risk_band"] == "Medium Risk"
    else:
        assert result["risk_band"] == "Low Risk"


def test_invalid_record_is_rejected(scorer):
    with pytest.raises(InputValidationError):
        scorer.score_transaction({"amount": -10, "product_cd": "Z", "txn_hour": 1})


def test_score_batch_matches_single_transaction(scorer):
    from src.preprocessing.validation import records_to_frame, validate_transaction

    clean = records_to_frame([validate_transaction(RECORD)])
    batch, defaulted = scorer.score_batch(clean)
    single = scorer.score_transaction(RECORD)
    assert batch["fraud_probability"].iloc[0] == pytest.approx(single["fraud_probability"], abs=1e-6)
    assert "action" in batch.columns and "alert" in batch.columns
    assert isinstance(defaulted, list)


def test_feature_dictionary_is_consistent():
    dictionary = feature_dictionary()
    assert dictionary["n_features"] == len(FEATURE_COLUMNS)
    assert dictionary["n_categorical"] == len([c for c in FEATURE_COLUMNS if c.startswith("cat_")])
    assert dictionary["columns"] == FEATURE_COLUMNS
    assert dictionary["blocks"]


def test_max_txn_hour_covers_the_timeline():
    assert MAX_TXN_HOUR >= 4_000
