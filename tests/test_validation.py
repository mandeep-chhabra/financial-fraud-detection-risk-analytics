"""Input validation tests for the scoring endpoint."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import MAX_TXN_HOUR
from src.preprocessing.validation import (
    InputValidationError,
    records_to_frame,
    validate_transaction,
    validate_upload,
)

VALID = {
    "amount": 129.99,
    "product_cd": "W",
    "txn_hour": 4000,
    "card4": "visa",
    "card6": "credit",
    "card1": "9500",
    "addr1": "299",
    "p_emaildomain": "gmail.com",
    "device_type": "desktop",
}


def test_valid_record_is_normalised():
    clean = validate_transaction(VALID)
    assert clean["product_cd"] == "W"
    assert clean["amount"] == pytest.approx(129.99)
    assert clean["card1"] == 9500
    assert clean["has_identity"] == 1
    assert clean["warning"] is None


def test_missing_required_fields_are_all_reported():
    with pytest.raises(InputValidationError) as exc:
        validate_transaction({"product_cd": "W"})
    assert "amount" in str(exc.value)
    assert "txn_hour" in str(exc.value)


def test_unknown_product_is_rejected():
    record = {**VALID, "product_cd": "Z"}
    with pytest.raises(InputValidationError) as exc:
        validate_transaction(record)
    assert "product_cd" in str(exc.value)


def test_negative_amount_is_rejected():
    with pytest.raises(InputValidationError):
        validate_transaction({**VALID, "amount": -5})


def test_non_numeric_amount_is_rejected():
    with pytest.raises(InputValidationError):
        validate_transaction({**VALID, "amount": "one hundred"})


def test_txn_hour_out_of_range_is_rejected():
    with pytest.raises(InputValidationError):
        validate_transaction({**VALID, "txn_hour": MAX_TXN_HOUR + 1})


def test_unknown_card_brand_warns_but_scores():
    clean = validate_transaction({**VALID, "card4": "jcb"})
    assert clean["card4"] is None
    assert "warning" in clean


def test_blank_optional_fields_become_none():
    clean = validate_transaction({**VALID, "card1": "", "addr1": None, "p_emaildomain": ""})
    assert clean["card1"] is None
    assert clean["addr1"] is None
    assert clean["p_emaildomain"] is None


def test_identity_flag_defaults_to_presence_of_device_fields():
    clean = validate_transaction({"amount": 10, "product_cd": "C", "txn_hour": 10})
    assert clean["has_identity"] == 0
    with_device = validate_transaction({"amount": 10, "product_cd": "C", "txn_hour": 10, "device_type": "mobile"})
    assert with_device["has_identity"] == 1


def test_records_to_frame_keeps_expected_columns():
    frame = records_to_frame([validate_transaction(VALID)])
    assert list(frame.columns) == [
        "amount", "txn_hour", "product_cd", "card1", "card2", "card3", "card4",
        "card5", "card6", "addr1", "addr2", "dist1", "dist2", "p_emaildomain",
        "device_type", "device_info", "has_identity",
    ]


def test_upload_validates_rows_and_reports_errors():
    frame = pd.DataFrame(
        [
            {"amount": 10.0, "product_cd": "W", "txn_hour": 10},
            {"amount": -1.0, "product_cd": "W", "txn_hour": 10},
            {"amount": 20.0, "product_cd": "Q", "txn_hour": 10},
        ]
    )
    clean, errors = validate_upload(frame)
    assert len(clean) == 1
    assert len(errors) == 2


def test_upload_rejects_missing_columns():
    with pytest.raises(InputValidationError):
        validate_upload(pd.DataFrame([{"amount": 10.0}]))
