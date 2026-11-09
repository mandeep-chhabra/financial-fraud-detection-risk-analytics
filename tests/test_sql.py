"""SQL analytics tests against a small deterministic fixture database."""

from __future__ import annotations

import sqlite3

import pytest

from src.database.build_db import SCHEMA
from src.database.queries import QUERIES, get_connection, run_query

ROWS = [
    # txn_id, dt, hour, day, product, card4, card6, card1, addr1, email, identity, amount, fraud, split, prob, band, alerted
    (1, 100_000, 1, 10, "W", "visa", "credit", 9500, 299, "gmail.com", 1, 100.0, 1, "test", 0.90, "High Risk", 1),
    (2, 190_000, 2, 11, "W", "visa", "credit", 9501, 299, "gmail.com", 0, 50.0, 0, "test", 0.20, "Low Risk", 0),
    (3, 190_100, 2, 11, "C", "mastercard", "debit", 9502, 325, "yahoo.com", 1, 300.0, 1, "test", 0.70, "Medium Risk", 1),
    (4, 280_000, 3, 12, "C", "mastercard", "debit", 9503, 325, "yahoo.com", 1, 20.0, 0, "test", 0.10, "Low Risk", 0),
    (5, 280_100, 3, 12, "W", "visa", "credit", 9504, 299, None, 0, 10.0, 0, "test", 0.05, "Low Risk", 0),
]

INSERT = """
INSERT INTO transactions (
    txn_id, transaction_dt, txn_hour, day_index, product_cd, card4, card6, card1,
    addr1, p_emaildomain, has_identity, amount, is_fraud, split,
    fraud_probability, risk_band, alerted
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "fixture.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(INSERT, ROWS)
    conn.commit()
    conn.close()
    return path


def test_overview_counts(db):
    row = run_query("overview", db_path=db).iloc[0]
    assert row["total_transactions"] == 5
    assert row["fraud_transactions"] == 2
    assert row["legitimate_transactions"] == 3
    assert row["fraud_rate_pct"] == pytest.approx(40.0)
    assert row["total_value"] == pytest.approx(480.0)
    assert row["fraud_value"] == pytest.approx(400.0)


def test_fraud_by_product(db):
    frame = run_query("fraud_by_product", db_path=db).set_index("product_cd")
    assert frame.loc["W", "transactions"] == 3
    assert frame.loc["W", "frauds"] == 1
    assert frame.loc["C", "frauds"] == 1
    assert frame.loc["C", "fraud_rate_pct"] == pytest.approx(50.0)


def test_confusion_at_threshold(db):
    row = run_query("confusion_at_threshold", {"threshold": 0.5}, db_path=db).iloc[0]
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 0, 0, 3)

    loose = run_query("confusion_at_threshold", {"threshold": 0.15}, db_path=db).iloc[0]
    assert (loose["tp"], loose["fp"], loose["fn"], loose["tn"]) == (2, 1, 0, 2)


def test_risk_bands_are_ordered(db):
    frame = run_query("risk_bands", db_path=db)
    assert frame["risk_band"].tolist() == ["High Risk", "Medium Risk", "Low Risk"]
    assert frame["transactions"].tolist() == [1, 1, 3]


def test_identity_coverage(db):
    frame = run_query("identity_coverage", db_path=db).set_index("identity_status")
    assert frame.loc["identity present", "transactions"] == 3
    assert frame.loc["identity absent", "transactions"] == 2


def test_daily_trends_group_by_day(db):
    frame = run_query("daily_trends", db_path=db)
    assert frame["day"].tolist() == [10, 11, 12]
    assert frame["transactions"].tolist() == [1, 2, 2]


def test_top_risk_transactions_respects_limit(db):
    frame = run_query("top_risk_transactions", {"limit": 2}, db_path=db)
    assert len(frame) == 2
    assert frame["fraud_probability"].iloc[0] == pytest.approx(0.90)


def test_unknown_query_raises(db):
    with pytest.raises(KeyError):
        run_query("does_not_exist", db_path=db)


def test_missing_database_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_connection(tmp_path / "absent.db")


def test_every_registered_query_is_valid_sql(db):
    """Guards against a typo breaking a dashboard page at runtime."""
    for name, sql in QUERIES.items():
        params = {}
        if ":limit" in sql:
            params["limit"] = 5
        if ":threshold" in sql:
            params["threshold"] = 0.5
        frame = run_query(name, params or None, db_path=db)
        assert hasattr(frame, "columns"), name
