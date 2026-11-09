"""Feature engineering tests, with emphasis on leak-free history aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TXN_DT_MIN
from src.features.engineering import (
    add_category_codes,
    add_frequency_features,
    add_history_features,
    add_row_features,
    compute_frequency_maps,
    feature_matrix,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["TransactionAmt"] = frame["TransactionAmt"].astype("float32")
    return frame


@pytest.fixture
def history_frame() -> pd.DataFrame:
    # Two cards, interleaved timestamps, deliberately unsorted input order.
    return _frame(
        [
            {"TransactionDT": 3_000, "TransactionAmt": 30.0, "card1": 1, "addr1": 10, "D1": 5},
            {"TransactionDT": 1_000, "TransactionAmt": 10.0, "card1": 1, "addr1": 10, "D1": 5},
            {"TransactionDT": 1_500, "TransactionAmt": 5.0, "card1": 2, "addr1": 20, "D1": 1},
            {"TransactionDT": 2_000, "TransactionAmt": 20.0, "card1": 1, "addr1": 10, "D1": 5},
            {"TransactionDT": 100_000, "TransactionAmt": 7.0, "card1": 1, "addr1": 10, "D1": 5},
        ]
    )


def test_history_features_use_only_earlier_rows(history_frame):
    out = add_history_features(history_frame)
    card_one = out[out["card1"] == 1].sort_values("TransactionDT")

    assert card_one["card1_prior_count"].tolist() == [0.0, 1.0, 2.0, 3.0]
    # First transaction has no history -> NaN mean, handled downstream.
    assert np.isnan(card_one["card1_prior_amt_mean"].iloc[0])
    assert card_one["card1_prior_amt_mean"].iloc[1] == pytest.approx(10.0)
    assert card_one["card1_prior_amt_mean"].iloc[2] == pytest.approx(15.0)
    assert card_one["card1_prior_amt_mean"].iloc[3] == pytest.approx(20.0)

    # Ratios are neutral (1.0) when the card has no history.
    assert card_one["amount_vs_card1_mean"].iloc[0] == pytest.approx(1.0)
    assert card_one["amount_vs_card1_mean"].iloc[1] == pytest.approx(2.0)

    # Seconds since the previous transaction on the same card.
    assert card_one["card1_secs_since_prev"].iloc[1] == pytest.approx(1_000)
    assert np.isnan(card_one["card1_secs_since_prev"].iloc[0])

    # The last row is 98,000s after the previous one: outside 24h, inside 7d.
    assert card_one["card1_prior_count_24h"].iloc[3] == 0.0
    assert card_one["card1_prior_count_7d"].iloc[3] == 3.0


def test_second_card_history_is_independent(history_frame):
    out = add_history_features(history_frame)
    card_two = out[out["card1"] == 2]
    assert card_two["card1_prior_count"].iloc[0] == 0.0
    assert card_two["uid_prior_count"].iloc[0] == 0.0


def test_future_perturbation_cannot_change_past_features(history_frame):
    """Changing later transactions must not alter earlier rows' features."""
    baseline = add_history_features(history_frame).sort_values("TransactionDT")

    perturbed = history_frame.copy()
    perturbed.loc[perturbed["TransactionDT"] >= 3_000, "TransactionAmt"] *= 1_000
    after = add_history_features(perturbed).sort_values("TransactionDT")

    early = baseline["TransactionDT"] < 3_000
    columns = [
        "card1_prior_count", "card1_prior_amt_mean", "card1_prior_amt_std",
        "amount_vs_card1_mean", "card1_prior_count_24h", "uid_prior_count",
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[early, columns].reset_index(drop=True),
        after.loc[early, columns].reset_index(drop=True),
    )


def test_amount_features():
    frame = _frame([{"TransactionDT": TXN_DT_MIN, "TransactionAmt": 125.0, "card1": 1, "addr1": 1, "D1": 0}])
    out = add_row_features(frame)
    assert out["log_amount"].iloc[0] == pytest.approx(np.log1p(125.0))
    assert out["amount_cents"].iloc[0] == 0
    assert out["amount_is_whole"].iloc[0] == 1
    assert out["amount_is_round"].iloc[0] == 0  # 125 is not a multiple of 100

    round_frame = _frame([{"TransactionDT": TXN_DT_MIN, "TransactionAmt": 300.0, "card1": 1, "addr1": 1, "D1": 0}])
    assert add_row_features(round_frame)["amount_is_round"].iloc[0] == 1


def test_calendar_features_are_anchored_to_dataset_start():
    # 1 day and 3 hours after the dataset start.
    frame = _frame([{"TransactionDT": TXN_DT_MIN + 86_400 + 3 * 3_600, "TransactionAmt": 10.0, "card1": 1, "addr1": 1, "D1": 0}])
    out = add_row_features(frame)
    assert out["day_index"].iloc[0] == 1
    assert out["txn_hour"].iloc[0] == 27
    assert out["hour_of_day"].iloc[0] == 3


def test_single_row_calendar_matches_batch():
    """Scoring one record must not shift the calendar (offset anchored to start)."""
    dt = TXN_DT_MIN + 5 * 86_400 + 7 * 3_600
    single = add_row_features(_frame([{"TransactionDT": dt, "TransactionAmt": 10.0, "card1": 1, "addr1": 1, "D1": 0}]))
    batch = add_row_features(
        _frame(
            [
                {"TransactionDT": dt, "TransactionAmt": 10.0, "card1": 1, "addr1": 1, "D1": 0},
                {"TransactionDT": dt + 3_600, "TransactionAmt": 11.0, "card1": 1, "addr1": 1, "D1": 0},
            ]
        )
    )
    for column in ("txn_hour", "day_index", "hour_of_day"):
        assert single[column].iloc[0] == batch[column].iloc[0]


def test_categorical_codes_reserve_zero_for_missing():
    frame = pd.DataFrame({"ProductCD": ["W", "C", None, "W"], "card4": ["visa", None, "visa", "visa"]})
    encoded, mappings = add_category_codes(frame)

    assert encoded["cat_ProductCD"].tolist()[2] == 0
    assert set(encoded["cat_ProductCD"]) - {0} == {mappings["ProductCD"]["W"], mappings["ProductCD"]["C"]}
    assert encoded["cat_ProductCD"].min() == 0

    # Re-applying the stored mapping must be deterministic and mark unseen as 0.
    again, _ = add_category_codes(pd.DataFrame({"ProductCD": ["C", "UNSEEN"], "card4": ["visa", "visa"]}), mappings)
    assert again["cat_ProductCD"].tolist() == [mappings["ProductCD"]["C"], 0]


def test_frequency_encoding_is_train_only():
    train = pd.DataFrame({"card1": [1, 1, 1, 2], "addr1": [10, 10, 10, 10], "P_emaildomain": ["a", "a", "a", "a"], "DeviceInfo": ["x"] * 4})
    maps = compute_frequency_maps(train)
    assert maps["card1_freq"]["1"] == pytest.approx(0.75)

    held_out = pd.DataFrame({"card1": [1, 99], "addr1": [10, 10], "P_emaildomain": ["a", "zzz"], "DeviceInfo": ["x", "x"]})
    encoded = add_frequency_features(held_out, maps)
    assert encoded["card1_freq"].tolist() == [pytest.approx(0.75), 0.0]
    assert encoded["P_emaildomain_freq"].tolist() == [pytest.approx(1.0), 0.0]


def test_feature_matrix_has_every_model_column():
    frame = _frame(
        [
            {"TransactionDT": TXN_DT_MIN + i * 100, "TransactionAmt": 10.0 + i, "card1": 1, "addr1": 1, "D1": 0}
            for i in range(5)
        ]
    )
    frame, _ = add_category_codes(frame)
    frame = add_row_features(frame)
    frame = add_history_features(frame)
    matrix = feature_matrix(frame)

    assert list(matrix.columns) == FEATURE_COLUMNS
    assert matrix.shape == (5, len(FEATURE_COLUMNS))
    assert all(str(dtype).startswith(("float32", "int32")) for dtype in matrix.dtypes)
