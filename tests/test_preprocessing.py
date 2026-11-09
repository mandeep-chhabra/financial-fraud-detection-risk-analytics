"""Temporal splitting and sampling tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TRAIN_END_DT, TXN_DT_MIN, VAL_END_DT
from src.preprocessing.splits import (
    assert_temporal_order,
    capped_negative_sample,
    stratified_subsample,
)
from src.data.load import assign_split


def test_assign_split_boundaries_are_forward_only():
    frame = pd.DataFrame(
        {
            "TransactionDT": [
                TXN_DT_MIN,
                TRAIN_END_DT,
                TRAIN_END_DT + 1,
                VAL_END_DT,
                VAL_END_DT + 1,
            ]
        }
    )
    labels = assign_split(frame).tolist()
    assert labels == ["train", "train", "validation", "validation", "test"]


def test_assert_temporal_order_accepts_disjoint_windows():
    train = pd.DataFrame({"TransactionDT": [TXN_DT_MIN, TRAIN_END_DT]})
    validation = pd.DataFrame({"TransactionDT": [TRAIN_END_DT + 1, VAL_END_DT]})
    test = pd.DataFrame({"TransactionDT": [VAL_END_DT + 1, VAL_END_DT + 1_000]})
    assert_temporal_order(train, validation, test)


def test_assert_temporal_order_rejects_overlap():
    train = pd.DataFrame({"TransactionDT": [TXN_DT_MIN, TRAIN_END_DT + 10]})
    validation = pd.DataFrame({"TransactionDT": [TRAIN_END_DT + 1, VAL_END_DT]})
    test = pd.DataFrame({"TransactionDT": [VAL_END_DT + 1, VAL_END_DT + 1_000]})
    with pytest.raises(AssertionError):
        assert_temporal_order(train, validation, test)


def test_stratified_subsample_preserves_prevalence():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"isFraud": (rng.random(10_000) < 0.05).astype(int)})
    sample = stratified_subsample(frame, 2_000)
    assert len(sample) == 2_000
    assert abs(sample["isFraud"].mean() - frame["isFraud"].mean()) < 0.01


def test_stratified_subsample_returns_everything_when_n_is_large():
    frame = pd.DataFrame({"isFraud": [0, 1, 0, 1]})
    assert len(stratified_subsample(frame, 100)) == 4


def test_capped_negative_sample_keeps_all_frauds():
    frame = pd.DataFrame({"isFraud": [1] * 30 + [0] * 5_000})
    sample = capped_negative_sample(frame, max_legit=500)
    assert int(sample["isFraud"].sum()) == 30
    assert len(sample) == 530
