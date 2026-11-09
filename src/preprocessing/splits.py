"""Reproducible train / validation / test preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET, TRAIN_END_DT, VAL_END_DT
from src.features.engineering import feature_matrix


def split_xy(df: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the model matrix and the binary label for a split."""
    X = feature_matrix(df, columns=columns)
    y = df[TARGET].to_numpy(dtype=np.int8)
    return X, y


def assert_temporal_order(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if the temporal boundaries are violated.

    Overlap would mean the model was evaluated on rows it could have memorised,
    which silently inflates every reported metric.
    """
    bounds = {
        "train": (int(train["TransactionDT"].min()), int(train["TransactionDT"].max())),
        "validation": (int(validation["TransactionDT"].min()), int(validation["TransactionDT"].max())),
        "test": (int(test["TransactionDT"].min()), int(test["TransactionDT"].max())),
    }
    if bounds["train"][1] > TRAIN_END_DT:
        raise AssertionError(f"Train exceeds the split boundary: {bounds}")
    if bounds["validation"][0] <= TRAIN_END_DT or bounds["validation"][1] > VAL_END_DT:
        raise AssertionError(f"Validation outside its window: {bounds}")
    if bounds["test"][0] <= VAL_END_DT:
        raise AssertionError(f"Test starts before validation ends: {bounds}")


def stratified_subsample(
    df: pd.DataFrame, n: int, seed: int = 42, label_col: str = TARGET
) -> pd.DataFrame:
    """Sample ``n`` rows preserving the fraud rate.

    Applied to the *training* split only; validation and test are always scored
    in full.
    """
    if n >= len(df):
        return df
    fraud = df[df[label_col] == 1]
    legit = df[df[label_col] == 0]
    rate = len(fraud) / len(df)
    n_fraud = max(int(round(n * rate)), 1)
    n_legit = n - n_fraud
    sampled = pd.concat(
        [
            fraud.sample(n=min(n_fraud, len(fraud)), random_state=seed),
            legit.sample(n=min(n_legit, len(legit)), random_state=seed),
        ]
    )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def capped_negative_sample(
    df: pd.DataFrame, max_legit: int, seed: int = 42, label_col: str = TARGET
) -> pd.DataFrame:
    """Keep all frauds and at most ``max_legit`` legitimate rows (train only)."""
    fraud = df[df[label_col] == 1]
    legit = df[df[label_col] == 0]
    if len(legit) > max_legit:
        legit = legit.sample(n=max_legit, random_state=seed)
    return pd.concat([fraud, legit]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


__all__ = [
    "FEATURE_COLUMNS",
    "assert_temporal_order",
    "capped_negative_sample",
    "split_xy",
    "stratified_subsample",
]
