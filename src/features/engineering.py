"""Feature engineering for IEEE-CIS.

Design rules
------------
1. Row-level features use only fields of the transaction record itself: amount,
   card, address, distance, calendar, Vesta C/D/V blocks, identity fields.
   All of them are recorded at authorisation time.
2. Behavioural features (card/uid velocity, prior spend) use *strictly earlier*
   transactions. Each group is processed in timestamp order and the current row
   is always subtracted, so no feature can see its own or a future event.
   ``tests/test_features.py`` asserts this by construction and by perturbation.
3. Frequency encodings are fitted on the training period only and applied to
   validation/test; unseen categories map to 0. No target encoding is used
   anywhere, which removes the classic IEEE-CIS leakage route.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    C_COLUMNS,
    CATEGORICAL_CODE_FEATURES,
    CATEGORICAL_RAW,
    D_COLUMNS,
    FEATURE_COLUMNS,
    HIGH_CARD_CATEGORICALS,
    HISTORY_FEATURES,
    ID_NUMERIC,
    M_COLUMNS,
    ROW_FEATURES,
    TXN_DT_MIN,
    V_COLUMNS,
)

IDENTITY_BLOCK = ID_NUMERIC + [c for c in CATEGORICAL_RAW if c.startswith("id_") or c.startswith("Device")]
CORE_BLOCK = [
    c
    for c in ["card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]
    + C_COLUMNS
    + D_COLUMNS
    + V_COLUMNS
]

FREQUENCY_SOURCES = {
    "card1_freq": "card1",
    "addr1_freq": "addr1",
    "P_emaildomain_freq": "P_emaildomain",
    "DeviceInfo_freq": "DeviceInfo",
}


def add_category_codes(
    df: pd.DataFrame, mappings: dict[str, dict[str, int]] | None = None
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Integer-code every categorical column.

    Missing / unseen values get code 0 and real levels start at 1. A dedicated
    non-negative code matters because LightGBM treats *negative* categorical
    values as missing and warns about every one of them.

    Returns the frame and the value->code mapping for each column so the exact
    encoding can be persisted and reused by the scoring endpoint.
    """
    mappings = mappings or {}
    out = df
    for column in CATEGORICAL_RAW:
        if column not in out.columns:
            out[f"cat_{column}"] = np.int32(0)
            continue
        values = out[column].astype("string")
        if column in mappings:
            mapping = mappings[column]
        else:
            categories = pd.Index(values.dropna().unique())
            mapping = {str(value): index + 1 for index, value in enumerate(categories)}
            mappings[column] = mapping
        out[f"cat_{column}"] = values.map(mapping).fillna(0).astype("int32")
    return out, mappings


def timestamp_column(df: pd.DataFrame) -> str:
    """Return the timestamp column name used by the frame.

    The raw frame calls it ``TransactionDT``; the processed frame renames it to
    ``transaction_dt``. Accepting both keeps one engineering implementation for
    dataset building and single-record scoring.
    """
    for candidate in ("transaction_dt", "TransactionDT"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Frame has no TransactionDT / transaction_dt column")


def add_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Amount shape, calendar, match flags and missingness counters."""
    out = df
    amount = out["TransactionAmt"].astype("float64")

    out["log_amount"] = np.log1p(amount).astype("float32")
    cents = np.rint(amount * 100).astype("int64") % 100
    out["amount_cents"] = cents.astype("int16")
    # Round-value bias: fraudsters and card testers often use exact hundreds.
    out["amount_is_round"] = ((amount % 100) == 0).astype("int8")
    out["amount_is_whole"] = (cents == 0).astype("int8")

    dt = out[timestamp_column(out)].astype("int64")
    # Anchored to the dataset start, not to the frame's own minimum, so scoring a
    # single row produces the same calendar features as scoring a batch.
    offset = dt - TXN_DT_MIN
    out["txn_hour"] = (offset // 3600).astype("int32")
    out["day_index"] = (offset // 86400).astype("int16")
    hour = ((dt // 3600) % 24).astype("int8")
    out["hour_of_day"] = hour
    out["day_of_week"] = (out["day_index"] % 7).astype("int8")
    out["is_night"] = ((hour < 6) | (hour >= 22)).astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")

    for column in M_COLUMNS:
        if column in out.columns:
            out[column] = (
                out[column].astype("string").map({"T": 1, "F": 0}).astype("float32")
            )
        else:
            out[column] = np.float32(np.nan)

    if "id_01" in out.columns:
        out["has_identity"] = out["id_01"].notna().astype("int8")
    else:
        out["has_identity"] = np.int8(0)

    # Missingness is informative in this dataset: a missing D7 usually means the
    # card has no history, and a missing identity block means an unidentified
    # device. Counting it gives the trees a direct handle on that.
    identity_present = [c for c in IDENTITY_BLOCK if c in out.columns]
    identity_missing = np.zeros(len(out), dtype="int16")
    for column in identity_present:
        identity_missing += out[column].isna().to_numpy()
    out["identity_missing_count"] = identity_missing.astype("int16")

    core_present = [c for c in CORE_BLOCK if c in out.columns]
    core_missing = np.zeros(len(out), dtype="int16")
    for column in core_present:
        core_missing += out[column].isna().to_numpy()
    out["core_missing_count"] = core_missing.astype("int16")
    return out


def compute_frequency_maps(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Proportion of *training* rows containing each category value.

    Fitted on train only so validation and test cannot influence the encoding.
    """
    maps: dict[str, dict[str, float]] = {}
    for feature, source in FREQUENCY_SOURCES.items():
        if source not in train.columns:
            maps[feature] = {}
            continue
        values = train[source].astype("string")
        counts = values.value_counts(normalize=True, dropna=True)
        maps[feature] = {str(k): float(v) for k, v in counts.items()}
    return maps


def add_frequency_features(df: pd.DataFrame, maps: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = df
    for feature, source in FREQUENCY_SOURCES.items():
        mapping = maps.get(feature, {})
        if source in out.columns and mapping:
            out[feature] = (
                out[source].astype("string").map(mapping).fillna(0.0).astype("float32")
            )
        else:
            out[feature] = np.float32(0.0)
    return out


def _group_prior_stats(keys: np.ndarray, times: np.ndarray, values: np.ndarray):
    """Prior count / mean / std for each row within its group.

    Rows are ordered by (key, time) and aggregated cumulatively; the current row
    is subtracted, so every statistic describes strictly earlier transactions.
    """
    order = np.lexsort((times, keys))
    k = keys[order]
    v = values[order]

    n = len(k)
    counts = np.zeros(n, dtype="float32")
    means = np.full(n, np.nan, dtype="float32")
    stds = np.full(n, np.nan, dtype="float32")

    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True])
    for start, end in zip(starts[:-1], starts[1:]):
        group_values = v[start:end]
        prior_count = np.arange(len(group_values), dtype="float32")
        cumsum = np.cumsum(group_values)
        prior_sum = cumsum - group_values
        prior_sq_sum = np.cumsum(group_values * group_values) - group_values**2

        counts[start:end] = prior_count
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(prior_count > 0, prior_sum / np.maximum(prior_count, 1), np.nan)
            var = np.where(
                prior_count > 0,
                prior_sq_sum / np.maximum(prior_count, 1) - mean**2,
                np.nan,
            )
        means[start:end] = mean
        stds[start:end] = np.sqrt(np.clip(var, 0, None))

    # Scatter back to the original row order.
    result_counts = np.empty(n, dtype="float32")
    result_means = np.empty(n, dtype="float32")
    result_stds = np.empty(n, dtype="float32")
    result_counts[order] = counts
    result_means[order] = means
    result_stds[order] = stds
    return result_counts, result_means, result_stds


def _seconds_since_previous(keys: np.ndarray, times: np.ndarray) -> np.ndarray:
    order = np.lexsort((times, keys))
    k = keys[order]
    t = times[order]
    delta = np.full(len(k), np.nan, dtype="float32")
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True])
    for start, end in zip(starts[:-1], starts[1:]):
        group_times = t[start:end].astype("float64")
        delta[start + 1 : end] = group_times[1:] - group_times[:-1]
    result = np.empty(len(k), dtype="float32")
    result[order] = delta
    return result


def _window_counts(keys: np.ndarray, times: np.ndarray, windows: dict[str, int]) -> dict[str, np.ndarray]:
    """Count of strictly earlier events of the same key inside a time window."""
    order = np.lexsort((times, keys))
    k = keys[order]
    t = times[order].astype("int64")
    out = {name: np.zeros(len(k), dtype="float32") for name in windows}

    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1], True])
    for start, end in zip(starts[:-1], starts[1:]):
        group_times = t[start:end]
        if len(group_times) < 2:
            continue
        # For row i, count rows j < i with t[i] - t[j] <= window.
        left = {
            name: np.searchsorted(group_times, group_times[1:] - seconds, side="left")
            for name, seconds in windows.items()
        }
        positions = np.arange(1, len(group_times))
        for name in windows:
            out[name][start + 1 : end] = (positions - left[name]).astype("float32")

    result = {name: np.empty(len(k), dtype="float32") for name in windows}
    for name in windows:
        result[name][order] = out[name]
    return result


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only behavioural features for the card and the card+address uid."""
    out = df
    times = out[timestamp_column(out)].to_numpy(dtype="int64")
    amount = out["TransactionAmt"].astype("float64").to_numpy()

    card = out["card1"].fillna(-1).to_numpy(dtype="int64")
    counts, means, stds = _group_prior_stats(card, times, amount)
    out["card1_prior_count"] = counts
    out["card1_prior_amt_mean"] = means
    out["card1_prior_amt_std"] = stds

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(means > 0, amount / means, 1.0)
    ratio = np.clip(ratio, 0, 1e6)
    ratio[counts == 0] = 1.0
    out["amount_vs_card1_mean"] = ratio.astype("float32")

    out["card1_secs_since_prev"] = _seconds_since_previous(card, times)

    windows = _window_counts(card, times, {"24h": 86_400, "7d": 7 * 86_400})
    out["card1_prior_count_24h"] = windows["24h"]
    out["card1_prior_count_7d"] = windows["7d"]

    # uid: the same card used from the same billing region on the same day-offset.
    # It is the closest proxy this dataset offers for "same customer account".
    uid = (
        out["card1"].fillna(-1).astype("int64").astype(str)
        + "_"
        + out["addr1"].fillna(-1).astype("int64").astype(str)
        + "_"
        + out["D1"].fillna(-1).round().astype("int64").astype(str)
    )
    uid_codes, _ = pd.factorize(uid, sort=False)
    uid_counts, uid_means, _ = _group_prior_stats(uid_codes, times, amount)
    out["uid_prior_count"] = uid_counts
    out["uid_prior_amt_mean"] = uid_means
    with np.errstate(invalid="ignore", divide="ignore"):
        uid_ratio = np.where(uid_means > 0, amount / uid_means, 1.0)
    uid_ratio = np.clip(uid_ratio, 0, 1e6)
    uid_ratio[uid_counts == 0] = 1.0
    out["amount_vs_uid_mean"] = uid_ratio.astype("float32")
    return out


def feature_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return the model matrix with a stable order and dtypes.

    Numeric columns are float32; categorical code columns stay int32 so LightGBM
    can treat them as categorical. Missing values are preserved (NaN / -1) on
    purpose: absence is meaningful in this dataset and tree models handle it
    natively.
    """
    columns = columns or FEATURE_COLUMNS
    categorical = set(CATEGORICAL_CODE_FEATURES)
    data = {}
    for c in columns:
        if c in df.columns:
            data[c] = df[c]
        elif c in categorical:
            data[c] = np.int32(0)
        else:
            data[c] = np.float32(np.nan)

    matrix = pd.DataFrame(data, index=df.index)
    for c in columns:
        if c in categorical:
            matrix[c] = matrix[c].fillna(0).astype("int32")
        else:
            matrix[c] = matrix[c].astype("float32")
    return matrix[columns]


def categorical_feature_names(columns: list[str] | None = None) -> list[str]:
    columns = columns or FEATURE_COLUMNS
    return [c for c in columns if c in CATEGORICAL_CODE_FEATURES]


FEATURE_BLOCKS = [
    {
        "block": "Amount shape",
        "features": ["TransactionAmt", "log_amount", "amount_cents", "amount_is_round", "amount_is_whole"],
        "reasoning": (
            "Amount is the strongest single behavioural field. The log removes the "
            "heavy right tail; the cents and round-hundred flags capture card-testing "
            "patterns, where an attacker probes with exact round values."
        ),
        "available_at_prediction": True,
        "leakage_risk": "None",
    },
    {
        "block": "Calendar",
        "features": ["txn_hour", "hour_of_day", "day_of_week", "day_index", "is_night", "is_weekend"],
        "reasoning": (
            "Derived from TransactionDT. Fraud has a diurnal profile: card-not-present "
            "fraud clusters in low-monitoring hours. day_index also lets the tree model "
            "track slow drift across the 182 days."
        ),
        "available_at_prediction": True,
        "leakage_risk": "None (timestamp only, never the label)",
    },
    {
        "block": "Card, address and distance",
        "features": ["card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2", *CATEGORICAL_CODE_FEATURES[:6]],
        "reasoning": (
            "card1 is an anonymised issuer/card identifier and is the main grouping key "
            "for history. card4/card6 give brand and debit/credit, which carry different "
            "fraud rates. dist1 is the distance between the buyer and the merchant."
        ),
        "available_at_prediction": True,
        "leakage_risk": "None; identifiers, not outcomes",
    },
    {
        "block": "Identity match flags",
        "features": M_COLUMNS,
        "reasoning": (
            "M1-M9 encode whether the name, address, phone and other fields matched the "
            "cardholder record. A failed match is a direct authorisation signal."
        ),
        "available_at_prediction": True,
        "leakage_risk": "None",
    },
    {
        "block": "Identity presence / completeness",
        "features": ["has_identity", "identity_missing_count"],
        "reasoning": (
            "Only 24.4% of rows have identity data. Presence is itself informative: "
            "identified devices transact at a 7.85% fraud rate versus 2.09% for "
            "unidentified ones, because card-not-present and new-device traffic is "
            "where the fraud concentrates."
        ),
        "available_at_prediction": True,
        "leakage_risk": (
            "Not leakage, but a distribution artefact of this dataset: the identity "
            "block was only collected for part of the traffic, so the flag partly "
            "encodes the data-collection process."
        ),
    },
    {
        "block": "Vesta C / D / V blocks",
        "features": ["C1-C14", "D1-D15", "V1-V339"],
        "reasoning": (
            "C columns count how many cards/addresses/devices are tied to an entity; "
            "D columns are day-deltas since a previous event (D1 = days since the "
            "card's first transaction); V columns are Vesta's own engineered "
            "aggregations. They are used as provided; their exact semantics are "
            "undisclosed, which caps interpretability."
        ),
        "available_at_prediction": True,
        "leakage_risk": (
            "None in principle, but Vesta computed them over the full competition "
            "window, so a few may embed mild future information. This is a property "
            "of the released dataset, not something this pipeline introduces."
        ),
    },
    {
        "block": "Identity / device fields",
        "features": ["id_01-id_11, id_13, id_14, id_17-id_26, id_32", "cat_id_12 ... cat_DeviceInfo"],
        "reasoning": (
            "Network, device, OS, browser and screen-resolution fields. Shared devices "
            "and unusual browser/OS combinations are classic account-takeover signals."
        ),
        "available_at_prediction": True,
        "leakage_risk": "None",
    },
    {
        "block": "Frequency encodings",
        "features": list(FREQUENCY_SOURCES),
        "reasoning": (
            "How common this card / address / email domain / device is in the training "
            "period. Rare values behave differently from popular ones. Fitted on train "
            "only; unseen values map to 0."
        ),
        "available_at_prediction": True,
        "leakage_risk": "Train-only fit prevents validation/test information leaking in",
    },
    {
        "block": "Past-only history",
        "features": HISTORY_FEATURES,
        "reasoning": (
            "Per card and per card+address uid: number of earlier transactions, prior "
            "mean/std spend, the current amount relative to that mean, seconds since the "
            "previous transaction, and 24h/7d velocity counts. Sudden spend deviation "
            "and burst activity are the strongest non-Vesta signals available."
        ),
        "available_at_prediction": True,
        "leakage_risk": "None: strictly earlier rows only, verified by tests",
    },
    {
        "block": "Missingness counters",
        "features": ["core_missing_count"],
        "reasoning": (
            "Number of missing core (C/D/V/card) fields. A card with little history "
            "produces many missing V/D values, so this is a compact proxy for "
            "'how much is known about this card'."
        ),
        "available_at_prediction": True,
        "leakage_risk": "None",
    },
]


def feature_dictionary() -> dict:
    return {
        "n_features": len(FEATURE_COLUMNS),
        "n_numeric": len(ROW_FEATURES) + len(HISTORY_FEATURES),
        "n_categorical": len(CATEGORICAL_CODE_FEATURES),
        "columns": FEATURE_COLUMNS,
        "blocks": FEATURE_BLOCKS,
        "frequency_sources": FREQUENCY_SOURCES,
        "categorical_columns": CATEGORICAL_CODE_FEATURES,
        "high_cardinality_categoricals": HIGH_CARD_CATEGORICALS,
    }


__all__ = [
    "CATEGORICAL_CODE_FEATURES",
    "FEATURE_BLOCKS",
    "add_category_codes",
    "add_frequency_features",
    "add_history_features",
    "add_row_features",
    "categorical_feature_names",
    "compute_frequency_maps",
    "feature_dictionary",
    "feature_matrix",
]
