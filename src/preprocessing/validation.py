"""Input validation for the interactive scoring endpoint.

The dashboard accepts the handful of fields a reviewer could actually know at
authorisation time. Everything else the model consumes is filled from the
training-set defaults, and the UI says so explicitly. Bad input is rejected with
a readable list of field-level errors instead of silently scoring garbage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import MAX_TXN_HOUR

VALID_PRODUCT_CODES = ("W", "C", "R", "H", "S")
VALID_CARD_BRANDS = ("visa", "mastercard", "american express", "discover")
VALID_CARD_TYPES = ("debit", "credit", "charge card", "debit or credit")
VALID_DEVICE_TYPES = ("desktop", "mobile")

REQUIRED_FIELDS = ("amount", "product_cd", "txn_hour")
UPLOAD_REQUIRED_COLUMNS = list(REQUIRED_FIELDS)
UPLOAD_OPTIONAL_COLUMNS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "p_emaildomain",
    "device_type", "device_info", "has_identity",
]

MAX_AMOUNT = 100_000.0
MAX_ADDRESS_CODE = 1_000_000


class InputValidationError(ValueError):
    """Raised when a transaction record fails validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _number(
    value: Any,
    field: str,
    errors: list[str],
    *,
    integer: bool = False,
    maximum: float | None = None,
    minimum: float = 0.0,
):
    if value is None or (isinstance(value, str) and not value.strip()):
        errors.append(f"{field} is required")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be numeric, got {value!r}")
        return None
    if not math.isfinite(number):
        errors.append(f"{field} must be finite")
        return None
    if number < minimum:
        errors.append(f"{field} must be at least {minimum:g} (got {number})")
        return None
    if maximum is not None and number > maximum:
        errors.append(f"{field} exceeds the plausible maximum ({maximum:,.0f})")
        return None
    if integer:
        if number != int(number):
            errors.append(f"{field} must be a whole number (got {number})")
            return None
        return int(number)
    return float(number)


def _optional_number(
    value: Any, field: str, errors: list[str], *, integer: bool = False,
    maximum: float | None = None, minimum: float = 0.0,
):
    if value is None or (isinstance(value, float) and math.isnan(value)) or (
        isinstance(value, str) and not value.strip()
    ):
        return None
    return _number(value, field, errors, integer=integer, maximum=maximum, minimum=minimum)


def validate_transaction(record: dict) -> dict:
    """Validate and normalise a single transaction record."""
    if not isinstance(record, dict):
        raise InputValidationError(["Transaction must be provided as a key/value record"])

    errors: list[str] = []
    warnings: list[str] = []
    clean: dict = {}

    missing = [f for f in REQUIRED_FIELDS if record.get(f) in (None, "")]
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")

    clean["amount"] = _number(record.get("amount"), "amount", errors, maximum=MAX_AMOUNT, minimum=0.01)
    clean["txn_hour"] = _number(
        record.get("txn_hour"), "txn_hour", errors, integer=True, maximum=MAX_TXN_HOUR
    )

    product = str(record.get("product_cd", "")).strip().upper()
    if product not in VALID_PRODUCT_CODES:
        errors.append(
            f"product_cd must be one of {', '.join(VALID_PRODUCT_CODES)} (got {record.get('product_cd')!r})"
        )
    clean["product_cd"] = product

    brand = str(record.get("card4") or "").strip().lower()
    if brand and brand not in VALID_CARD_BRANDS:
        warnings.append(f"Unrecognised card brand {brand!r}; treated as unknown")
        brand = ""
    clean["card4"] = brand or None

    card_type = str(record.get("card6") or "").strip().lower()
    if card_type and card_type not in VALID_CARD_TYPES:
        warnings.append(f"Unrecognised card type {card_type!r}; treated as unknown")
        card_type = ""
    clean["card6"] = card_type or None

    for field in ("card1", "card2", "card3", "card5", "addr1", "addr2"):
        clean[field] = _optional_number(
            record.get(field), field, errors, integer=True, maximum=MAX_ADDRESS_CODE
        )

    for field in ("dist1", "dist2"):
        clean[field] = _optional_number(record.get(field), field, errors, maximum=10_000)

    email = str(record.get("p_emaildomain") or "").strip().lower()
    if len(email) > 100:
        errors.append("p_emaildomain is unreasonably long")
    clean["p_emaildomain"] = email or None

    device_type = str(record.get("device_type") or "").strip().lower()
    if device_type and device_type not in VALID_DEVICE_TYPES:
        warnings.append(f"Unrecognised device_type {device_type!r}; treated as unknown")
        device_type = ""
    clean["device_type"] = device_type or None

    device_info = str(record.get("device_info") or "").strip().lower()
    if len(device_info) > 100:
        errors.append("device_info is unreasonably long")
    clean["device_info"] = device_info or None

    has_identity = record.get("has_identity", None)
    if has_identity is None:
        # Assume identity data exists if the caller supplied device fields.
        has_identity = bool(device_type or device_info or email)
    clean["has_identity"] = int(bool(has_identity))

    if errors:
        raise InputValidationError(errors)

    if clean["amount"] < 1.0:
        warnings.append("Sub-dollar amount: typical of card-testing probes")
    if clean["amount"] >= 1000:
        warnings.append("High-value transaction: review capacity is consumed faster")
    clean["warning"] = "; ".join(warnings) if warnings else None
    return clean


def validate_upload(df: pd.DataFrame, max_rows: int = 5000) -> tuple[pd.DataFrame, list[str]]:
    """Validate a batch CSV upload, dropping invalid rows."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise InputValidationError(["The uploaded file contains no rows"])

    missing = [c for c in UPLOAD_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InputValidationError([f"Upload is missing required columns: {', '.join(missing)}"])
    if len(df) > max_rows:
        raise InputValidationError([f"Upload has {len(df):,} rows; the limit is {max_rows:,}"])

    clean_rows: list[dict] = []
    errors: list[str] = []
    for position, row in df.iterrows():
        try:
            clean_rows.append(validate_transaction(row.to_dict()))
        except InputValidationError as exc:
            errors.append(f"Row {position}: {'; '.join(exc.errors)}")
    if not clean_rows:
        raise InputValidationError(errors or ["No valid rows found"])
    return pd.DataFrame(clean_rows), errors


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Turn validated records into a raw-shaped frame for feature engineering."""
    columns = [
        "amount", "txn_hour", "product_cd", "card1", "card2", "card3", "card4",
        "card5", "card6", "addr1", "addr2", "dist1", "dist2", "p_emaildomain",
        "device_type", "device_info", "has_identity",
    ]
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


__all__ = [
    "InputValidationError",
    "REQUIRED_FIELDS",
    "UPLOAD_OPTIONAL_COLUMNS",
    "UPLOAD_REQUIRED_COLUMNS",
    "VALID_CARD_BRANDS",
    "VALID_CARD_TYPES",
    "VALID_PRODUCT_CODES",
    "records_to_frame",
    "validate_transaction",
    "validate_upload",
]
