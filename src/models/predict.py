"""Model loading and transaction scoring.

The scorer is the single entry point used by the dashboard, the tests and the
batch helper, so there is exactly one implementation of "how a transaction
becomes a risk score".

Scoring a transaction with partial information
----------------------------------------------
At authorisation time the reviewer knows the amount, the product code, the card
attributes, the calendar position and (sometimes) device/identity data. The
Vesta C/D/V blocks are computed by the processor and are not known to the
reviewer, so those inputs default to their training medians. Every defaulted
input is disclosed by the endpoint, and the resulting score is a
partial-information estimate, not a substitute for a full production feature
vector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_ENCODER_PATH,
    HIGH_RISK_ACTION,
    HISTORY_DEFAULTS,
    LOW_RISK_ACTION,
    MEDIUM_RISK_ACTION,
    MODEL_PATH,
    RISK_BAND_LABELS,
    TXN_DT_MIN,
)
from src.evaluation.metrics import assign_risk_band
from src.features.engineering import (
    add_category_codes,
    add_frequency_features,
    add_row_features,
    feature_matrix,
)
from src.preprocessing.validation import records_to_frame, validate_transaction

RAW_OVERRIDES = {
    "amount": "TransactionAmt",
    "card1": "card1",
    "card2": "card2",
    "card3": "card3",
    "card5": "card5",
    "addr1": "addr1",
    "addr2": "addr2",
    "dist1": "dist1",
    "dist2": "dist2",
}
CATEGORICAL_OVERRIDES = {
    "product_cd": "ProductCD",
    "card4": "card4",
    "card6": "card6",
    "p_emaildomain": "P_emaildomain",
    "device_type": "DeviceType",
    "device_info": "DeviceInfo",
}


@dataclass
class FraudScorer:
    """Fitted model plus the thresholds, encoders and defaults it was built with."""

    model: object
    feature_columns: list[str]
    thresholds: dict
    operating_threshold: float
    category_mappings: dict = field(default_factory=dict)
    frequency_maps: dict = field(default_factory=dict)
    raw_defaults: dict = field(default_factory=dict)
    feature_defaults: dict = field(default_factory=dict)
    model_name: str = "unknown"
    cost_per_investigation: float = 4.0

    @classmethod
    def load(
        cls,
        path: str | Path = MODEL_PATH,
        encoder_path: str | Path = FEATURE_ENCODER_PATH,
    ) -> "FraudScorer":
        bundle = joblib.load(Path(path))
        encoders = {}
        encoder_path = Path(encoder_path)
        if encoder_path.exists():
            encoders = json.loads(encoder_path.read_text(encoding="utf-8"))
        return cls(
            model=bundle["model"],
            feature_columns=list(bundle["feature_columns"]),
            thresholds=dict(bundle["thresholds"]),
            operating_threshold=float(bundle["operating_threshold"]),
            category_mappings=encoders.get("category_mappings", {}),
            frequency_maps=encoders.get("frequency_maps", {}),
            raw_defaults=dict(bundle.get("raw_defaults", {})),
            feature_defaults=dict(bundle.get("feature_defaults", {})),
            model_name=str(bundle.get("model_name", "unknown")),
            cost_per_investigation=float(bundle.get("cost_per_investigation", 4.0)),
        )

    # --------------------------------------------------------------- engineering
    def prepare_features(self, records: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Turn validated raw records into the exact model matrix.

        Returns the engineered frame and the list of inputs that fell back to
        training medians, so the UI can disclose them.
        """
        frame = records.reset_index(drop=True)
        raw = pd.DataFrame(
            {
                column: np.full(len(frame), value, dtype="float32")
                for column, value in self.raw_defaults.items()
            }
        )

        raw["TransactionDT"] = (
            TXN_DT_MIN + frame["txn_hour"].to_numpy(dtype="int64") * 3600
        ).astype("int64")
        raw["TransactionAmt"] = frame["amount"].to_numpy(dtype="float64")

        defaulted: list[str] = []
        for source, target in RAW_OVERRIDES.items():
            if source == "amount":
                continue
            if source in frame.columns:
                values = pd.to_numeric(frame[source], errors="coerce")
                provided = values.notna().to_numpy()
                raw[target] = np.where(provided, values.to_numpy(dtype="float32"), raw[target])
                if not provided.all():
                    defaulted.append(target)
            else:
                defaulted.append(target)

        for source, target in CATEGORICAL_OVERRIDES.items():
            values = frame[source].astype("string") if source in frame.columns else pd.Series(
                [pd.NA] * len(frame), dtype="string"
            )
            raw[target] = values
            if values.isna().any():
                defaulted.append(target)

        raw, _ = add_category_codes(raw, self.category_mappings)
        raw = add_row_features(raw)
        raw = add_frequency_features(raw, self.frequency_maps)

        for column, value in HISTORY_DEFAULTS.items():
            raw[column] = np.float32(value)

        # The dashboard lets the user declare whether identity data exists; that
        # override must win over "did we fill a median into id_01".
        if "has_identity" in frame.columns:
            raw["has_identity"] = frame["has_identity"].to_numpy(dtype="int8")

        for column in self.feature_columns:
            if column not in raw.columns:
                raw[column] = self.feature_defaults.get(column, np.nan)
            elif column in self.feature_defaults:
                raw[column] = raw[column].fillna(self.feature_defaults[column])

        return raw[self.feature_columns], defaulted

    # ------------------------------------------------------------------ scoring
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = self.model.predict_proba(X)[:, 1]
        return np.asarray(probabilities, dtype="float64")

    def score_features(self, engineered: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(feature_matrix(engineered, self.feature_columns))

    def score_transaction(self, record: dict) -> dict:
        """Validate, engineer and score one transaction."""
        clean = validate_transaction(record)
        frame = records_to_frame([clean])
        engineered, defaulted = self.prepare_features(frame)
        probability = float(self.predict_proba(engineered)[0])
        result = self.decision(probability, warning=clean.get("warning"))
        result["engineered_features"] = engineered
        result["defaulted_inputs"] = sorted(set(defaulted))
        return result

    def score_batch(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Score a validated batch; every row goes through the same pipeline."""
        engineered, defaulted = self.prepare_features(frame)
        probabilities = self.predict_proba(engineered)
        out = frame.copy()
        out["fraud_probability"] = probabilities
        out["risk_band"] = assign_risk_band(
            probabilities, self.thresholds["high_threshold"], self.thresholds["medium_threshold"]
        )
        out["action"] = [self.action_for_band(b) for b in out["risk_band"]]
        out["alert"] = probabilities >= self.operating_threshold
        return out, sorted(set(defaulted))

    # ------------------------------------------------------------------- policy
    def decision(self, probability: float, warning: str | None = None) -> dict:
        band = assign_risk_band(
            [probability], self.thresholds["high_threshold"], self.thresholds["medium_threshold"]
        )[0]
        return {
            "fraud_probability": float(probability),
            "risk_band": band,
            "action": self.action_for_band(band),
            "operating_threshold": self.operating_threshold,
            "alert": bool(probability >= self.operating_threshold),
            "thresholds": self.thresholds,
            "model_name": self.model_name,
            "warning": warning,
        }

    @staticmethod
    def action_for_band(band: str) -> str:
        return {
            "High Risk": HIGH_RISK_ACTION,
            "Medium Risk": MEDIUM_RISK_ACTION,
            "Low Risk": LOW_RISK_ACTION,
        }.get(band, "Review")


def load_thresholds(path: str | Path) -> dict:
    with open(Path(path), encoding="utf-8") as fh:
        return json.load(fh)


__all__ = ["FraudScorer", "RISK_BAND_LABELS", "load_thresholds"]
