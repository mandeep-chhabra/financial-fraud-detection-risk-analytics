"""Build the SQLite analytics database.

Schema
------
transactions        590,540 IEEE-CIS rows + score / risk band for scored splits
model_metrics       held-out metrics of the selected model
threshold_analysis  validation and test precision/recall/FPR sweeps
model_metadata      key/value facts used by the dashboard

All rows are loaded so SQL analytics describe the whole labelled dataset, while
prediction columns are populated only for the validation and test periods.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    DB_PATH,
    PREDICTIONS_PARQUET,
    PROCESSED_PARQUET,
    TEST_METRICS_JSON,
    THRESHOLD_ANALYSIS_CSV,
)
from src.utils import get_logger

logger = get_logger("database")

SCHEMA = """
DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    txn_id            INTEGER PRIMARY KEY,
    transaction_dt    INTEGER NOT NULL,
    txn_hour          INTEGER NOT NULL,
    day_index         INTEGER NOT NULL,
    product_cd        TEXT    NOT NULL,
    card4             TEXT,
    card6             TEXT,
    card1             INTEGER,
    addr1             INTEGER,
    p_emaildomain     TEXT,
    has_identity      INTEGER NOT NULL,
    amount            REAL    NOT NULL,
    is_fraud          INTEGER NOT NULL,
    split             TEXT    NOT NULL,
    fraud_probability REAL,
    risk_band         TEXT,
    alerted           INTEGER
);

DROP TABLE IF EXISTS scored_staging;
CREATE TABLE scored_staging (
    txn_id            INTEGER PRIMARY KEY,
    fraud_probability REAL,
    risk_band         TEXT,
    alerted           INTEGER
);

DROP TABLE IF EXISTS threshold_analysis;
CREATE TABLE threshold_analysis (
    threshold REAL, tp INTEGER, fp INTEGER, fn INTEGER, tn INTEGER,
    alerts INTEGER, precision REAL, recall REAL, fpr REAL, fnr REAL,
    alert_rate REAL, f1 REAL, fraud_value_captured REAL,
    fraud_value_captured_pct REAL, investigation_cost REAL, net_benefit REAL
);

DROP TABLE IF EXISTS model_metrics;
CREATE TABLE model_metrics (
    metric TEXT, value REAL, eval_split TEXT
);

DROP TABLE IF EXISTS model_metadata;
CREATE TABLE model_metadata (
    key TEXT PRIMARY KEY, value TEXT
);
"""

TRANSACTION_COLUMNS = [
    "txn_id", "transaction_dt", "txn_hour", "day_index", "product_cd", "card4",
    "card6", "card1", "addr1", "p_emaildomain", "has_identity", "amount",
    "isFraud", "split",
]

INSERT_SQL = """
INSERT INTO transactions (
    txn_id, transaction_dt, txn_hour, day_index, product_cd, card4, card6,
    card1, addr1, p_emaildomain, has_identity, amount, is_fraud, split
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _insert_transactions(conn: sqlite3.Connection, features_path: Path) -> int:
    parquet = pq.ParquetFile(features_path)
    total = 0
    for batch in parquet.iter_batches(batch_size=100_000, columns=TRANSACTION_COLUMNS):
        frame = batch.to_pandas()
        for column in ("product_cd", "card4", "card6", "p_emaildomain", "split"):
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
        conn.executemany(
            INSERT_SQL,
            frame[TRANSACTION_COLUMNS].itertuples(index=False, name=None),
        )
        total += len(frame)
    return total


def _insert_predictions(conn: sqlite3.Connection, predictions_path: Path) -> int:
    predictions = pd.read_parquet(
        predictions_path, columns=["txn_id", "fraud_probability", "risk_band", "alerted"]
    )
    conn.executemany(
        "INSERT INTO scored_staging (txn_id, fraud_probability, risk_band, alerted) VALUES (?, ?, ?, ?)",
        predictions.itertuples(index=False, name=None),
    )
    conn.execute(
        """
        UPDATE transactions
        SET fraud_probability = s.fraud_probability,
            risk_band = s.risk_band,
            alerted = s.alerted
        FROM scored_staging AS s
        WHERE transactions.txn_id = s.txn_id
        """
    )
    return len(predictions)


def _insert_threshold_analysis(conn: sqlite3.Connection, csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    sweep = pd.read_csv(csv_path)
    columns = [
        "threshold", "tp", "fp", "fn", "tn", "alerts", "precision", "recall",
        "fpr", "fnr", "alert_rate", "f1", "fraud_value_captured",
        "fraud_value_captured_pct", "investigation_cost", "net_benefit",
    ]
    for column in columns:
        if column not in sweep.columns:
            sweep[column] = None
    conn.executemany(
        f"INSERT INTO threshold_analysis ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
        sweep[columns].itertuples(index=False, name=None),
    )
    return len(sweep)


def _insert_metrics(conn: sqlite3.Connection, metrics_path: Path) -> None:
    if not metrics_path.exists():
        return
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

    rows = []
    for split_name, payload in (
        ("test", metrics["test_at_operating_threshold"]),
        ("validation", metrics["validation"]["at_operating_threshold"]),
    ):
        rows += [
            (key, float(value), split_name)
            for key, value in payload.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
    # Ratio metrics that are not part of the threshold snapshot.
    for key in ("pr_auc", "roc_auc"):
        rows.append((key, float(metrics["validation"][key]), "validation"))
        rows.append((key, float(metrics["test_at_operating_threshold"][key]), "test"))
    rows.append(("recall_at_1pct_fpr", float(metrics["test_recall_at_1pct_fpr"]["recall"]), "test"))
    conn.executemany("INSERT INTO model_metrics (metric, value, eval_split) VALUES (?, ?, ?)", rows)

    metadata = {
        "model": metrics["model"],
        "operating_threshold": metrics["operating_threshold"],
        "operating_policy": metrics["operating_policy"],
        "high_risk_threshold": metrics["risk_bands"]["high_threshold"],
        "medium_risk_threshold": metrics["risk_bands"]["medium_threshold"],
        "risk_band_rule": metrics["risk_bands"]["rule"],
        "train_rows": metrics["splits"]["train_rows"],
        "validation_rows": metrics["splits"]["validation_rows"],
        "test_rows": metrics["splits"]["test_rows"],
        "interpretation": metrics["interpretation"],
    }
    conn.executemany(
        "INSERT OR REPLACE INTO model_metadata (key, value) VALUES (?, ?)",
        [(k, str(v)) for k, v in metadata.items()],
    )


def build_database(
    features_path: str | Path = PROCESSED_PARQUET,
    predictions_path: str | Path = PREDICTIONS_PARQUET,
    db_path: str | Path = DB_PATH,
    metrics_path: str | Path = TEST_METRICS_JSON,
    threshold_csv: str | Path = THRESHOLD_ANALYSIS_CSV,
) -> Path:
    """Create (or rebuild) the SQLite database used by the dashboard."""
    features_path, predictions_path = Path(features_path), Path(predictions_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    started = time.time()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.executescript(SCHEMA)

        logger.info("Inserting transactions from %s", features_path)
        rows = _insert_transactions(conn, features_path)
        logger.info("  %s rows inserted", f"{rows:,}")

        logger.info("Applying model predictions")
        scored = _insert_predictions(conn, predictions_path)
        logger.info("  %s scored rows", f"{scored:,}")

        sweep_rows = _insert_threshold_analysis(conn, Path(threshold_csv))
        _insert_metrics(conn, Path(metrics_path))

        conn.execute("DROP TABLE IF EXISTS scored_staging")
        logger.info("Creating indexes")
        for statement in (
            "CREATE INDEX idx_txn_split ON transactions(split)",
            "CREATE INDEX idx_txn_product ON transactions(product_cd)",
            "CREATE INDEX idx_txn_day ON transactions(day_index)",
            "CREATE INDEX idx_txn_fraud ON transactions(is_fraud)",
            "CREATE INDEX idx_txn_band ON transactions(risk_band)",
            "CREATE INDEX idx_txn_probability ON transactions(fraud_probability)",
            "CREATE INDEX idx_txn_alerted ON transactions(alerted)",
            "CREATE INDEX idx_txn_card1 ON transactions(card1)",
        ):
            conn.execute(statement)
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()

    size_mb = db_path.stat().st_size / 1e6
    logger.info(
        "Database built at %s (%.1f MB, %s rows, %s thresholds) in %.1fs",
        db_path, size_mb, f"{rows:,}", f"{sweep_rows:,}", time.time() - started,
    )
    return db_path


if __name__ == "__main__":
    build_database()
