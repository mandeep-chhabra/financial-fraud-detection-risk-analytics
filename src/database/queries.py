"""Named analytical SQL used by the dashboard.

Keeping the SQL in one module means the numbers shown in Streamlit are the same
numbers covered by ``tests/test_sql.py``, not ad-hoc pandas calculations.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from src.config import DB_PATH

QUERIES: dict[str, str] = {
    "overview": """
        SELECT
            COUNT(*)                                        AS total_transactions,
            SUM(is_fraud)                                   AS fraud_transactions,
            COUNT(*) - SUM(is_fraud)                        AS legitimate_transactions,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
            ROUND(SUM(amount), 2)                           AS total_value,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value,
            ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END)
                  / SUM(amount), 4)                         AS fraud_value_pct
        FROM transactions
    """,
    "scored_overview": """
        SELECT
            COUNT(*)                                        AS scored_transactions,
            SUM(alerted)                                    AS alerts,
            ROUND(100.0 * SUM(alerted) / COUNT(*), 4)       AS alert_rate_pct,
            SUM(CASE WHEN alerted = 1 AND is_fraud = 1 THEN 1 ELSE 0 END) AS true_positives,
            SUM(CASE WHEN alerted = 1 AND is_fraud = 0 THEN 1 ELSE 0 END) AS false_positives,
            SUM(CASE WHEN alerted = 0 AND is_fraud = 1 THEN 1 ELSE 0 END) AS false_negatives,
            ROUND(100.0 * SUM(CASE WHEN alerted = 1 AND is_fraud = 1 THEN 1 ELSE 0 END)
                  / NULLIF(SUM(is_fraud), 0), 4)            AS recall_pct,
            ROUND(100.0 * SUM(CASE WHEN alerted = 1 AND is_fraud = 1 THEN 1 ELSE 0 END)
                  / NULLIF(SUM(alerted), 0), 4)             AS precision_pct
        FROM transactions
        WHERE fraud_probability IS NOT NULL
    """,
    "fraud_by_product": """
        SELECT
            product_cd,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
            ROUND(AVG(amount), 2)                           AS avg_amount,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value
        FROM transactions
        GROUP BY product_cd
        ORDER BY frauds DESC
    """,
    "fraud_by_card": """
        SELECT
            COALESCE(card4, 'unknown')                      AS card_brand,
            COALESCE(card6, 'unknown')                      AS card_type,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
            ROUND(AVG(amount), 2)                           AS avg_amount
        FROM transactions
        GROUP BY card_brand, card_type
        HAVING COUNT(*) >= 500
        ORDER BY fraud_rate_pct DESC
    """,
    "identity_coverage": """
        SELECT
            CASE has_identity WHEN 1 THEN 'identity present' ELSE 'identity absent' END AS identity_status,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
            ROUND(AVG(amount), 2)                           AS avg_amount
        FROM transactions
        GROUP BY has_identity
        ORDER BY has_identity DESC
    """,
    "risk_bands": """
        SELECT
            risk_band,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            ROUND(AVG(fraud_probability), 6)                AS avg_probability,
            ROUND(SUM(amount), 2)                           AS total_value,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct
        FROM transactions
        WHERE risk_band IS NOT NULL
        GROUP BY risk_band
        ORDER BY CASE risk_band
                    WHEN 'High Risk' THEN 1
                    WHEN 'Medium Risk' THEN 2
                    ELSE 3 END
    """,
    "daily_trends": """
        SELECT
            day_index                                      AS day,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
            ROUND(SUM(amount), 2)                           AS total_value,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value
        FROM transactions
        GROUP BY day_index
        ORDER BY day_index
    """,
    "amount_statistics": """
        SELECT
            is_fraud,
            COUNT(*)                                        AS transactions,
            ROUND(AVG(amount), 2)                           AS mean_amount,
            ROUND(MIN(amount), 2)                           AS min_amount,
            ROUND(MAX(amount), 2)                           AS max_amount,
            ROUND(SUM(amount), 2)                           AS total_amount
        FROM transactions
        GROUP BY is_fraud
        ORDER BY is_fraud
    """,
    "fraud_by_amount_band": """
        SELECT
            CASE
                WHEN amount < 25   THEN '1. under $25'
                WHEN amount < 100  THEN '2. $25-$100'
                WHEN amount < 300  THEN '3. $100-$300'
                WHEN amount < 1000 THEN '4. $300-$1k'
                ELSE '5. $1k+'
            END                                             AS amount_band,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
            ROUND(AVG(amount), 2)                           AS avg_amount
        FROM transactions
        GROUP BY amount_band
        ORDER BY amount_band
    """,
    "hourly_fraud": """
        SELECT
            txn_hour % 24                                   AS hour_of_day,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 5)      AS fraud_rate_pct
        FROM transactions
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """,
    "error_breakdown": """
        SELECT
            CASE
                WHEN is_fraud = 1 AND alerted = 1 THEN 'True Positive'
                WHEN is_fraud = 0 AND alerted = 1 THEN 'False Positive'
                WHEN is_fraud = 1 AND alerted = 0 THEN 'False Negative'
                ELSE 'True Negative'
            END                                             AS outcome,
            COUNT(*)                                        AS transactions,
            ROUND(SUM(amount), 2)                           AS total_value,
            ROUND(AVG(amount), 2)                           AS avg_value
        FROM transactions
        WHERE fraud_probability IS NOT NULL
        GROUP BY outcome
        ORDER BY transactions DESC
    """,
    "errors_by_product": """
        SELECT
            product_cd,
            SUM(CASE WHEN is_fraud = 0 AND alerted = 1 THEN 1 ELSE 0 END) AS false_positives,
            SUM(CASE WHEN is_fraud = 1 AND alerted = 0 THEN 1 ELSE 0 END) AS false_negatives,
            SUM(CASE WHEN is_fraud = 1 AND alerted = 1 THEN 1 ELSE 0 END) AS true_positives
        FROM transactions
        WHERE fraud_probability IS NOT NULL
        GROUP BY product_cd
        ORDER BY false_positives DESC
    """,
    "split_summary": """
        SELECT
            split,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            COUNT(fraud_probability)                        AS scored,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
            MIN(day_index)                                  AS first_day,
            MAX(day_index)                                  AS last_day
        FROM transactions
        GROUP BY split
        ORDER BY CASE split
                    WHEN 'train' THEN 1
                    WHEN 'validation' THEN 2
                    ELSE 3 END
    """,
    "card_concentration": """
        SELECT
            card1,
            COUNT(*)                                        AS transactions,
            SUM(is_fraud)                                   AS frauds,
            ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4)      AS fraud_rate_pct,
            ROUND(SUM(amount), 2)                           AS total_value
        FROM transactions
        WHERE card1 IS NOT NULL
        GROUP BY card1
        HAVING COUNT(*) >= 200
        ORDER BY frauds DESC
        LIMIT 15
    """,
    "model_metrics": """
        SELECT metric, value, eval_split
        FROM model_metrics
        ORDER BY eval_split, metric
    """,
    "model_metadata": """
        SELECT key, value FROM model_metadata ORDER BY key
    """,
    "threshold_analysis": """
        SELECT * FROM threshold_analysis ORDER BY threshold
    """,
    "top_risk_transactions": """
        SELECT
            txn_id, day_index, product_cd, amount, is_fraud,
            fraud_probability, risk_band, alerted
        FROM transactions
        WHERE fraud_probability IS NOT NULL
        ORDER BY fraud_probability DESC
        LIMIT :limit
    """,
    "risk_band_fraud_value": """
        SELECT
            risk_band,
            COUNT(*)                                        AS transactions,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value,
            ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END)
                  / (SELECT SUM(amount) FROM transactions
                     WHERE fraud_probability IS NOT NULL AND is_fraud = 1), 4) AS share_of_fraud_value_pct
        FROM transactions
        WHERE fraud_probability IS NOT NULL
        GROUP BY risk_band
        ORDER BY CASE risk_band WHEN 'High Risk' THEN 1 WHEN 'Medium Risk' THEN 2 ELSE 3 END
    """,
    "confusion_at_threshold": """
        SELECT
            SUM(CASE WHEN is_fraud = 1 AND fraud_probability >= :threshold THEN 1 ELSE 0 END) AS tp,
            SUM(CASE WHEN is_fraud = 0 AND fraud_probability >= :threshold THEN 1 ELSE 0 END) AS fp,
            SUM(CASE WHEN is_fraud = 1 AND fraud_probability <  :threshold THEN 1 ELSE 0 END) AS fn,
            SUM(CASE WHEN is_fraud = 0 AND fraud_probability <  :threshold THEN 1 ELSE 0 END) AS tn,
            SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END)                                 AS total_fraud_amount,
            SUM(CASE WHEN is_fraud = 1 AND fraud_probability >= :threshold THEN amount ELSE 0 END) AS fraud_value_captured
        FROM transactions
        WHERE fraud_probability IS NOT NULL
    """,
}

QUERY_GROUPS = {
    "overview": "Portfolio",
    "scored_overview": "Portfolio",
    "fraud_by_product": "Segments",
    "fraud_by_card": "Segments",
    "identity_coverage": "Segments",
    "fraud_by_amount_band": "Segments",
    "card_concentration": "Segments",
    "risk_bands": "Risk",
    "risk_band_fraud_value": "Risk",
    "confusion_at_threshold": "Risk",
    "daily_trends": "Time",
    "hourly_fraud": "Time",
    "amount_statistics": "Distribution",
    "error_breakdown": "Model",
    "errors_by_product": "Model",
    "split_summary": "Model",
    "model_metrics": "Model",
    "model_metadata": "Model",
    "threshold_analysis": "Model",
    "top_risk_transactions": "Risk",
}


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run scripts/build_database.py first."
        )
    # URI paths must be POSIX-style and URL-quoted on Windows (spaces, drive letters).
    uri_path = quote(db_path.resolve().as_posix())
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def run_query(name: str, params: dict | None = None, db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Execute a named query and return a DataFrame."""
    if name not in QUERIES:
        raise KeyError(f"Unknown query {name!r}. Available: {sorted(QUERIES)}")
    with get_connection(db_path) as conn:
        return pd.read_sql_query(QUERIES[name], conn, params=params or {})


def scalar(name: str, params: dict | None = None, db_path: str | Path = DB_PATH):
    frame = run_query(name, params, db_path)
    return frame.iloc[0].to_dict() if len(frame) else {}


def list_queries() -> list[str]:
    return sorted(QUERIES)
