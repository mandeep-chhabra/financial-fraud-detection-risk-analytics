"""Create the SQLite database used by the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.build_db import build_database  # noqa: E402

if __name__ == "__main__":
    build_database()
