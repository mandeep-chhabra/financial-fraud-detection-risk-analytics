"""Fit, compare, select and evaluate the fraud models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_PARQUET  # noqa: E402
from src.models.train import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(PROCESSED_PARQUET))
    parser.add_argument("--quick", action="store_true", help="small subsamples, for smoke tests")
    parser.add_argument("--cost", type=float, default=4.0, help="investigation cost per alert (USD)")
    args = parser.parse_args()

    result = run_training(args.features, quick=args.quick, cost_per_investigation=args.cost)
    print(result["leaderboard"].to_string(index=False))


if __name__ == "__main__":
    main()
