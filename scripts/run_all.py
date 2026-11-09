"""Run the whole pipeline end-to-end, in dependency order.

If ``ieee-fraud-detection.zip`` is in the repository root it is extracted;
otherwise the two labelled files are fetched from a public mirror.

    python scripts/download_data.py
    python scripts/run_all.py

Each stage is idempotent at the artefact level: data is downloaded once, the
feature table is reused unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_dataset import build  # noqa: E402
from scripts.download_data import download  # noqa: E402
from scripts.run_eda import run_eda  # noqa: E402
from src.config import PROCESSED_PARQUET  # noqa: E402
from src.database.build_db import build_database  # noqa: E402
from src.evaluation.report import generate_report  # noqa: E402
from src.explainability.explain import run_shap_analysis  # noqa: E402
from src.models.train import run_training  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("run_all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild the feature table")
    parser.add_argument("--quick", action="store_true", help="smoke-test configuration")
    args = parser.parse_args()

    started = time.time()
    logger.info("[1/6] dataset")
    download()

    logger.info("[2/6] feature table")
    build(force=args.force)

    logger.info("[3/6] EDA")
    run_eda()

    logger.info("[4/6] training + evaluation")
    run_training(features_path=PROCESSED_PARQUET, quick=args.quick)

    logger.info("[5/6] database")
    build_database()

    logger.info("[6/7] explainability")
    run_shap_analysis()

    logger.info("[7/7] evaluation report")
    generate_report()

    logger.info("Pipeline finished in %.1f minutes", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
