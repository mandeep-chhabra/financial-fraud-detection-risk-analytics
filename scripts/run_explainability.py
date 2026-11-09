"""Compute SHAP explanations for a sample of scored test transactions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explainability.explain import run_shap_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    result = run_shap_analysis(n_samples=args.samples)
    print(result["importance"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
