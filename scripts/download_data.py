"""Provide the IEEE-CIS raw files.

Two paths are supported:

1. A local Kaggle archive (``ieee-fraud-detection.zip`` in the repository root),
   which is what most users will have after accepting the competition rules.
2. A public Hugging Face mirror, used when the archive is absent, so the project
   stays reproducible on a fresh machine.

Only the labelled ``train_*`` files are needed: the competition test files ship
without ``isFraud``, so the model is trained and evaluated on a temporal split of
the labelled file instead.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import IEEE_DIR, PROJECT_ROOT  # noqa: E402

ZIP_PATH = PROJECT_ROOT / "ieee-fraud-detection.zip"
MIRROR = "https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main/{name}"
FILES = {
    "train_transaction.csv": 600_000_000,
    "train_identity.csv": 20_000_000,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _already_present() -> bool:
    return all(
        (IEEE_DIR / name).exists() and (IEEE_DIR / name).stat().st_size >= minimum
        for name, minimum in FILES.items()
    )


def _extract() -> None:
    print(f"Extracting {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        for name in FILES:
            archive.extract(name, IEEE_DIR)


def _download(name: str) -> None:
    url = MIRROR.format(name=name)
    print(f"Downloading {url}")
    target = IEEE_DIR / name
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        tmp = target.with_suffix(".csv.part")
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(target)


def download(force: bool = False) -> Path:
    IEEE_DIR.mkdir(parents=True, exist_ok=True)
    if not force and _already_present():
        print(f"IEEE-CIS files already present in {IEEE_DIR}")
    elif ZIP_PATH.exists():
        _extract()
    else:
        for name in FILES:
            _download(name)

    for name, minimum in FILES.items():
        path = IEEE_DIR / name
        if not path.exists() or path.stat().st_size < minimum:
            raise RuntimeError(f"{path} is missing or unexpectedly small")
        print(f"{name}: {path.stat().st_size:,} bytes  sha256={sha256(path)[:16]}...")
    return IEEE_DIR


if __name__ == "__main__":
    download(force="--force" in sys.argv)
