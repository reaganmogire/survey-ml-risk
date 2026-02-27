#!/usr/bin/env python3
"""
Download large assets from GitHub Releases.

This script:
- Downloads BRFSS zipped annual extracts (data-v1)
- Downloads trained model artifacts (model-v1)

Keeps the git repository lightweight and fully reproducible.
"""

from pathlib import Path
import argparse
import requests

OWNER = "reaganmogire"
REPO = "survey-ml-risk"
BASE_URL = f"https://github.com/{OWNER}/{REPO}/releases/download"

DATA_FILES = [
    "2011.csv.zip",
    "2012.csv.zip",
    "2013.csv.zip",
    "2014.csv.zip",
    "2015.csv.zip",
]

MODEL_FILES = [
    "disease_models.joblib",
    "optimal_thresholds.joblib",
    "predictor_cols.joblib",
]


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"✓ Already exists: {dest}")
        return

    print(f"Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp.replace(dest)
    print(f"✓ Saved to {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-tag", default="data-v1")
    parser.add_argument("--model-tag", default="model-v1")
    args = parser.parse_args()

    # Download data zips
    for file in DATA_FILES:
        url = f"{BASE_URL}/{args.data_tag}/{file}"
        download(url, Path("data/raw") / file)

    # Download model artifacts
    for file in MODEL_FILES:
        url = f"{BASE_URL}/{args.model_tag}/{file}"
        download(url, Path("app/artifacts") / file)


if __name__ == "__main__":
    main()
