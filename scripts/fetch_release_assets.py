#!/usr/bin/env python3
"""scripts/fetch_release_assets.py

Download large assets from GitHub Releases to keep the repository lightweight and reproducible.

What it downloads
-----------------
1) Data release (default tag: data-v1)
   - BRFSS annual extracts (2011–2015): 2011.csv.zip ... 2015.csv.zip  -> data/raw/
   - Optional pooled cleaned modeling dataset: BRFSS_2011_2015_clean_model.csv.gz -> data/derived/

2) Model release (default tag: model-v1)
   - disease_models.joblib
   - optimal_thresholds.joblib
   - predictor_cols.joblib
   -> app/artifacts/

Notes
-----
- Uses direct release-download URLs:
    https://github.com/<owner>/<repo>/releases/download/<tag>/<asset>
- If a destination file exists and is non-empty, it will be skipped unless --force is set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import requests

OWNER = "reaganmogire"
REPO = "survey-ml-risk"
BASE_URL = f"https://github.com/{OWNER}/{REPO}/releases/download"

RAW_DATA_FILES = [
    "2011.csv.zip",
    "2012.csv.zip",
    "2013.csv.zip",
    "2014.csv.zip",
    "2015.csv.zip",
]

DERIVED_DATA_FILES = [
    "BRFSS_2011_2015_clean_model.csv.gz",
]

MODEL_FILES = [
    "disease_models.joblib",
    "optimal_thresholds.joblib",
    "predictor_cols.joblib",
]


def download(url: str, dest: Path, *, force: bool = False, timeout: int = 180) -> None:
    """Stream-download a URL to a destination file."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and dest.exists() and dest.stat().st_size > 0:
        print(f"✓ Already exists: {dest}")
        return

    print(f"↓ Downloading: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", "0")) or None
            downloaded = 0

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100.0 * downloaded / total
                        sys.stdout.write(
                            f"\r  {downloaded/1e6:,.1f} MB / {total/1e6:,.1f} MB ({pct:5.1f}%)"
                        )
                        sys.stdout.flush()

            if total:
                sys.stdout.write("\n")

        tmp.replace(dest)
        print(f"✓ Saved to: {dest}")

    except requests.HTTPError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"HTTP error while downloading {url}: {e}") from e
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description="Download data/model assets from GitHub Releases.")
    p.add_argument("--data-tag", default="data-v1", help="Release tag for data assets (default: data-v1)")
    p.add_argument("--model-tag", default="model-v1", help="Release tag for model assets (default: model-v1)")
    p.add_argument("--only-data", action="store_true", help="Download only data assets")
    p.add_argument("--only-model", action="store_true", help="Download only model artifacts")
    p.add_argument("--force", action="store_true", help="Re-download and overwrite existing files")
    p.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds (default: 180)")
    args = p.parse_args()

    if args.only_data and args.only_model:
        raise SystemExit("ERROR: choose at most one of --only-data or --only-model")

    do_data = args.only_data or (not args.only_model)
    do_model = args.only_model or (not args.only_data)

    if do_data:
        for file in RAW_DATA_FILES:
            url = f"{BASE_URL}/{args.data_tag}/{file}"
            download(url, Path("data/raw") / file, force=args.force, timeout=args.timeout)

        for file in DERIVED_DATA_FILES:
            url = f"{BASE_URL}/{args.data_tag}/{file}"
            download(url, Path("data/derived") / file, force=args.force, timeout=args.timeout)

    if do_model:
        for file in MODEL_FILES:
            url = f"{BASE_URL}/{args.model_tag}/{file}"
            download(url, Path("app/artifacts") / file, force=args.force, timeout=args.timeout)


if __name__ == "__main__":
    main()

