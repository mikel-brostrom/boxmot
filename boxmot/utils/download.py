#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to download and extract BoxMOT releases and MOT evaluation tools,
now with detailed logging of download and extraction paths.
"""

import argparse
import os
import logging
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Setup basic logger (will be reconfigured in main)
LOGGER = logging.getLogger("BoxMOT")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

# Mapping for deprecated numpy types
DEPRECATED_TYPES = {"np.float": "float", "np.int": "int", "np.bool": "bool"}


def configure_logger(verbose: bool):
    """Configure root logger level."""
    level = logging.DEBUG if verbose else logging.INFO
    LOGGER.setLevel(level)
    LOGGER.debug(f"Logger configured to level: {logging.getLevelName(level)}")


def get_http_session(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
    """Create HTTP session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        backoff_factor=backoff_factor
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    LOGGER.debug("Created HTTP session with retry strategy retries=%d, backoff_factor=%0.1f", retries, backoff_factor)
    return session


def download_file(url: str, dest: Path, chunk_size: int = 8192, overwrite: bool = False, timeout: int = 10) -> Path:
    """
    Download a file from a URL to a destination path, with progress and logging.
    Returns the path to the downloaded file.
    """
    if dest.exists() and not overwrite:
        LOGGER.info(f"[BoxMOT] ✅ Skipping download; file already exists at: {dest.resolve()}")
        return dest

    # Ensure parent dir
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"[BoxMOT] ⬇️  Starting download: {url}\n            → Saving to: {dest.resolve()}")

    session = get_http_session()
    response = session.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0))
    LOGGER.debug(f"Expected download size: {total} bytes")

    with open(dest, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {dest.name}"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    file_size = dest.stat().st_size
    LOGGER.info(f"[BoxMOT] ✅ Download complete: {dest.resolve()} ({file_size} bytes)")
    LOGGER.debug(f"Downloaded URL '{url}' to '{dest}' with final size {file_size} bytes")
    return dest


def extract_zip(zip_path: Path, extract_to: Path, overwrite: bool = False) -> None:
    """
    Extract a ZIP archive to a target directory, logging each extracted file.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    LOGGER.info(f"[BoxMOT] 📦 Preparing to extract: {zip_path.resolve()}\n            → Destination: {extract_to.resolve()}")
    try:
        with ZipFile(zip_path, 'r') as zf:
            members = zf.infolist()
            total_files = len(members)
            LOGGER.debug(f"ZIP contains {total_files} entries to consider for extraction.")

            if not overwrite:
                already = [member.filename for member in members if (extract_to / member.filename).exists()]
                if len(already) == total_files:
                    LOGGER.info(f"[BoxMOT] ✅ All files already extracted in {extract_to.resolve()}, skipping.")
                    return

            LOGGER.info(f"[BoxMOT] 📂 Extracting {zip_path.name} ({total_files} files)...")
            for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
                target = extract_to / member.filename
                if target.exists() and not overwrite:
                    LOGGER.debug(f"Skipping existing file: {target.resolve()}")
                    continue
                extract_to.mkdir(parents=True, exist_ok=True)
                zf.extract(member, extract_to)
                LOGGER.debug(f"Extracted: {target.resolve()}")

        LOGGER.info(f"[BoxMOT] ✅ Extraction complete for {zip_path.name} to {extract_to.resolve()}")

    except BadZipFile:
        LOGGER.error(f"[BoxMOT] ❌ Corrupt ZIP detected: {zip_path.resolve()}. Removing corrupted file.")
        try:
            zip_path.unlink()
            LOGGER.debug(f"Removed corrupted ZIP: {zip_path.resolve()}")
        except FileNotFoundError:
            LOGGER.warning(f"ZIP file already removed: {zip_path.resolve()}")
        raise


def patch_deprecated_types(root: Path, deprecated: dict = DEPRECATED_TYPES) -> None:
    LOGGER.info(f"[BoxMOT] 🛠️  Patching deprecated numpy types in directory: {root.resolve()}")
    for file in root.rglob("*"):
        if file.suffix not in {".py", ".txt"}:
            continue
        text = file.read_text(encoding="utf-8")
        updated = text
        for old, new in deprecated.items():
            updated = updated.replace(old, new)
        if updated != text:
            file.write_text(updated, encoding="utf-8")
            LOGGER.debug(f"Patched deprecated types in: {file.resolve()}")
    LOGGER.info(f"[BoxMOT] ✅ Deprecated numpy type patching complete.")


def download_trackeval(dest: Path, branch: str = "master", overwrite: bool = False) -> None:
    if dest.exists() and not overwrite:
        LOGGER.info(f"[BoxMOT] ✅ TrackEval already present at {dest.resolve()}, skipping download.")
        return

    LOGGER.info(f"[BoxMOT] ⬇️  Downloading TrackEval (branch: {branch})")
    repo_url = "https://github.com/JonathonLuiten/TrackEval"
    zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
    zip_file = dest.parent / f"{dest.name}-{branch}.zip"

    zip_path = download_file(zip_url, zip_file, overwrite=overwrite)
    extract_zip(zip_path, dest.parent, overwrite=overwrite)

    extract_dir = dest.parent / f"{dest.name}-{branch}"
    if extract_dir.exists():
        extract_dir.rename(dest)
        LOGGER.debug(f"Renamed extracted folder: {extract_dir.resolve()} → {dest.resolve()}")
    try:
        zip_file.unlink()
        LOGGER.debug(f"Cleaned up ZIP file: {zip_file.resolve()}")
    except FileNotFoundError:
        LOGGER.warning(f"ZIP file not found for cleanup: {zip_file.resolve()}")

    patch_deprecated_types(dest)
    LOGGER.info(f"[BoxMOT] ✅ TrackEval setup complete at: {dest.resolve()}")


def download_MOT17_eval_data(runs_url: str, mot17_url: str, mot17_dest: Path, overwrite: bool = False) -> None:
    LOGGER.info(f"[BoxMOT] ⬇️  Downloading evaluation data from runs_url and mot17_url")

    # Download runs.zip
    runs_zip = download_file(runs_url, Path("runs.zip"), overwrite=overwrite)
    extract_zip(runs_zip, Path("."), overwrite=overwrite)

    # Download MOT17 ZIP
    mot17_zip = download_file(mot17_url, mot17_dest, overwrite=overwrite)
    data_dir = mot17_dest.parent / "data"
    extract_zip(mot17_zip, data_dir, overwrite=overwrite)

    LOGGER.info(f"[BoxMOT] ✅ Evaluation data setup complete. Runs data at './runs.zip', MOT17 data at '{data_dir.resolve()}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BoxMOT datasets and MOT evaluation tools.")
    parser.add_argument("--branch", default="master", help="Branch of TrackEval to download.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads and extractions.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    args = parser.parse_args()

    configure_logger(args.verbose)

    LOGGER.debug(f"Script started with args: branch={args.branch}, overwrite={args.overwrite}, verbose={args.verbose}")

    download_trackeval(
        dest=Path("TrackEval"),
        branch=args.branch,
        overwrite=args.overwrite
    )

    download_MOT17_eval_data(
        runs_url="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.7/runs.zip",
        mot17_url="https://github.com/mikel-brostrom/boxmot/releases/download/v10.0.83/MOT17-50.zip",
        mot17_dest=Path("boxmot/engine/TrackEval/MOT17-ablation.zip"),
        overwrite=args.overwrite
    )
