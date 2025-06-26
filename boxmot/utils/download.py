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
from typing import Optional, List, Dict, Generator, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from boxmot.utils import logger as LOGGER

# Mapping for deprecated numpy types
DEPRECATED_TYPES = {"np.float": "float", "np.int": "int", "np.bool": "bool"}


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
    """
    Download and set up the TrackEval repository into the given destination folder.

    Args:
        dest (Path): target directory for TrackEval (e.g. boxmot/engine/trackeval)
        branch (str): Git branch to download (default "master")
        overwrite (bool): if True, force re-download even if dest already exists
    """
    # If already exists and we’re not overwriting, skip
    if dest.exists() and not overwrite:
        LOGGER.info(f"[BoxMOT] ✅ TrackEval already present at {dest.resolve()}, skipping download.")
        return

    LOGGER.info(f"[BoxMOT] ⬇️  Downloading TrackEval (branch: {branch})")
    repo_url = "https://github.com/JonathonLuiten/TrackEval"
    zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
    zip_file = dest.parent / f"{dest.name}-{branch}.zip"

    # Download the archive
    zip_path = download_file(zip_url, zip_file, overwrite=overwrite)

    # Extract into the parent folder
    extract_zip(zip_path, dest.parent, overwrite=overwrite)

    # GitHub will unpack to "TrackEval-master" (with original casing); 
    # rename it case-insensitively to our lowercase 'trackeval' folder
    extracted = None
    for d in dest.parent.iterdir():
        if d.is_dir() and d.name.lower().startswith("trackeval") and d.name.lower().endswith(f"-{branch}"):
            extracted = d
            break

    if extracted is None:
        LOGGER.warning(f"[BoxMOT] ❗️ Couldn't locate extracted TrackEval in {dest.parent}, expected folder ending with '-{branch}'")
    else:
        extracted.rename(dest)
        LOGGER.debug(f"[BoxMOT] Renamed extracted folder: {extracted.resolve()} → {dest.resolve()}")

    # Clean up the downloaded zip
    try:
        zip_file.unlink()
        LOGGER.debug(f"[BoxMOT] Cleaned up ZIP file: {zip_file.resolve()}")
    except FileNotFoundError:
        LOGGER.warning(f"[BoxMOT] ZIP file not found for cleanup: {zip_file.resolve()}")

    # Apply any necessary patches for deprecated types
    patch_deprecated_types(dest)

    LOGGER.info(f"[BoxMOT] ✅ TrackEval setup complete at: {dest.resolve()}")

    
def download_eval_data(
    *,
    runs_url: Optional[str] = None,
    dataset_url: str,
    dataset_dest: Path,
    overwrite: bool = False
) -> None:
    """
    Download & extract TrackEval evaluation data.
    If `runs_url` is truthy, downloads+unzips runs.zip; otherwise skips it.
    Always downloads+unzips the MOT17 data.
    """
    LOGGER.info(f"[BoxMOT] ⬇️  Setting up evaluation data")

    # Optional runs data
    if runs_url:
        LOGGER.info(f"[BoxMOT] ⬇️  Downloading runs.zip from {runs_url}")
        runs_zip = download_file(runs_url, Path("runs.zip"), overwrite=overwrite)
        extract_zip(runs_zip, Path("."), overwrite=overwrite)
    else:
        LOGGER.debug(f"[BoxMOT] ⚠️  No runs_url provided, skipping runs download")

    # MOT17 ZIP
    LOGGER.info(f"[BoxMOT] ⬇️  Downloading MOT17 data from {dataset_url}")
    mot17_zip = download_file(dataset_url, dataset_dest, overwrite=overwrite)
    data_dir = dataset_dest.parent / "data"
    extract_zip(mot17_zip, data_dir, overwrite=overwrite)

    LOGGER.info(
        f"[BoxMOT] ✅ Evaluation data setup complete. "
        f"MOT17 data at '{data_dir.resolve()}'"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BoxMOT datasets and MOT evaluation tools.")
    parser.add_argument("--branch", default="master", help="Branch of TrackEval to download.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads and extractions.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    args = parser.parse_args()

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
