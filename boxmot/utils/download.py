#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to download and extract BoxMOT releases and MOT evaluation tools.
"""

import argparse
from pathlib import Path
from typing import Optional
from zipfile import BadZipFile, ZipFile

import gdown
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

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
    return session


def download_file(url: str, dest: Path, chunk_size: int = 8192, overwrite: bool = False, timeout: int = 10) -> Path:
    """
    Download a file from a URL to a destination path, with progress and logging.
    Returns the path to the downloaded file.
    """
    if dest.exists() and not overwrite:
        LOGGER.debug(f"Cached: {dest.name}")
        return dest

    # Ensure parent dir
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Downloading {dest.name}...")

    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        # Google Drive: use gdown (handles confirm tokens automatically)
        gdown.download(
            url=url,
            output=str(dest),
            quiet=False,
            fuzzy=True
        )
    else:
        session = get_http_session()
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", 0))

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

    return dest


def extract_zip(zip_path: Path, extract_to: Path, overwrite: bool = False) -> None:
    """
    Extract a ZIP archive to a target directory.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    try:
        with ZipFile(zip_path, 'r') as zf:
            members = zf.infolist()
            total_files = len(members)

            if not overwrite:
                already = [member.filename for member in members if (extract_to / member.filename).exists()]
                if len(already) == total_files:
                    LOGGER.debug(f"Cached: {zip_path.name} (extracted)")
                    return

            LOGGER.info(f"Extracting {zip_path.name}...")
            for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
                target = extract_to / member.filename
                if target.exists() and not overwrite:
                    continue
                extract_to.mkdir(parents=True, exist_ok=True)
                zf.extract(member, extract_to)

    except BadZipFile:
        LOGGER.error(f"Corrupt ZIP: {zip_path.name}")
        try:
            zip_path.unlink()
        except FileNotFoundError:
            pass
        raise


def patch_deprecated_types(root: Path, deprecated: dict = DEPRECATED_TYPES) -> None:
    """Patch deprecated numpy types in Python files."""
    LOGGER.debug(f"Patching numpy types in: {root.name}")
    for file in root.rglob("*"):
        if file.suffix not in {".py", ".txt"}:
            continue
        text = file.read_text(encoding="utf-8")
        updated = text
        for old, new in deprecated.items():
            updated = updated.replace(old, new)
        if updated != text:
            file.write_text(updated, encoding="utf-8")


def download_trackeval(dest: Path, branch: str = "master", overwrite: bool = False) -> None:
    """
    Download and set up the TrackEval repository into the given destination folder.

    Args:
        dest (Path): target directory for TrackEval (e.g. boxmot/engine/trackeval)
        branch (str): Git branch to download (default "master")
        overwrite (bool): if True, force re-download even if dest already exists
    """
    # If already exists and we're not overwriting, skip
    if dest.exists() and not overwrite:
        LOGGER.debug("TrackEval already present")
        return

    LOGGER.info("Downloading TrackEval...")
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
        LOGGER.warning("Couldn't locate extracted TrackEval folder")
    else:
        extracted.rename(dest)

    # Clean up the downloaded zip
    try:
        zip_file.unlink()
    except FileNotFoundError:
        pass

    # Apply any necessary patches for deprecated types
    patch_deprecated_types(dest)

    LOGGER.debug("TrackEval setup complete")

    
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
    Always downloads+unzips the benchmark data.
    """
    LOGGER.info("Setting up evaluation data...")

    # Optional runs data
    if runs_url:
        runs_zip = download_file(runs_url, Path("runs.zip"), overwrite=overwrite)
        extract_zip(runs_zip, Path("."), overwrite=overwrite)

    if not dataset_url:
        return

    # benchmark ZIP
    benchmark_zip = download_file(dataset_url, dataset_dest, overwrite=overwrite)
    extract_zip(benchmark_zip, dataset_dest.parent, overwrite=overwrite)

    LOGGER.debug(f"Benchmark data ready at: {dataset_dest.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BoxMOT datasets and MOT evaluation tools.")
    parser.add_argument("--branch", default="master", help="Branch of TrackEval to download.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads and extractions.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    args = parser.parse_args()

    download_trackeval(
        dest=Path("TrackEval"),
        branch=args.branch,
        overwrite=args.overwrite
    )

    download_eval_data(
        runs_url="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.7/runs.zip",
        dataset_url="https://github.com/mikel-brostrom/boxmot/releases/download/v10.0.83/MOT17-50.zip",
        dataset_dest=Path("boxmot/engine/TrackEval/MOT17-ablation.zip"),
        overwrite=args.overwrite
    )
