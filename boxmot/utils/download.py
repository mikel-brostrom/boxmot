#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to download and extract BoxMOT releases and MOT evaluation tools.
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
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

# Mapping for deprecated numpy types
DEPRECATED_TYPES = {"np.float": "float", "np.int": "int", "np.bool": "bool"}


def configure_logger(verbose: bool):
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


def get_http_session(retries: int = 3, backoff_factor: float = 0.3) -> requests.Session:
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
    LOGGER.debug("Created HTTP session with retry strategy.")
    return session


def download_file(url: str, dest: Path, chunk_size: int = 8192, overwrite: bool = False, timeout: int = 10) -> Path:
    if dest.exists() and not overwrite:
        LOGGER.debug(f"Skipping download; file exists: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"[BoxMOT] ⬇️  Downloading {dest.name}...")
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

    LOGGER.debug(f"Downloaded {url} → {dest}")
    return dest


def extract_zip(zip_path: Path, extract_to: Path, overwrite: bool = False) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    try:
        with ZipFile(zip_path, 'r') as zf:
            members = zf.infolist()
            if not overwrite:
                all_exist = all((extract_to / member.filename).exists() for member in members)
                if all_exist:
                    LOGGER.info(f"[BoxMOT] ✅ {zip_path.name} already extracted, skipping.")
                    return

            LOGGER.info(f"[BoxMOT] 📦 Extracting {zip_path.name}...")
            for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
                target = extract_to / member.filename
                if target.exists() and not overwrite:
                    LOGGER.debug(f"Skipping existing file: {target}")
                    continue
                extract_to.mkdir(parents=True, exist_ok=True)
                zf.extract(member, extract_to)

        LOGGER.info(f"[BoxMOT] ✅ Extracted {zip_path.name}")

    except BadZipFile:
        LOGGER.error(f"[BoxMOT] ❌ Corrupt ZIP file: {zip_path}. Removing.")
        try:
            zip_path.unlink()
        except FileNotFoundError:
            pass
        raise


def patch_deprecated_types(root: Path, deprecated: dict = DEPRECATED_TYPES) -> None:
    LOGGER.info(f"[BoxMOT] 🛠️  Patching deprecated numpy types in: {root}")
    for file in root.rglob("*"):
        if file.suffix not in {".py", ".txt"}:
            continue
        text = file.read_text(encoding="utf-8")
        updated = text
        for old, new in deprecated.items():
            updated = updated.replace(old, new)
        if updated != text:
            file.write_text(updated, encoding="utf-8")
            LOGGER.debug(f"Patched deprecated types in {file}")
    LOGGER.info(f"[BoxMOT] ✅ Patching complete.")


def download_trackeval(dest: Path, branch: str = "master", overwrite: bool = False) -> None:
    if dest.exists() and not overwrite:
        LOGGER.info("[BoxMOT] ✅ TrackEval already present, skipping download.")
        return

    LOGGER.info(f"[BoxMOT] ⬇️  Downloading TrackEval ({branch} branch)...")
    repo_url = "https://github.com/JonathonLuiten/TrackEval"
    zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
    zip_file = dest.parent / f"{dest.name}-{branch}.zip"

    download_file(zip_url, zip_file, overwrite=overwrite)
    extract_dir = dest.parent / f"{dest.name}-{branch}"
    extract_zip(zip_file, dest.parent, overwrite=overwrite)

    if extract_dir.exists():
        extract_dir.rename(dest)
        LOGGER.debug(f"Renamed {extract_dir} → {dest}")
    try:
        zip_file.unlink()
    except FileNotFoundError:
        LOGGER.warning(f"ZIP file not found for cleanup: {zip_file}")

    patch_deprecated_types(dest)
    LOGGER.info(f"[BoxMOT] ✅ TrackEval ready at: {dest}")


def download_MOT17_eval_data(runs_url: str, mot17_url: str, mot17_dest: Path, overwrite: bool = False) -> None:
    LOGGER.info("[BoxMOT] ⬇️  Downloading evaluation data...")

    runs_zip = download_file(runs_url, Path("runs.zip"), overwrite=overwrite)
    extract_zip(runs_zip, Path("."), overwrite=overwrite)

    mot17_zip = download_file(mot17_url, mot17_dest, overwrite=overwrite)
    data_dir = mot17_dest.parent / "data"
    extract_zip(mot17_zip, data_dir, overwrite=overwrite)

    LOGGER.info("[BoxMOT] ✅ Evaluation data setup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BoxMOT datasets and MOT evaluation tools.")
    parser.add_argument("--branch", default="master", help="Branch of TrackEval to download.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads and extractions.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    args = parser.parse_args()

    configure_logger(args.verbose)

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
