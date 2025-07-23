from __future__ import annotations

import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

ZENODO_RECORD_URL = "https://zenodo.org/api/records/15236131"
ATLASES_FOLDER = Path(__file__).parent.parent / "registration" / "atlases"
ATLASES_DIR_PATTERN = "atlases_v*.*.*"


def verify_or_download_atlases() -> Path:
    """Check if latest atlases are present and download them otherwise.

    Returns:
        Path: Path to the atlases folder.
    """

    zenodo_data = _get_zenodo_metadata_and_archive_url()
    if zenodo_data:
        zenodo_metadata, archive_url = zenodo_data

    matching_folders = list(ATLASES_FOLDER.glob(ATLASES_DIR_PATTERN))
    # Get the latest downloaded atlases
    latest_downloaded_atlases = _get_latest_version_folder_name(matching_folders)

    if not latest_downloaded_atlases:
        if not zenodo_data:
            logger.error(
                "Atlases not found locally and Zenodo could not be reached. Exiting..."
            )
            sys.exit()
        logger.info(f"Atlases not found locally")

        return _download_atlases(
            zenodo_metadata=zenodo_metadata,
            archive_url=archive_url,
        )

    logger.info(f"Found downloaded local atlases: {latest_downloaded_atlases}")

    if not zenodo_metadata:
        logger.warning(
            "Zenodo server could not be reached. Using the latest downloaded atlases."
        )
        return ATLASES_FOLDER / latest_downloaded_atlases

    # Compare the latest downloaded atlases with the latest Zenodo version
    if zenodo_metadata["version"] == latest_downloaded_atlases.split("_v")[1]:
        logger.info(
            f"Latest atlases ({zenodo_metadata['version']}) are already present."
        )
        return ATLASES_FOLDER / latest_downloaded_atlases

    logger.info(
        f"New atlases available on Zenodo ({zenodo_metadata['version']}). Deleting old and fetching new atlases..."
    )
    # delete old atlases
    try:
        shutil.rmtree(
            ATLASES_FOLDER / latest_downloaded_atlases,
        )
    except OSError as e:
        logger.warning(f"Failed to delete old atlases: {e}")
    return _download_atlases(zenodo_metadata=zenodo_metadata, archive_url=archive_url)


def _get_latest_version_folder_name(folders: List[Path]) -> str | None:
    """Get the latest (non empty) version folder name from the list of folders.

    Args:
        folders (List[Path]): List of folders matching the pattern.

    Returns:
        str | None: Latest version folder name if one exists, else None.
    """
    if not folders:
        return None
    latest_downloaded_folder = sorted(
        folders,
        reverse=True,
        key=lambda x: tuple(map(int, str(x).split("_v")[1].split("."))),
    )[0]
    # check folder is not empty
    if not list(latest_downloaded_folder.glob("*")):
        return None
    return latest_downloaded_folder.name


def _get_zenodo_metadata_and_archive_url() -> Tuple[Dict, str] | None:
    """Get the metadata for the Zenodo record and the files archive url.

    Returns:
        Tuple: (dict: Metadata for the Zenodo record, str: URL to the archive file)
    """
    try:
        response = requests.get(f"{ZENODO_RECORD_URL}")
        if response.status_code != 200:
            logger.error(
                f"Cant find atlases on Zenodo ({ZENODO_RECORD_URL}). Exiting..."
            )
            return None
        data = response.json()
        return data["metadata"], data["links"]["archive"]

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch Zenodo metadata: {e}")
        return None


def _download_atlases(zenodo_metadata: Dict, archive_url: str) -> Path:
    """Download the latest atlases from Zenodo for the requested record and extract them to the target folder.

    Args:
        zenodo_metadata (Dict): Metadata for the Zenodo record.
        archive_url (str): URL to the archive file.

    Returns:
        Path: Path to the atlases folder for the requested record.
    """
    record_folder = ATLASES_FOLDER / f"atlases_v{zenodo_metadata['version']}"
    # ensure folder exists
    record_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading atlases from Zenodo. This might take a while...")
    # Make a GET request to the URL
    response = requests.get(archive_url, stream=True)
    # Ensure the request was successful
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download atlases from {archive_url}. Status code: {response.status_code}"
        )

    _extract_archive(response=response, record_folder=record_folder)

    logger.info(f"Zip file extracted successfully to {record_folder}")
    return record_folder


def _extract_archive(response: requests.Response, record_folder: Path):
    # Download with progress bar
    chunk_size = 1024  # 1KB
    bytes_io = BytesIO()

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Downloading atlases..."),
        TextColumn("[cyan]{task.completed:.2f} MB"),
        transient=True,
    ) as progress:
        task = progress.add_task("", total=None)  # Indeterminate progress

        for data in response.iter_content(chunk_size=chunk_size):
            bytes_io.write(data)
            progress.update(
                task, advance=len(data) / (chunk_size**2)
            )  # Convert bytes to MB

    # Extract the downloaded zip file to the target folder
    with zipfile.ZipFile(bytes_io) as zip_ref:
        zip_ref.extractall(record_folder)

    # check if the extracted file is still a zip
    for f in record_folder.iterdir():
        if f.is_file() and f.suffix == ".zip":
            with zipfile.ZipFile(f) as zip_ref:
                files = zip_ref.namelist()
                with Progress(transient=True) as progress:
                    task = progress.add_task(
                        "[cyan]Extracting files...", total=len(files)
                    )
                    # Iterate over the files and extract them
                    for i, file in enumerate(files):
                        zip_ref.extract(file, record_folder)
                        # Update the progress bar
                        progress.update(task, completed=i + 1)
            f.unlink()  # remove zip after extraction
