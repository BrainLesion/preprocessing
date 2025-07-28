from __future__ import annotations

import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

ATLASES_FOLDER = Path(__file__).parent.parent / "registration" / "atlases"
ATLASES_RECORD_ID = "15236131"

SYNTHSTRIP_FOLDER = Path(__file__).parent.parent / "brain_extraction" / "weights"
SYNTHSTRIP_RECORD_ID = "16535633"


def fetch_atlases() -> Path:
    """
    Ensure that the required atlases are available locally, downloading them if necessary.

    Returns:
        Path: The path to the folder containing the atlases.
    """
    record = ZenodoRecord(
        record_id=ATLASES_RECORD_ID,
        target_dir=ATLASES_FOLDER,
        label="atlases",
    )
    return record.fetch()


def fetch_synthstrip() -> Path:
    """
    Ensure that the SynthStrip weights are available locally, downloading them if necessary.

    Returns:
        Path: The path to the folder containing the SynthStrip weights.
    """
    record = ZenodoRecord(
        record_id=SYNTHSTRIP_RECORD_ID,
        target_dir=SYNTHSTRIP_FOLDER,
        label="SynthStrip",
    )
    return record.fetch()


class ZenodoException(Exception):
    """Raised when Zenodo cannot be reached or the request fails."""

    pass


class ZenodoRecord:
    BASE_URL = "https://zenodo.org/api/records"

    def __init__(
        self,
        record_id: str,
        target_dir: Path,
        label: str = "asset",
    ):
        self.record_id = record_id
        self.target_dir = target_dir
        self.label = label

    def fetch(self) -> Path:
        """Fetch the latest version of the record from Zenodo or from local storage."""
        zenodo_response = self._get_metadata_and_archive_url()

        pattern = self._glob_pattern()
        matching_folders = list(self.target_dir.glob(pattern))
        latest_local = self._get_latest_version_folder_name(matching_folders)

        if not latest_local:
            if not zenodo_response:
                msg = f"{self.label.title()} not found locally and Zenodo could not be reached."
                logger.error(msg)
                raise ZenodoException(msg)

            logger.info(f"{self.label.title()} not found locally.")
            metadata, archive_url = zenodo_response
            return self._download(metadata, archive_url)

        logger.info(f"Found local {self.label}: {latest_local}")

        if not zenodo_response:
            logger.warning(f"Zenodo unreachable. Using latest downloaded {self.label}.")
            return self.target_dir / latest_local

        metadata, archive_url = zenodo_response
        remote_version = metadata["version"]
        local_version = latest_local.split("_v")[1]

        if remote_version == local_version:
            logger.info(f"Latest {self.label} ({remote_version}) already present.")
            return self.target_dir / latest_local

        logger.info(
            f"New version of {self.label} available on Zenodo ({remote_version}). Replacing local copy..."
        )
        shutil.rmtree(
            self.target_dir / latest_local,
            onerror=lambda func, path, excinfo: logger.warning(
                f"Failed to delete {path}: {excinfo}"
            ),
        )
        return self._download(metadata, archive_url)

    def _glob_pattern(self) -> str:
        return f"{self.record_id}_v*.*.*"

    def _build_folder_path(
        self,
        version: str,
    ) -> Path:
        return self.target_dir / f"{self.record_id}_v{version}"

    def _get_latest_version_folder_name(
        self,
        folders: List[Path],
    ) -> str | None:
        if not folders:
            return None
        latest = sorted(
            folders,
            reverse=True,
            key=lambda x: tuple(map(int, str(x.name).split("_v")[1].split("."))),
        )[0]
        if not list(latest.glob("*")):
            return None
        return latest.name

    def _get_metadata_and_archive_url(self) -> Tuple[Dict, str] | None:
        try:
            response = requests.get(f"{self.BASE_URL}/{self.record_id}")
            if response.status_code != 200:
                error_msg = (
                    f"Cannot find record '{self.record_id}' on Zenodo "
                    f"({response.status_code=})."
                )
                logger.error(error_msg)
                raise ZenodoException(error_msg)

            data = response.json()
            return data["metadata"], data["links"]["archive"]

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch metadata from Zenodo: {e}")
            return None

    def _download(
        self,
        metadata: Dict,
        archive_url: str,
    ) -> Path:
        folder = self._build_folder_path(metadata["version"])
        folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {self.label} from Zenodo. This may take a while...")

        response = requests.get(archive_url, stream=True)
        if response.status_code != 200:
            msg = (
                f"Failed to download {self.label}. Status code: {response.status_code}"
            )
            logger.error(msg)
            raise ZenodoException(msg)

        self._extract_archive(response, folder)
        logger.info(f"{self.label.title()} extracted to {folder}")
        return folder

    def _extract_archive(
        self,
        response: requests.Response,
        folder: Path,
    ):
        chunk_size = 1024  # 1KB
        buffer = BytesIO()

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Downloading {self.label}..."),
            TextColumn("[cyan]{task.completed:.2f} MB"),
            transient=True,
        ) as progress:
            task = progress.add_task("", total=None)
            for chunk in response.iter_content(chunk_size=chunk_size):
                buffer.write(chunk)
                progress.update(task, advance=len(chunk) / (chunk_size**2))

        with zipfile.ZipFile(buffer) as z:
            z.extractall(folder)

        for file in folder.iterdir():
            if file.is_file() and file.suffix == ".zip":
                with zipfile.ZipFile(file) as inner_zip:
                    files = inner_zip.namelist()
                    with Progress(transient=True) as progress:
                        task = progress.add_task(
                            f"[cyan]Extracting inner zip...", total=len(files)
                        )
                        for i, f in enumerate(files):
                            inner_zip.extract(f, folder)
                            progress.update(task, completed=i + 1)
                file.unlink()  # Remove inner zip after extraction
