from __future__ import annotations

import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple

import requests
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

ATLASES_FOLDER = Path(__file__).parent.parent / "registration" / "atlases"
ATLASES_RECORD_ID = "15236131"

SYNTHSTRIP_FOLDER = Path(__file__).parent.parent / "brain_extraction" / "weights"
SYNTHSTRIP_RECORD_ID = "16535633"

# Timeout (seconds) for Zenodo HTTP requests. Without this a stalled connection
# (e.g. behind an authenticating proxy) can hang the process indefinitely.
METADATA_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60


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

    # Process-level cache: maps (record_id, target_dir) -> resolved Path
    _cache: ClassVar[Dict[Tuple[str, str], Path]] = {}

    def __init__(
        self,
        record_id: str,
        target_dir: Path,
        label: str = "asset",
    ):
        self.record_id = record_id
        self.target_dir = target_dir
        self.label = label

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the process-level fetch cache. Primarily intended for testing."""
        cls._cache.clear()

    def _cache_key(self) -> Tuple[str, str]:
        return (self.record_id, str(self.target_dir))

    def fetch(self) -> Path:
        """Fetch the latest version of the record from Zenodo or from local storage.

        Results are cached for the lifetime of the process so that repeated calls
        (e.g. when processing many subjects in a loop) do not trigger redundant
        Zenodo API requests.
        """
        key = self._cache_key()

        cached = ZenodoRecord._cache.get(key)
        if cached is not None:
            logger.debug(f"Using cached {self.label} path: {cached}")
            return cached

        result = self._fetch_uncached()
        ZenodoRecord._cache[key] = result
        return result

    def _fetch_uncached(self) -> Path:
        """Perform the actual Zenodo check / download without consulting the cache."""
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
        local_folder = self.target_dir / latest_local

        if not zenodo_response:
            logger.warning(f"Zenodo unreachable. Using latest downloaded {self.label}.")
            return local_folder

        metadata, archive_url = zenodo_response
        remote_version = metadata["version"]
        local_version = latest_local.split("_v")[1]

        if remote_version == local_version:
            logger.info(f"Latest {self.label} ({remote_version}) already present.")
            return local_folder

        logger.info(
            f"New version of {self.label} available on Zenodo ({remote_version}). Replacing local copy..."
        )
        # Download the new version *before* removing the old one, so a failed
        # upgrade never leaves us without a usable copy.
        try:
            new_folder = self._download(metadata, archive_url)
        except ZenodoException as e:
            logger.warning(
                f"Failed to download {self.label} {remote_version} ({e}). "
                f"Keeping local version {local_version}."
            )
            return local_folder

        if new_folder != local_folder and local_folder.exists():
            shutil.rmtree(
                local_folder,
                onerror=lambda func, path, excinfo: logger.warning(
                    f"Failed to delete {path}: {excinfo}"
                ),
            )
        return new_folder

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
        """Return the name of the newest non-empty version folder, if any.

        Folders that are empty or whose name cannot be parsed as a version are
        skipped rather than treated as "nothing available locally", so a leftover
        empty directory cannot shadow an intact older copy.
        """
        candidates = []
        for folder in folders:
            if not folder.is_dir():
                continue
            try:
                version = tuple(map(int, folder.name.split("_v")[1].split(".")))
            except (IndexError, ValueError):
                logger.debug(f"Ignoring unparsable {self.label} folder: {folder.name}")
                continue
            candidates.append((version, folder))

        for _, folder in sorted(candidates, key=lambda item: item[0], reverse=True):
            if any(folder.iterdir()):
                return folder.name
            logger.warning(f"Ignoring empty {self.label} folder: {folder.name}")

        return None

    def _get_metadata_and_archive_url(self) -> Tuple[Dict, str] | None:
        """Return (metadata, archive_url) or None if Zenodo could not be queried.

        Returning None on *any* failure — including HTTP error statuses such as
        502/504 — lets the caller fall back to a local copy.
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}/{self.record_id}", timeout=METADATA_TIMEOUT
            )
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch metadata from Zenodo: {e}")
            return None

        if response.status_code != 200:
            logger.warning(
                f"Zenodo returned an unexpected status for record "
                f"'{self.record_id}' ({response.status_code=})."
            )
            return None

        try:
            data = response.json()
            return data["metadata"], data["links"]["archive"]
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Unexpected response payload from Zenodo: {e}")
            return None

    def _download(
        self,
        metadata: Dict,
        archive_url: str,
    ) -> Path:
        """Download and extract the record, staging it in a temporary directory.

        The final folder only appears once extraction succeeded, so a failed
        download cannot leave an empty version folder behind.
        """
        folder = self._build_folder_path(metadata["version"])

        logger.info(f"Downloading {self.label} from Zenodo. This may take a while...")

        try:
            response = requests.get(
                archive_url, stream=True, timeout=DOWNLOAD_TIMEOUT
            )
        except requests.exceptions.RequestException as e:
            msg = f"Failed to download {self.label}: {e}"
            logger.error(msg)
            raise ZenodoException(msg) from e

        if response.status_code != 200:
            msg = (
                f"Failed to download {self.label}. Status code: {response.status_code}"
            )
            logger.error(msg)
            raise ZenodoException(msg)

        self.target_dir.mkdir(parents=True, exist_ok=True)
        # Staged inside target_dir so the final move is a rename on the same
        # filesystem. The leading dot keeps it out of _glob_pattern().
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".tmp_{self.record_id}_", dir=self.target_dir)
        )
        try:
            self._extract_archive(response, staging_dir)
            if folder.exists():
                shutil.rmtree(
                    folder,
                    onerror=lambda func, path, excinfo: logger.warning(
                        f"Failed to delete {path}: {excinfo}"
                    ),
                )
            staging_dir.replace(folder)
        except Exception as e:
            shutil.rmtree(staging_dir, ignore_errors=True)
            if isinstance(e, ZenodoException):
                raise
            msg = f"Failed to extract {self.label}: {e}"
            logger.error(msg)
            raise ZenodoException(msg) from e

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
