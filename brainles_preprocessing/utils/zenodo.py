from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple

import requests
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

# (connect, read) timeout in seconds for every request to Zenodo.
REQUEST_TIMEOUT = (10, 60)

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
        zenodo_response = self._get_metadata_and_files()

        pattern = self._glob_pattern()
        matching_folders = list(self.target_dir.glob(pattern))
        latest_local = self._get_latest_version_folder_name(matching_folders)

        if not latest_local:
            if not zenodo_response:
                msg = f"{self.label.title()} not found locally and Zenodo could not be reached."
                logger.error(msg)
                raise ZenodoException(msg)

            logger.info(f"{self.label.title()} not found locally.")
            metadata, files = zenodo_response
            return self._download(metadata, files)

        logger.info(f"Found local {self.label}: {latest_local}")

        if not zenodo_response:
            logger.warning(f"Zenodo unreachable. Using latest downloaded {self.label}.")
            return self.target_dir / latest_local

        metadata, files = zenodo_response
        remote_version = metadata["version"]
        local_version = latest_local.split("_v")[1]

        if remote_version == local_version:
            logger.info(f"Latest {self.label} ({remote_version}) already present.")
            return self.target_dir / latest_local

        logger.info(
            f"New version of {self.label} available on Zenodo ({remote_version}). Replacing local copy..."
        )
        # The old version is only removed once the new one has been fully
        # downloaded and staged, so a failed download never destroys a
        # known-good local copy.
        return self._download(
            metadata,
            files,
            stale_folder=self.target_dir / latest_local,
        )

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

    def _get_metadata_and_files(self) -> Tuple[Dict, List[Dict]] | None:
        """Retrieve the record metadata and its file list from Zenodo.

        Zenodo's API used to expose a single ``links["archive"]`` URL pointing to
        a zip of all files. That link has been removed, so the record's ``files``
        list is used instead: each entry provides a ``key`` and a ``links["self"]``
        download URL.

        Returns:
            A ``(metadata, files)`` tuple, or ``None`` if Zenodo is unreachable or
            returns an unexpected response.
        """
        try:
            response = requests.get(
                f"{self.BASE_URL}/{self.record_id}", timeout=REQUEST_TIMEOUT
            )
            if response.status_code != 200:
                error_msg = (
                    f"Cannot find record '{self.record_id}' on Zenodo "
                    f"({response.status_code=})."
                )
                logger.error(error_msg)
                raise ZenodoException(error_msg)

            data = response.json()
            metadata = data["metadata"]
            if not isinstance(metadata, dict) or not isinstance(
                metadata.get("version"), str
            ):
                raise ValueError("record metadata is missing a string 'version'")

            files = data.get("files")
            if files:
                for entry in files:
                    if (
                        not isinstance(entry, dict)
                        or not isinstance(entry.get("key"), str)
                        or not isinstance(entry.get("links", {}).get("self"), str)
                    ):
                        raise ValueError(
                            f"invalid file descriptor in record: {entry!r}"
                        )
                return metadata, files

            # Fallback for the deprecated archive link (kept for compatibility).
            archive_url = data.get("links", {}).get("archive")
            if isinstance(archive_url, str):
                return metadata, [
                    {"key": f"{self.record_id}.zip", "links": {"self": archive_url}}
                ]

            logger.warning(
                f"Zenodo record '{self.record_id}' does not expose any downloadable files."
            )
            return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch metadata from Zenodo: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.warning(
                f"Unexpected Zenodo response for record '{self.record_id}': {e}"
            )
            return None

    def _download(
        self,
        metadata: Dict,
        files: List[Dict],
        stale_folder: Path | None = None,
    ) -> Path:
        folder = self._build_folder_path(metadata["version"])
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Stage everything in a temporary directory first, so a failed download
        # never leaves a partial install or destroys an existing good version.
        temp_folder = Path(
            tempfile.mkdtemp(dir=self.target_dir, prefix=f".{folder.name}.tmp.")
        )

        logger.info(f"Downloading {self.label} from Zenodo. This may take a while...")

        try:
            for file in files:
                key = file["key"]
                file_url = file["links"]["self"]
                destination = temp_folder / key
                self._download_file(file_url, destination)

                if destination.suffix == ".zip":
                    self._extract_zip(destination, temp_folder)

            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            if (
                stale_folder is not None
                and stale_folder != folder
                and stale_folder.exists()
            ):
                shutil.rmtree(
                    stale_folder,
                    onerror=lambda func, path, excinfo: logger.warning(
                        f"Failed to delete {path}: {excinfo}"
                    ),
                )

            temp_folder.rename(folder)
        except BaseException:
            shutil.rmtree(temp_folder, ignore_errors=True)
            raise

        logger.info(f"{self.label.title()} downloaded to {folder}")
        return folder

    def _download_file(self, url: str, destination: Path) -> None:
        chunk_size = 1024 * 1024  # 1 MiB
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                msg = (
                    f"Failed to download {self.label} file '{destination.name}'. "
                    f"Status code: {response.status_code}"
                )
                logger.error(msg)
                raise ZenodoException(msg)

            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]Downloading {destination.name}..."),
                TextColumn("[cyan]{task.completed:.2f} MB"),
                transient=True,
            ) as progress:
                task = progress.add_task("", total=None)
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        progress.update(task, advance=len(chunk) / (1024**2))
        except requests.exceptions.RequestException as e:
            msg = f"Failed to download {self.label} file '{destination.name}': {e}"
            logger.error(msg)
            raise ZenodoException(msg) from e

    def _extract_zip(
        self,
        zip_path: Path,
        folder: Path,
    ) -> None:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            with Progress(transient=True) as progress:
                task = progress.add_task(
                    f"[cyan]Extracting {zip_path.name}...", total=len(names)
                )
                for i, name in enumerate(names):
                    z.extract(name, folder)
                    progress.update(task, completed=i + 1)
        zip_path.unlink()  # Remove the archive after extraction

        # Extract any archives nested inside this one (e.g. a record whose
        # archive contains the SynthStrip weights zip).
        for nested in folder.rglob("*.zip"):
            if nested.is_file():
                self._extract_zip(nested, folder)
