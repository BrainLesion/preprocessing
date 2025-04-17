import pytest
import shutil
import zipfile
from unittest.mock import MagicMock, patch
from pathlib import Path
from io import BytesIO

from requests import RequestException

from brainles_preprocessing.utils.zenodo import (
    ATLASES_FOLDER,
    verify_or_download_atlases,
    _get_latest_version_folder_name,
    _get_zenodo_metadata_and_archive_url,
    _download_atlases,
    _extract_archive,
)


@pytest.fixture
def mock_zenodo_metadata():
    return {"version": "1.0.0"}, "https://fakeurl.com/archive.zip"


@pytest.fixture
def mock_atlases_folder(tmp_path):
    atlases_path = tmp_path / "atlases"
    atlases_path.mkdir()
    return atlases_path


@patch("brainles_preprocessing.utils.zenodo.requests.get")
def test_get_zenodo_metadata_and_archive_url(mock_get, mock_zenodo_metadata):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "metadata": mock_zenodo_metadata[0],
        "links": {"archive": mock_zenodo_metadata[1]},
    }
    mock_get.return_value = mock_response

    metadata, archive_url = _get_zenodo_metadata_and_archive_url()
    assert metadata["version"] == "1.0.0"
    assert archive_url == "https://fakeurl.com/archive.zip"


@patch("brainles_preprocessing.utils.zenodo.requests.get")
def test_get_zenodo_metadata_and_archive_url_failure(mock_get):
    mock_get.side_effect = RequestException()
    assert _get_zenodo_metadata_and_archive_url() == None


@patch(
    "brainles_preprocessing.utils.zenodo._get_latest_version_folder_name",
    return_value=None,
)
@patch(
    "brainles_preprocessing.utils.zenodo._get_zenodo_metadata_and_archive_url",
    return_value=None,
)
@patch("brainles_preprocessing.utils.zenodo.logger.error")
def test_verify_or_download_atlases_no_local_no_meta(
    mock_sys_exit, mock_get_meta, mock_get_latest_version
):
    with pytest.raises(SystemExit):
        verify_or_download_atlases()
    mock_sys_exit.assert_called_once_with(
        "Atlases not found locally and Zenodo could not be reached. Exiting..."
    )


@patch(
    "brainles_preprocessing.utils.zenodo._get_latest_version_folder_name",
    return_value=None,
)
@patch("brainles_preprocessing.utils.zenodo._get_zenodo_metadata_and_archive_url")
@patch("brainles_preprocessing.utils.zenodo._download_atlases")
def test_verify_or_download_atlases_no_local(
    mock_download, mock_zenodo_meta, mock_atlases_folder
):
    mock_zenodo_meta.return_value = ({"version": "1.0.0"}, "https://fakeurl.com")
    mock_download.return_value = mock_atlases_folder / "atlases_v1.0.0"

    atlases_path = verify_or_download_atlases()
    assert atlases_path == mock_atlases_folder / "atlases_v1.0.0"


@patch(
    "brainles_preprocessing.utils.zenodo._get_latest_version_folder_name",
    return_value="atlases_v1.0.0",
)
@patch("brainles_preprocessing.utils.zenodo.logger.info")
@patch("brainles_preprocessing.utils.zenodo._get_zenodo_metadata_and_archive_url")
def test_verify_or_download_atlases_latest_local(
    mock_zenodo_meta, mock_logger_info, mock_atlases_folder
):
    mock_zenodo_meta.return_value = ({"version": "1.0.0"}, "https://fakeurl.com")

    atlases_path = verify_or_download_atlases()
    assert atlases_path == ATLASES_FOLDER / "atlases_v1.0.0"
    mock_logger_info.assert_called_with(f"Latest atlases (1.0.0) are already present.")


@patch("brainles_preprocessing.utils.zenodo.shutil.rmtree")
@patch(
    "brainles_preprocessing.utils.zenodo._get_latest_version_folder_name",
    return_value="atlases_v1.0.0",
)
@patch("brainles_preprocessing.utils.zenodo.logger.info", return_value=None)
@patch("brainles_preprocessing.utils.zenodo._get_zenodo_metadata_and_archive_url")
@patch("brainles_preprocessing.utils.zenodo._download_atlases")
def test_verify_or_download_atlases_old_local(
    mock_download,
    mock_zenodo_meta,
    mock_logger_info,
    mock_shutil_rmtree,
    mock_atlases_folder,
):
    mock_zenodo_meta.return_value = ({"version": "2.0.0"}, "https://fakeurl.com")
    mock_download.return_value = mock_atlases_folder / "atlases_v2.0.0"

    atlases_path = verify_or_download_atlases()

    mock_logger_info.assert_called_with(
        "New atlases available on Zenodo (2.0.0). Deleting old and fetching new atlases..."
    )

    mock_shutil_rmtree.assert_called_once()

    assert atlases_path == mock_atlases_folder / "atlases_v2.0.0"


@patch("brainles_preprocessing.utils.zenodo._extract_archive")
@patch("brainles_preprocessing.utils.zenodo.requests.get")
def test_download_atlases(mock_get, mock_extract_archive, mock_zenodo_metadata):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content = lambda chunk_size: [b"data"]
    mock_get.return_value = mock_response

    atlases_path = _download_atlases(mock_zenodo_metadata[0], mock_zenodo_metadata[1])
    assert atlases_path.exists()


@patch("brainles_preprocessing.utils.zenodo.zipfile.ZipFile")
def test_extract_archive(mock_zipfile, tmp_path):
    mock_response = MagicMock()
    mock_response.iter_content = lambda chunk_size: [b"data"]
    record_folder = tmp_path / "atlases_v1.0.0"
    record_folder.mkdir()

    dummy_zip = record_folder / "archive.zip"
    dummy_zip.touch()

    mock_zip = MagicMock()
    mock_zip.namelist.return_value = ["file1.txt", "file2.txt"]
    mock_zip.__enter__.return_value = mock_zip
    mock_zipfile.return_value = mock_zip

    _extract_archive(mock_response, record_folder)

    mock_zip.extract.assert_called()
