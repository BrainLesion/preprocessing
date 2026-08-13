from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from brainles_preprocessing.utils.zenodo import (
    ZenodoException,
    ZenodoRecord,
    fetch_atlases,
    fetch_synthstrip,
)

# ---- Fixtures ----


@pytest.fixture(autouse=True)
def clear_zenodo_cache():
    """Ensure the process-level ZenodoRecord cache is clean before every test."""
    ZenodoRecord.clear_cache()
    yield
    ZenodoRecord.clear_cache()


@pytest.fixture
def dummy_metadata():
    return {
        "version": "1.2.3",
        "title": "Test Record",
    }


@pytest.fixture
def dummy_archive_url():
    return "https://zenodo.org/record/dummy/archive.zip"


@pytest.fixture
def dummy_zenodo_response(dummy_metadata, dummy_archive_url):
    return dummy_metadata, dummy_archive_url


# ---- Tests for _get_metadata_and_archive_url ----


def test_get_metadata_and_archive_url_success(
    monkeypatch, dummy_metadata, dummy_archive_url
):
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "metadata": dummy_metadata,
        "links": {"archive": dummy_archive_url},
    }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("123", Path("/tmp"), "test")

    metadata, url = record._get_metadata_and_archive_url()

    assert metadata == dummy_metadata
    assert url == dummy_archive_url


def test_get_metadata_and_archive_url_failure(monkeypatch):
    response_mock = MagicMock()
    response_mock.status_code = 404
    response_mock.json.return_value = {
        "message": "The record does not exist.",
        "status": 404,
    }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("invalid", Path("/tmp"), "test")

    with pytest.raises(ZenodoException, match="The record does not exist"):
        record._get_metadata_and_archive_url()


def test_get_metadata_and_archive_url_failure_with_text_body(monkeypatch):
    response_mock = MagicMock()
    response_mock.status_code = 403
    response_mock.json.side_effect = ValueError("No JSON")
    response_mock.text = "Rate limit exceeded. Please try again later."

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("123", Path("/tmp"), "test")

    with pytest.raises(ZenodoException, match="Rate limit exceeded"):
        record._get_metadata_and_archive_url()


@patch(
    "requests.get", side_effect=requests.exceptions.RequestException("Connection error")
)
def test_get_metadata_and_archive_url_connection_error(mock_get):
    record = ZenodoRecord("123", Path("/tmp"), "test")

    assert record._get_metadata_and_archive_url() is None


# ---- Tests for _get_latest_version_folder_name ----


def test_get_latest_version_folder_name(tmp_path):
    folder = tmp_path / "123_v1.2.3"
    folder.mkdir()
    (folder / "dummy.txt").touch()

    record = ZenodoRecord("123", tmp_path, "test")
    result = record._get_latest_version_folder_name(list(tmp_path.glob("*")))

    assert result == "123_v1.2.3"


def test_get_latest_version_folder_name_empty(tmp_path):
    record = ZenodoRecord("123", tmp_path, "test")
    assert record._get_latest_version_folder_name([]) is None


def test_get_latest_version_folder_name_ignores_empty_folder(tmp_path):
    folder = tmp_path / "123_v1.2.3"
    folder.mkdir()
    record = ZenodoRecord("123", tmp_path, "test")
    assert record._get_latest_version_folder_name([folder]) is None


# ---- Tests for fetch() method ----


@patch.object(ZenodoRecord, "_get_metadata_and_archive_url")
@patch.object(ZenodoRecord, "_download")
def test_fetch_downloads_new_if_no_local(
    mock_download, mock_metadata, tmp_path, dummy_zenodo_response
):
    mock_metadata.return_value = dummy_zenodo_response
    mock_download.return_value = tmp_path / "123_v1.2.3"

    record = ZenodoRecord("123", tmp_path, "test")
    result = record.fetch()

    assert result.name == "123_v1.2.3"
    mock_download.assert_called_once()


@patch.object(ZenodoRecord, "_get_metadata_and_archive_url", return_value=None)
def test_fetch_zenodo_unreachable_raises(mock_metadata, tmp_path):
    record = ZenodoRecord("123", tmp_path, "test")

    with pytest.raises(ZenodoException):
        record.fetch()


@patch.object(ZenodoRecord, "_get_metadata_and_archive_url")
@patch.object(ZenodoRecord, "_download")
def test_fetch_skips_if_latest_present(
    mock_download, mock_metadata, tmp_path, dummy_zenodo_response
):
    local_folder = tmp_path / "123_v1.2.3"
    local_folder.mkdir()
    (local_folder / "dummy.txt").touch()

    mock_metadata.return_value = dummy_zenodo_response
    record = ZenodoRecord("123", tmp_path, "test")

    result = record.fetch()
    assert result.name == "123_v1.2.3"
    mock_download.assert_not_called()


@patch.object(ZenodoRecord, "_get_metadata_and_archive_url")
@patch.object(ZenodoRecord, "_download")
def test_fetch_replaces_old_version(
    mock_download, mock_metadata, tmp_path, dummy_zenodo_response
):
    old_folder = tmp_path / "123_v1.0.0"
    old_folder.mkdir()
    (old_folder / "dummy.txt").touch()

    mock_metadata.return_value = dummy_zenodo_response
    mock_download.return_value = tmp_path / "123_v1.2.3"

    record = ZenodoRecord("123", tmp_path, "test")

    result = record.fetch()
    assert result.name == "123_v1.2.3"
    assert not old_folder.exists()
    mock_download.assert_called_once()


# ---- Tests for process-level caching ----


@patch.object(ZenodoRecord, "_fetch_uncached")
def test_fetch_caches_result(mock_fetch_uncached, tmp_path):
    """Second call to fetch() returns cached result without hitting _fetch_uncached."""
    expected_path = tmp_path / "123_v1.2.3"
    mock_fetch_uncached.return_value = expected_path

    record = ZenodoRecord("123", tmp_path, "test")

    result1 = record.fetch()
    result2 = record.fetch()

    assert result1 == expected_path
    assert result2 == expected_path
    # _fetch_uncached should only be called once despite two fetch() calls
    mock_fetch_uncached.assert_called_once()


@patch.object(ZenodoRecord, "_fetch_uncached")
def test_fetch_cache_shared_across_instances(mock_fetch_uncached, tmp_path):
    """Two separate ZenodoRecord instances with the same record_id share the cache."""
    expected_path = tmp_path / "123_v1.2.3"
    mock_fetch_uncached.return_value = expected_path

    record1 = ZenodoRecord("123", tmp_path, "test")
    record2 = ZenodoRecord("123", tmp_path, "other_label")

    result1 = record1.fetch()
    result2 = record2.fetch()

    assert result1 == expected_path
    assert result2 == expected_path
    mock_fetch_uncached.assert_called_once()


@patch.object(ZenodoRecord, "_fetch_uncached")
def test_fetch_cache_different_records_independent(mock_fetch_uncached, tmp_path):
    """Different record_ids each trigger their own fetch."""
    path_a = tmp_path / "aaa_v1.0.0"
    path_b = tmp_path / "bbb_v2.0.0"
    mock_fetch_uncached.side_effect = [path_a, path_b]

    record_a = ZenodoRecord("aaa", tmp_path, "a")
    record_b = ZenodoRecord("bbb", tmp_path, "b")

    result_a = record_a.fetch()
    result_b = record_b.fetch()

    assert result_a == path_a
    assert result_b == path_b
    assert mock_fetch_uncached.call_count == 2


def test_clear_cache_resets_state(tmp_path):
    """clear_cache() forces a fresh Zenodo check on the next fetch() call."""
    with patch.object(ZenodoRecord, "_fetch_uncached") as mock_fetch_uncached:
        expected_path = tmp_path / "123_v1.2.3"
        mock_fetch_uncached.return_value = expected_path

        record = ZenodoRecord("123", tmp_path, "test")
        record.fetch()
        assert mock_fetch_uncached.call_count == 1

        ZenodoRecord.clear_cache()
        record.fetch()
        assert mock_fetch_uncached.call_count == 2


# ---- fetch_atlases and fetch_synthstrip ----


@patch("brainles_preprocessing.utils.zenodo.ZenodoRecord.fetch")
def test_fetch_atlases_calls_fetch(mock_fetch):
    mock_fetch.return_value = Path("/fake/path")
    result = fetch_atlases()
    assert result == Path("/fake/path")
    mock_fetch.assert_called_once()


@patch("brainles_preprocessing.utils.zenodo.ZenodoRecord.fetch")
def test_fetch_synthstrip_calls_fetch(mock_fetch):
    mock_fetch.return_value = Path("/fake/path")
    result = fetch_synthstrip()
    assert result == Path("/fake/path")
    mock_fetch.assert_called_once()
