from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch
import zipfile

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
def dummy_files():
    return [
        {
            "key": "dummy.nii",
            "links": {
                "self": "https://zenodo.org/record/dummy/files/dummy.nii/content"
            },
        }
    ]


@pytest.fixture
def dummy_zenodo_response(dummy_metadata, dummy_files):
    return dummy_metadata, dummy_files


# ---- Tests for _get_metadata_and_files ----


def test_get_metadata_and_files_success(monkeypatch, dummy_metadata, dummy_files):
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "metadata": dummy_metadata,
        "files": dummy_files,
    }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("123", Path("/tmp"), "test")

    metadata, files = record._get_metadata_and_files()

    assert metadata == dummy_metadata
    assert files == dummy_files


def test_get_metadata_and_files_falls_back_to_archive(monkeypatch, dummy_metadata):
    archive_url = "https://zenodo.org/record/dummy/archive.zip"
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "metadata": dummy_metadata,
        "links": {"archive": archive_url},
    }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("123", Path("/tmp"), "test")

    metadata, files = record._get_metadata_and_files()

    assert metadata == dummy_metadata
    assert files[0]["links"]["self"] == archive_url


def test_get_metadata_and_files_no_files_returns_none(monkeypatch, dummy_metadata):
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {"metadata": dummy_metadata}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("123", Path("/tmp"), "test")

    assert record._get_metadata_and_files() is None


def test_get_metadata_and_files_malformed_response_returns_none(monkeypatch):
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {"unexpected": "shape"}

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("123", Path("/tmp"), "test")

    assert record._get_metadata_and_files() is None


def test_get_metadata_and_files_failure(monkeypatch):
    response_mock = MagicMock()
    response_mock.status_code = 404

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("invalid", Path("/tmp"), "test")

    with pytest.raises(ZenodoException):
        record._get_metadata_and_files()


@patch(
    "requests.get", side_effect=requests.exceptions.RequestException("Connection error")
)
def test_get_metadata_and_files_connection_error(mock_get):
    record = ZenodoRecord("123", Path("/tmp"), "test")

    assert record._get_metadata_and_files() is None


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


@patch.object(ZenodoRecord, "_get_metadata_and_files")
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


@patch.object(ZenodoRecord, "_get_metadata_and_files", return_value=None)
def test_fetch_zenodo_unreachable_raises(mock_metadata, tmp_path):
    record = ZenodoRecord("123", tmp_path, "test")

    with pytest.raises(ZenodoException):
        record.fetch()


@patch.object(ZenodoRecord, "_get_metadata_and_files")
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


@patch.object(ZenodoRecord, "_get_metadata_and_files")
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
    mock_download.assert_called_once()
    # Removal of the old version is delegated to _download (atomic replace).
    assert mock_download.call_args.kwargs["stale_folder"] == old_folder


# ---- Tests for downloading record files ----


def _fake_response(content: bytes, status_code: int = 200):
    response = MagicMock()
    response.status_code = status_code
    response.iter_content.return_value = [content]
    return response


def test_download_writes_individual_files(tmp_path):
    files = [
        {"key": "a.nii", "links": {"self": "https://example.org/a.nii"}},
        {"key": "sub/b.nii", "links": {"self": "https://example.org/b.nii"}},
    ]
    responses = [_fake_response(b"AAAA"), _fake_response(b"BBBB")]
    with patch("requests.get", side_effect=responses):
        record = ZenodoRecord("123", tmp_path, "test")
        folder = record._download({"version": "1.2.3"}, files)

    assert folder == tmp_path / "123_v1.2.3"
    assert (folder / "a.nii").read_bytes() == b"AAAA"
    assert (folder / "sub" / "b.nii").read_bytes() == b"BBBB"


def test_download_extracts_zip_file(tmp_path):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("inner.txt", "hello")
    zip_bytes = buffer.getvalue()

    files = [{"key": "weights.zip", "links": {"self": "https://example.org/w.zip"}}]
    with patch("requests.get", return_value=_fake_response(zip_bytes)):
        record = ZenodoRecord("123", tmp_path, "test")
        folder = record._download({"version": "1.0.0"}, files)

    assert (folder / "inner.txt").read_text() == "hello"
    assert not (folder / "weights.zip").exists()


def test_download_file_raises_on_error(tmp_path):
    response = _fake_response(b"", status_code=500)
    with patch("requests.get", return_value=response):
        record = ZenodoRecord("123", tmp_path, "test")
        with pytest.raises(ZenodoException):
            record._download_file("https://example.org/a.nii", tmp_path / "a.nii")


def test_download_file_request_exception_raises_zenodo_exception(tmp_path):
    with patch(
        "requests.get",
        side_effect=requests.exceptions.RequestException("connection reset"),
    ):
        record = ZenodoRecord("123", tmp_path, "test")
        with pytest.raises(ZenodoException):
            record._download_file("https://example.org/a.nii", tmp_path / "a.nii")


def test_download_extracts_nested_zip(tmp_path):
    inner = BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("deep.txt", "deep")
    outer = BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("nested.zip", inner.getvalue())

    files = [{"key": "outer.zip", "links": {"self": "https://example.org/o.zip"}}]
    with patch("requests.get", return_value=_fake_response(outer.getvalue())):
        record = ZenodoRecord("123", tmp_path, "test")
        folder = record._download({"version": "1.0.0"}, files)

    assert (folder / "deep.txt").read_text() == "deep"
    assert not list(folder.rglob("*.zip"))


def test_download_failure_keeps_previous_version_and_cleans_temp(tmp_path):
    old_folder = tmp_path / "123_v1.0.0"
    old_folder.mkdir()
    (old_folder / "keep.nii").write_text("old")

    files = [{"key": "a.nii", "links": {"self": "https://example.org/a.nii"}}]
    with patch("requests.get", return_value=_fake_response(b"", status_code=500)):
        record = ZenodoRecord("123", tmp_path, "test")
        with pytest.raises(ZenodoException):
            record._download({"version": "1.2.3"}, files, stale_folder=old_folder)

    # Previous version is untouched and no partial/temp installs remain.
    assert (old_folder / "keep.nii").read_text() == "old"
    assert not (tmp_path / "123_v1.2.3").exists()
    assert not list(tmp_path.glob(".123_v1.2.3.tmp.*"))


def test_download_replaces_stale_version_on_success(tmp_path):
    old_folder = tmp_path / "123_v1.0.0"
    old_folder.mkdir()
    (old_folder / "keep.nii").write_text("old")

    files = [{"key": "a.nii", "links": {"self": "https://example.org/a.nii"}}]
    with patch("requests.get", return_value=_fake_response(b"AAAA")):
        record = ZenodoRecord("123", tmp_path, "test")
        folder = record._download({"version": "1.2.3"}, files, stale_folder=old_folder)

    assert not old_folder.exists()
    assert (folder / "a.nii").read_bytes() == b"AAAA"
    assert not list(tmp_path.glob(".123_v1.2.3.tmp.*"))


@pytest.mark.parametrize("unsafe_key", ["../escape.nii", "/tmp/escape.nii"])
def test_download_rejects_unsafe_key(tmp_path, unsafe_key):
    files = [{"key": unsafe_key, "links": {"self": "https://example.org/a.nii"}}]
    with patch("requests.get", return_value=_fake_response(b"AAAA")):
        record = ZenodoRecord("123", tmp_path, "test")
        with pytest.raises(ZenodoException):
            record._download({"version": "1.2.3"}, files)

    assert not (tmp_path / "escape.nii").exists()
    assert not (tmp_path.parent / "escape.nii").exists()
    assert not (tmp_path / "123_v1.2.3").exists()
    assert not list(tmp_path.glob(".123_v1.2.3.tmp.*"))


def test_download_extract_rejects_unsafe_zip_member(tmp_path):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("../escape.txt", "evil")

    files = [{"key": "evil.zip", "links": {"self": "https://example.org/e.zip"}}]
    with patch("requests.get", return_value=_fake_response(buffer.getvalue())):
        record = ZenodoRecord("123", tmp_path, "test")
        with pytest.raises(ZenodoException):
            record._download({"version": "1.2.3"}, files)

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path.parent / "escape.txt").exists()
    assert not list(tmp_path.glob(".123_v1.2.3.tmp.*"))


def test_download_keeps_stale_version_if_install_fails(tmp_path):
    old_folder = tmp_path / "123_v1.0.0"
    old_folder.mkdir()
    (old_folder / "keep.nii").write_text("old")

    files = [{"key": "a.nii", "links": {"self": "https://example.org/a.nii"}}]
    with patch("requests.get", return_value=_fake_response(b"AAAA")), patch(
        "pathlib.Path.rename", side_effect=OSError("boom")
    ):
        record = ZenodoRecord("123", tmp_path, "test")
        with pytest.raises(OSError):
            record._download({"version": "1.2.3"}, files, stale_folder=old_folder)

    # The known-good version must survive a failed install.
    assert (old_folder / "keep.nii").read_text() == "old"
    assert not (tmp_path / "123_v1.2.3").exists()
    assert not list(tmp_path.glob(".123_v1.2.3.tmp.*"))


def test_get_metadata_and_files_invalid_file_descriptor_returns_none(
    monkeypatch, dummy_metadata
):
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "metadata": dummy_metadata,
        "files": [{"key": "a.nii"}],  # missing links.self
    }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("123", Path("/tmp"), "test")

    assert record._get_metadata_and_files() is None


def test_get_metadata_and_files_missing_version_returns_none(monkeypatch):
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = {
        "metadata": {"title": "no version"},
        "files": [{"key": "a.nii", "links": {"self": "https://example.org/a.nii"}}],
    }

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response_mock)
    record = ZenodoRecord("123", Path("/tmp"), "test")

    assert record._get_metadata_and_files() is None


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
