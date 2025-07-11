from types import SimpleNamespace

import pytest

from brainles_preprocessing.modality import CenterModality, Modality
from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor


# --- Dummy classes for testing ---
class DummyModality:
    def __init__(self, name):
        self.modality_name = name


class DummyCenterModality(DummyModality):
    pass


@pytest.fixture
def dummy_registrator():
    return SimpleNamespace()


@pytest.fixture
def dummy_brain_extractor():
    return SimpleNamespace()


@pytest.fixture
def dummy_defacer():
    return SimpleNamespace()


# --- Tests ---


def test_no_name_conflicts(dummy_registrator, dummy_brain_extractor, dummy_defacer):
    center = CenterModality("T1", input_path="", raw_skull_output_path="tmp")
    moving = [
        Modality("T2", input_path="", raw_skull_output_path="tmp"),
        Modality("FLAIR", input_path="", raw_skull_output_path="tmp"),
    ]
    # Should not raise
    AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=moving,
        registrator=dummy_registrator,
        brain_extractor=dummy_brain_extractor,
        defacer=dummy_defacer,
    )


def test_single_duplicate_name_raises(
    dummy_registrator, dummy_brain_extractor, dummy_defacer
):
    center = CenterModality("T1", input_path="", raw_skull_output_path="tmp")
    moving = [
        Modality("T1", input_path="", raw_skull_output_path="tmp"),  # Duplicate name
        Modality("FLAIR", input_path="", raw_skull_output_path="tmp"),
    ]

    with pytest.raises(ValueError, match=r"Duplicate modality names found: T1"):
        AtlasCentricPreprocessor(
            center_modality=center,
            moving_modalities=moving,
            registrator=dummy_registrator,
            brain_extractor=dummy_brain_extractor,
            defacer=dummy_defacer,
        )


def test_multiple_duplicate_names_raises(
    dummy_registrator, dummy_brain_extractor, dummy_defacer
):
    center = CenterModality("T1", input_path="", raw_skull_output_path="tmp")
    moving = [
        Modality("T1", input_path="", raw_skull_output_path="tmp"),  # Duplicate
        Modality("FLAIR", input_path="", raw_skull_output_path="tmp"),
        Modality("FLAIR", input_path="", raw_skull_output_path="tmp"),  # Duplicate
    ]

    with pytest.raises(ValueError) as exc_info:
        AtlasCentricPreprocessor(
            center_modality=center,
            moving_modalities=moving,
            registrator=dummy_registrator,
            brain_extractor=dummy_brain_extractor,
            defacer=dummy_defacer,
        )

    # Check that all duplicates are reported in the error message
    error_msg = str(exc_info.value)
    assert "T1" in error_msg
    assert "FLAIR" in error_msg
