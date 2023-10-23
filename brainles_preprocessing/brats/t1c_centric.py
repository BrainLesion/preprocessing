from brainles_preprocessing.core import (
    preprocess_modality_centric_to_atlas_space,
    Modality,
)


def preprocess_brats_style_t1c_centric(
    input_t1c: str,
    output_t1c: str,
    input_t1: str,
    output_t1: str,
    input_t2: str,
    output_t2: str,
    input_flair: str,
    output_flair: str,
    bet_mode: str = "gpu",
    limit_cuda_visible_devices: str | None = None,
    keep_coregistration: str | None = None,
    keep_atlas_registration: str | None = None,
    keep_brainextraction: str | None = None,
) -> None:
    """
    Preprocesses multiple modalities in a BRATS-style dataset to atlas space.

    Args:
        input_t1c (str): Path to the input T1c modality data.
        output_t1c (str): Path to save the preprocessed T1c modality data.
        input_t1 (str): Path to the input T1 modality data.
        output_t1 (str): Path to save the preprocessed T1 modality data.
        input_t2 (str): Path to the input T2 modality data.
        output_t2 (str): Path to save the preprocessed T2 modality data.
        input_flair (str): Path to the input FLAIR modality data.
        output_flair (str): Path to save the preprocessed FLAIR modality data.
        bet_mode (str, optional): The mode for brain extraction, e.g., "gpu".
        limit_cuda_visible_devices (str | None, optional): Specify CUDA devices to use.
        keep_coregistration (str | None, optional): Specify if coregistration should be retained.
        keep_atlas_registration (str | None, optional): Specify if atlas registration should be retained.
        keep_brainextraction (str | None, optional): Specify if brain extraction should be retained.
    """
    # Create a Modality object for the primary T1c modality
    primary = Modality(
        modality_name="t1c",
        input_path=input_t1c,
        output_path=output_t1c,
        bet=True,
    )

    # Create Modality objects for other moving modalities
    moving_modalities = [
        Modality(
            modality_name="t1",
            input_path=input_t1,
            output_path=output_t1,
            bet=True,
        ),
        Modality(
            modality_name="t2",
            input_path=input_t2,
            output_path=output_t2,
            bet=True,
        ),
        Modality(
            modality_name="flair",
            input_path=input_flair,
            output_path=output_flair,
            bet=True,
        ),
    ]

    # Perform preprocessing to align modalities to the atlas space
    preprocess_modality_centric_to_atlas_space(
        center_modality=primary,
        moving_modalities=moving_modalities,
        bet_mode=bet_mode,
        limit_cuda_visible_devices=limit_cuda_visible_devices,
        keep_coregistration=keep_coregistration,
        keep_atlas_registration=keep_atlas_registration,
        keep_brainextraction=keep_brainextraction,
    )
