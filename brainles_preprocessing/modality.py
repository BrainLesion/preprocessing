# todo add typing and docs
import os
import shutil

from auxiliary.nifti.io import read_nifti, write_nifti
from auxiliary.normalization.normalizer_base import Normalizer
from auxiliary.turbopath import turbopath

from brainles_preprocessing.registration.registrator import Registrator


class Modality:
    """
    Represents a medical image modality with associated preprocessing information.

    Args:
        modality_name (str): Name of the modality, e.g., "T1", "T2", "FLAIR".
        input_path (str): Path to the input modality data.
        output_path (str): Path to save the preprocessed modality data.
        bet (bool): Indicates whether brain extraction should be performed (True) or not (False).
        normalizer (Normalizer, optional): An optional normalizer for intensity normalization.

    Attributes:
        modality_name (str): Name of the modality.
        input_path (str): Path to the input modality data.
        output_path (str): Path to save the preprocessed modality data.
        bet (bool): Indicates whether brain extraction is enabled.
        normalizer (Normalizer, optional): An optional normalizer for intensity normalization.

    Example:
        >>> t1_modality = Modality(
        ...     modality_name="T1",
        ...     input_path="/path/to/input_t1.nii",
        ...     output_path="/path/to/preprocessed_t1.nii",
        ...     bet=True
        ... )
    """

    def __init__(
        self,
        modality_name: str,
        input_path: str,
        output_path: str,
        bet: bool,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.modality_name = modality_name
        self.input_path = turbopath(input_path)
        self.output_path = turbopath(output_path)
        self.bet = bet
        self.normalizer = normalizer
        self.current = self.input_path

    def normalize(
        self,
        temporary_directory,
        store_unnormalized=None,
    ):
        # Backup the unnormalized file
        if store_unnormalized is not None:
            os.makedirs(store_unnormalized, exist_ok=True)
            shutil.copyfile(
                src=self.current,
                dst=f"{store_unnormalized}/unnormalized__{self.modality_name}.nii.gz",
            )

        if temporary_directory is not None:
            unnormalized_dir = f"{temporary_directory}/unnormalized"
            os.makedirs(unnormalized_dir, exist_ok=True)
            shutil.copyfile(
                src=self.current,
                dst=f"{unnormalized_dir}/unnormalized__{self.modality_name}.nii.gz",
            )

        # Normalize the image
        if self.normalizer is not None:
            image = read_nifti(self.current)
            normalized_image = self.normalizer.normalize(image=image)
            write_nifti(
                input_array=normalized_image,
                output_nifti_path=self.current,
                reference_nifti_path=self.current,
            )

    def register(
        self,
        registrator,
        fixed_image_path: str,
        registration_dir: str,
        moving_image_name: str,
    ):
        registered = os.path.join(registration_dir, f"{moving_image_name}.nii.gz")
        registered_matrix = os.path.join(registration_dir, f"{moving_image_name}.txt")
        registered_log = os.path.join(registration_dir, f"{moving_image_name}.log")

        registrator.register(
            fixed_image=fixed_image_path,
            moving_image=self.current,
            transformed_image=registered,
            matrix=registered_matrix,
            log_file=registered_log,
        )
        self.current = registered
        return registered_matrix

    def apply_mask(
        self,
        brain_extractor,
        brain_masked_dir,
        atlas_mask,
    ):
        if self.bet:
            brain_masked = os.path.join(
                brain_masked_dir,
                f"brain_masked__{self.modality_name}.nii.gz",
            )
            brain_extractor.apply_mask(
                input_image_path=self.current,
                mask_image_path=atlas_mask,
                masked_image_path=brain_masked,
            )
            self.current = brain_masked

    def transform(
        self,
        registrator: Registrator,
        fixed_image_path,
        registration_dir,
        moving_image_name,
        transformation_matrix,
    ):
        transformed = os.path.join(registration_dir, f"{moving_image_name}.nii.gz")
        transformed_log = os.path.join(registration_dir, f"{moving_image_name}.log")

        registrator.transform(
            fixed_image=fixed_image_path,
            moving_image=self.current,
            transformed_image=transformed,
            matrix=transformation_matrix,
            log_file=transformed_log,
        )
        self.current = transformed

    def extract_brain_region(
        self,
        brain_extractor,
        bet_dir,
    ):
        bet_log = os.path.join(bet_dir, "brain-extraction.log")
        atlas_bet_cm = os.path.join(bet_dir, f"atlas_bet_{self.modality_name}.nii.gz")
        atlas_mask = os.path.join(
            bet_dir, f"atlas_bet_{self.modality_name}_mask.nii.gz"
        )

        brain_extractor.extract(
            input_image_path=self.current,
            masked_image_path=atlas_bet_cm,
            brain_mask_path=atlas_mask,
            log_file_path=bet_log,
        )
        self.current = atlas_bet_cm
        return atlas_mask
