import logging
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

from auxiliary.nifti.io import read_nifti, write_nifti
from brainles_preprocessing.brain_extraction.brain_extractor import BrainExtractor
from brainles_preprocessing.constants import PreprocessorSteps
from brainles_preprocessing.defacing import Defacer, QuickshearDefacer
from brainles_preprocessing.normalization.normalizer_base import Normalizer
from brainles_preprocessing.registration.registrator import Registrator

logger = logging.getLogger(__name__)


class Modality:
    """
    Represents a medical image modality with associated preprocessing information.

    Args:
        modality_name (str): Name of the modality, e.g., "T1", "T2", "FLAIR".
        input_path (str or Path): Path to the input modality data.
        normalizer (Normalizer, optional): An optional normalizer for intensity normalization.
        raw_bet_output_path (str or Path, optional): Path to save the raw brain extracted modality data.
        raw_skull_output_path (str or Path, optional): Path to save the raw modality data with skull.
        raw_defaced_output_path (str or Path, optional): Path to save the raw defaced modality data.
        normalized_bet_output_path (str or Path, optional): Path to save the normalized brain extracted modality data. Requires a normalizer.
        normalized_skull_output_path (str or Path, optional): Path to save the normalized modality data with skull. Requires a normalizer.
        normalized_defaced_output_path (str or Path, optional): Path to save the normalized defaced modality data. Requires a normalizer.
        atlas_correction (bool, optional): Indicates whether atlas correction should be performed.

    Attributes:
        modality_name (str): Name of the modality.
        input_path (str or Path): Path to the input modality data.
        normalizer (Normalizer, optional): An optional normalizer for intensity normalization.
        raw_bet_output_path (str or Path, optional): Path to save the raw brain extracted modality data.
        raw_skull_output_path (str or Path, optional): Path to save the raw modality data with skull.
        raw_defaced_output_path (str or Path, optional): Path to save the raw defaced modality data.
        normalized_bet_output_path (str or Path, optional): Path to save the normalized brain extracted modality data. Requires a normalizer.
        normalized_skull_output_path (str or Path, optional): Path to save the normalized modality data with skull. Requires a normalizer.
        normalized_defaced_output_path (str or Path, optional): Path to save the normalized defaced modality data. Requires a normalizer.
        bet (bool): Indicates whether brain extraction is enabled.
        atlas_correction (bool): Indicates whether atlas correction should be performed.

    Example:
        >>> t1_modality = Modality(
        ...     modality_name="T1",
        ...     input_path="/path/to/input_t1.nii",
        ...     normalizer=PercentileNormalizer(),
        ...     raw_bet_output_path="/path/to/raw_bet_t1.nii",
        ...     normalized_bet_output_path="/path/to/norm_bet_t1.nii",
        ... )

    """

    def __init__(
        self,
        modality_name: str,
        input_path: Union[str, Path],
        normalizer: Optional[Normalizer] = None,
        raw_bet_output_path: Optional[Union[str, Path]] = None,
        raw_skull_output_path: Optional[Union[str, Path]] = None,
        raw_defaced_output_path: Optional[Union[str, Path]] = None,
        normalized_bet_output_path: Optional[Union[str, Path]] = None,
        normalized_skull_output_path: Optional[Union[str, Path]] = None,
        normalized_defaced_output_path: Optional[Union[str, Path]] = None,
        atlas_correction: bool = True,
    ) -> None:
        # Basics
        self.modality_name = modality_name
        self.input_path = Path(input_path)
        self.current = self.input_path
        self.normalizer = normalizer
        self.atlas_correction = atlas_correction

        # Check that atleast one output is generated
        if not any(
            [
                raw_bet_output_path,
                normalized_bet_output_path,
                raw_skull_output_path,
                normalized_skull_output_path,
                raw_defaced_output_path,
                normalized_defaced_output_path,
            ]
        ):
            raise ValueError(
                "All output paths are None. At least one output paths must be provided."
            )

        # handle input paths
        self.raw_bet_output_path = (
            Path(raw_bet_output_path) if raw_bet_output_path else None
        )
        self.raw_skull_output_path = (
            Path(raw_skull_output_path) if raw_skull_output_path else None
        )
        self.raw_defaced_output_path = (
            Path(raw_defaced_output_path) if raw_defaced_output_path else None
        )

        if normalized_bet_output_path:
            if normalizer is None:
                raise ValueError(
                    "A normalizer must be provided if normalized_bet_output_path is not None."
                )
            self.normalized_bet_output_path = Path(normalized_bet_output_path)
        else:
            self.normalized_bet_output_path = None

        if normalized_skull_output_path:
            if normalizer is None:
                raise ValueError(
                    "A normalizer must be provided if normalized_skull_output_path is not None."
                )
            self.normalized_skull_output_path = Path(normalized_skull_output_path)
        else:
            self.normalized_skull_output_path = None

        if normalized_defaced_output_path is not None:
            if normalizer is None:
                raise ValueError(
                    "A normalizer must be provided if normalized_defaced_output_path is not None."
                )
            self.normalized_defaced_output_path = Path(normalized_defaced_output_path)
        else:
            self.normalized_defaced_output_path = None

        self.steps = {k: None for k in PreprocessorSteps}

    @property
    def bet(self) -> bool:
        """
        Check if any brain extraction output is specified.

        Returns:
            bool: True if any brain extraction output is specified, False otherwise.
        """
        return any([self.raw_bet_output_path, self.normalized_bet_output_path])

    @property
    def requires_deface(self) -> bool:
        """
        Check if any defacing output is specified.

        Returns:
            bool: True if any defacing output is specified, False otherwise.
        """
        return any([self.raw_defaced_output_path, self.normalized_defaced_output_path])

    def normalize(
        self,
        temporary_directory: Union[str, Path],
        store_unnormalized: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Normalize the image using the specified normalizer.

        Args:
            temporary_directory (str or Path): Path to the temporary directory.
            store_unnormalized (str or Path, optional): Path to store unnormalized images.

        Returns:
            None
        """
        # Backup the unnormalized file
        if store_unnormalized:
            store_unnormalized = Path(store_unnormalized)
            store_unnormalized.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                src=str(self.current),
                dst=str(
                    store_unnormalized / f"unnormalized__{self.modality_name}.nii.gz"
                ),
            )

        if temporary_directory:
            unnormalized_dir = Path(temporary_directory) / "unnormalized"
            unnormalized_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                src=str(self.current),
                dst=str(
                    unnormalized_dir / f"unnormalized__{self.modality_name}.nii.gz"
                ),
            )

        # Normalize the image
        if self.normalizer:
            image = read_nifti(str(self.current))
            normalized_image = self.normalizer.normalize(image=image)
            write_nifti(
                input_array=normalized_image,
                output_nifti_path=str(self.current),
                reference_nifti_path=str(self.current),
            )
        else:
            logger.info("No normalizer specified; skipping normalization.")

    def register(
        self,
        registrator: Registrator,
        fixed_image_path: Union[str, Path],
        registration_dir: Union[str, Path],
        moving_image_name: str,
        step: PreprocessorSteps,
    ) -> Path:
        """
        Register the current modality to a fixed image using the specified registrator.

        Args:
            registrator (Registrator): The registrator object.
            fixed_image_path (str or Path): Path to the fixed image.
            registration_dir (str or Path): Directory to store registration results.
            moving_image_name (str): Name of the moving image.

        Returns:
            Path: Path to the registration matrix.
        """
        fixed_image_path = Path(fixed_image_path)
        registration_dir = Path(registration_dir)

        registered = registration_dir / f"{moving_image_name}.nii.gz"
        registered_log = registration_dir / f"{moving_image_name}.log"

        # Note, add file ending depending on registration backend!
        registered_matrix = registration_dir / f"{moving_image_name}"

        registrator.register(
            fixed_image_path=fixed_image_path,
            moving_image_path=self.current,
            transformed_image_path=registered,
            matrix_path=registered_matrix,
            log_file_path=registered_log,
        )
        self.current = registered
        self.steps[step] = registered
        return registered_matrix

    def apply_bet_mask(
        self,
        brain_extractor: BrainExtractor,
        mask_path: Union[str, Path],
        bet_dir: Union[str, Path],
    ) -> None:
        """
        Apply a brain mask to the current modality using the specified brain extractor.

        Args:
            brain_extractor (BrainExtractor): The brain extractor object.
            mask_path (str or Path): Path to the brain mask.
            bet_dir (str or Path): Directory to store computed bet images.

        Returns:
            None
        """
        if self.bet:
            mask_path = Path(mask_path)
            bet_dir = Path(bet_dir)
            bet_img = bet_dir / f"atlas__{self.modality_name}_bet.nii.gz"

            brain_extractor.apply_mask(
                input_image_path=self.current,
                mask_path=mask_path,
                bet_image_path=bet_img,
            )
            self.current = bet_img
            self.steps[PreprocessorSteps.BET] = bet_img
        else:
            logger.info("No Brain Extractor specified; skipping brain extraction.")

    def apply_deface_mask(
        self,
        defacer: Defacer,
        mask_path: Union[str, Path],
        deface_dir: Union[str, Path],
    ) -> None:
        """
        Apply a deface mask to the current modality using the specified brain extractor.

        Args:
            defacer (Defacer): The Defacer object.
            mask_path (str or Path): Path to the deface mask.
            defaced_masked_dir_path (str or Path): Directory to store masked images.
        """
        if self.requires_deface:
            mask_path = Path(mask_path)
            deface_dir = Path(deface_dir)
            defaced_img = deface_dir / f"atlas__{self.modality_name}_defaced.nii.gz"
            input_img = self.steps[
                (
                    PreprocessorSteps.ATLAS_CORRECTED
                    if self.atlas_correction
                    else PreprocessorSteps.ATLAS_REGISTERED
                )
            ]
            defacer.apply_mask(
                input_image_path=input_img,
                mask_path=mask_path,
                defaced_image_path=defaced_img,
            )
            self.current = defaced_img
            self.steps[PreprocessorSteps.DEFACED] = defaced_img

    def transform(
        self,
        registrator: Registrator,
        fixed_image_path: Union[str, Path],
        registration_dir_path: Union[str, Path],
        moving_image_name: str,
        transformation_matrix_path: Union[str, Path],
        step: PreprocessorSteps,
    ) -> None:
        """
        Transform the current modality using the specified registrator and transformation matrix.

        Args:
            registrator (Registrator): The registrator object.
            fixed_image_path (str or Path): Path to the fixed image.
            registration_dir_path (str or Path): Directory to store transformation results.
            moving_image_name (str): Name of the moving image.
            transformation_matrix_path (str or Path): Path to the transformation matrix.

        Returns:
            None
        """
        fixed_image_path = Path(fixed_image_path)
        registration_dir_path = Path(registration_dir_path)
        transformation_matrix_path = Path(transformation_matrix_path)

        transformed = registration_dir_path / f"{moving_image_name}.nii.gz"
        transformed_log = registration_dir_path / f"{moving_image_name}.log"

        registrator.transform(
            fixed_image_path=fixed_image_path,
            moving_image_path=self.current,
            transformed_image_path=transformed,
            matrix_path=transformation_matrix_path,
            log_file_path=transformed_log,
        )
        self.current = transformed
        self.steps[step] = transformed

    def extract_brain_region(
        self,
        brain_extractor: BrainExtractor,
        bet_dir_path: Union[str, Path],
    ) -> Path:
        """
        WARNING: Legacy method. Please Migrate to use the CenterModality Class. Will be removed in future versions.

        Extract the brain region using the specified brain extractor.

        Args:
            brain_extractor (BrainExtractor): The brain extractor object.
            bet_dir_path (str or Path): Directory to store brain extraction results.

        Returns:
            Path: Path to the extracted brain mask.
        """

        warnings.warn(
            "Legacy method. Please Migrate to use the CenterModality Class. Will be removed in future versions.",
            category=DeprecationWarning,
        )

        bet_dir_path = Path(bet_dir_path)
        bet_log = bet_dir_path / "brain-extraction.log"

        atlas_bet_cm = bet_dir_path / f"atlas__{self.modality_name}_bet.nii.gz"
        mask_path = bet_dir_path / f"atlas__{self.modality_name}_brain_mask.nii.gz"

        brain_extractor.extract(
            input_image_path=self.current,
            masked_image_path=atlas_bet_cm,
            brain_mask_path=mask_path,
            log_file_path=bet_log,
        )

        # always temporarily store bet image for center modality, since e.g. quickshear defacing could require it
        # down the line even if the user does not wish to save the bet image
        self.steps[PreprocessorSteps.BET] = atlas_bet_cm

        if self.bet:
            self.current = atlas_bet_cm
        return mask_path

    def deface(
        self,
        defacer,
        defaced_dir_path: Union[str, Path],
    ) -> Path:
        """
        WARNING: Legacy method. Please Migrate to use the CenterModality Class. Will be removed in future versions.

        Deface the current modality using the specified defacer.

        Args:
            defacer (Defacer): The defacer object.
            defaced_dir_path (str or Path): Directory to store defacing results.

        Returns:
            Path: Path to the extracted brain mask.
        """
        warnings.warn(
            "Legacy method. Please Migrate to use the CenterModality class. Will be removed in future versions.",
            category=DeprecationWarning,
        )
        if isinstance(defacer, QuickshearDefacer):

            defaced_dir_path = Path(defaced_dir_path)
            atlas_mask_path = (
                defaced_dir_path / f"atlas__{self.modality_name}_deface_mask.nii.gz"
            )

            defacer.deface(
                mask_image_path=atlas_mask_path,
                input_image_path=self.steps[PreprocessorSteps.BET],
            )
            return atlas_mask_path
        else:
            logger.warning(
                "Defacing method not implemented yet. Skipping defacing for this modality."
            )
            return None

    def save_current_image(
        self,
        output_path: Union[str, Path],
        normalization: bool = False,
    ) -> None:
        """
        Save the current image to the specified output path.

        Args:
            output_path (str or Path): The output file path.
            normalization (bool, optional): If True, apply normalization before saving.

        Returns:
            None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if normalization:
            if self.normalizer is None:
                raise ValueError("Normalizer is required for normalization.")
            image = read_nifti(str(self.current))
            normalized_image = self.normalizer.normalize(image=image)
            write_nifti(
                input_array=normalized_image,
                output_nifti_path=str(output_path),
                reference_nifti_path=str(self.current),
            )
        else:
            shutil.copyfile(
                src=str(self.current),
                dst=str(output_path),
            )


class CenterModality(Modality):
    """
    Represents a medical image center modality with associated preprocessing information.

    Args:
        modality_name (str): Name of the modality, e.g., "T1", "T2", "FLAIR".
        input_path (str or Path): Path to the input modality data.
        normalizer (Normalizer, optional): An optional normalizer for intensity normalization.
        raw_bet_output_path (str or Path, optional): Path to save the raw brain extracted modality data.
        raw_skull_output_path (str or Path, optional): Path to save the raw modality data with skull.
        raw_defaced_output_path (str or Path, optional): Path to save the raw defaced modality data.
        normalized_bet_output_path (str or Path, optional): Path to save the normalized brain extracted modality data. Requires a normalizer.
        normalized_skull_output_path (str or Path, optional): Path to save the normalized modality data with skull. Requires a normalizer.
        normalized_defaced_output_path (str or Path, optional): Path to save the normalized defaced modality data. Requires a normalizer.
        atlas_correction (bool, optional): Indicates whether atlas correction should be performed.
        bet_mask_output_path (str or Path, optional): Path to save the brain extraction mask.
        defacing_mask_output_path (str or Path, optional): Path to save the defacing mask.

    Attributes:
        modality_name (str): Name of the modality.
        input_path (str or Path): Path to the input modality data.
        normalizer (Normalizer, optional): An optional normalizer for intensity normalization.
        raw_bet_output_path (str or Path, optional): Path to save the raw brain extracted modality data.
        raw_skull_output_path (str or Path, optional): Path to save the raw modality data with skull.
        raw_defaced_output_path (str or Path, optional): Path to save the raw defaced modality data.
        normalized_bet_output_path (str or Path, optional): Path to save the normalized brain extracted modality data. Requires a normalizer.
        normalized_skull_output_path (str or Path, optional): Path to save the normalized modality data with skull. Requires a normalizer.
        normalized_defaced_output_path (str or Path, optional): Path to save the normalized defaced modality data. Requires a normalizer.
        bet (bool): Indicates whether brain extraction is enabled.
        atlas_correction (bool): Indicates whether atlas correction should be performed.
        bet_mask_output_path (Path, optional): Path to save the brain extraction mask.
        defacing_mask_output_path (Path, optional): Path to save the defacing mask.

    Example:
        >>> t1_modality = CenterModality(
        ...     modality_name="T1",
        ...     input_path="/path/to/input_t1.nii",
        ...     normalizer=PercentileNormalizer(),
        ...     raw_bet_output_path="/path/to/raw_bet_t1.nii",
        ...     normalized_bet_output_path="/path/to/norm_bet_t1.nii",
        ...     bet_mask_output_path="/path/to/bet_mask_t1.nii",
        ... )
    """

    def __init__(
        self,
        modality_name: str,
        input_path: Union[str, Path],
        normalizer: Optional[Normalizer] = None,
        raw_bet_output_path: Optional[Union[str, Path]] = None,
        raw_skull_output_path: Optional[Union[str, Path]] = None,
        raw_defaced_output_path: Optional[Union[str, Path]] = None,
        normalized_bet_output_path: Optional[Union[str, Path]] = None,
        normalized_skull_output_path: Optional[Union[str, Path]] = None,
        normalized_defaced_output_path: Optional[Union[str, Path]] = None,
        atlas_correction: bool = True,
        bet_mask_output_path: Optional[Union[str, Path]] = None,
        defacing_mask_output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(
            modality_name=modality_name,
            input_path=input_path,
            normalizer=normalizer,
            raw_bet_output_path=raw_bet_output_path,
            raw_skull_output_path=raw_skull_output_path,
            raw_defaced_output_path=raw_defaced_output_path,
            normalized_bet_output_path=normalized_bet_output_path,
            normalized_skull_output_path=normalized_skull_output_path,
            normalized_defaced_output_path=normalized_defaced_output_path,
            atlas_correction=atlas_correction,
        )
        # Only for CenterModality
        self.bet_mask_output_path = (
            Path(bet_mask_output_path) if bet_mask_output_path else None
        )
        self.defacing_mask_output_path = (
            Path(defacing_mask_output_path) if defacing_mask_output_path else None
        )

    def extract_brain_region(
        self,
        brain_extractor: BrainExtractor,
        bet_dir_path: Union[str, Path],
    ) -> Path:
        """
        Extract the brain region using the specified brain extractor.

        Args:
            brain_extractor (BrainExtractor): The brain extractor object.
            bet_dir_path (str or Path): Directory to store brain extraction results.

        Returns:
            Path: Path to the extracted brain mask.
        """
        bet_dir_path = Path(bet_dir_path)
        bet_log = bet_dir_path / "brain-extraction.log"

        atlas_bet_cm = bet_dir_path / f"atlas__{self.modality_name}_bet.nii.gz"
        mask_path = bet_dir_path / f"atlas__{self.modality_name}_brain_mask.nii.gz"

        brain_extractor.extract(
            input_image_path=self.current,
            masked_image_path=atlas_bet_cm,
            brain_mask_path=mask_path,
            log_file_path=bet_log,
        )

        if self.bet_mask_output_path:
            logger.debug(f"Saving bet mask to {self.bet_mask_output_path}")
            self.save_mask(mask_path=mask_path, output_path=self.bet_mask_output_path)

        # always temporarily store bet image for center modality, since e.g. quickshear defacing could require it
        # down the line even if the user does not wish to save the bet image
        self.steps[PreprocessorSteps.BET] = atlas_bet_cm

        if self.bet:
            self.current = atlas_bet_cm
        return mask_path

    def deface(
        self,
        defacer,
        defaced_dir_path: Union[str, Path],
    ) -> Path:
        """
        Deface the current modality using the specified defacer.

        Args:
            defacer (Defacer): The defacer object.
            defaced_dir_path (str or Path): Directory to store defacing results.

        Returns:
            Path: Path to the extracted brain mask.
        """

        if isinstance(defacer, QuickshearDefacer):

            defaced_dir_path = Path(defaced_dir_path)
            atlas_mask_path = (
                defaced_dir_path / f"atlas__{self.modality_name}_deface_mask.nii.gz"
            )

            defacer.deface(
                mask_image_path=atlas_mask_path,
                input_image_path=self.steps[PreprocessorSteps.BET],
            )

            if self.defacing_mask_output_path:
                logger.debug(f"Saving deface mask to {self.defacing_mask_output_path}")
                self.save_mask(
                    mask_path=atlas_mask_path,
                    output_path=self.defacing_mask_output_path,
                )

            return atlas_mask_path
        else:
            logger.warning(
                "Defacing method not implemented yet. Skipping defacing for this modality."
            )
            return None

    def save_mask(self, mask_path: Union[str, Path], output_path: Path) -> None:
        """
        Save the mask to the specified output path.

        Args:
            mask_path (Union[str, Path]): Mask NifTI file path.
            output_path (Path): Output NifTI file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            src=str(mask_path),
            dst=str(output_path),
        )
