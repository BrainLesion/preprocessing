import logging
import shutil
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from auxiliary.io import read_image, write_image
from loguru import logger

from brainles_preprocessing.brain_extraction.brain_extractor import BrainExtractor
from brainles_preprocessing.constants import Atlas, PreprocessorSteps
from brainles_preprocessing.defacing import Defacer, QuickshearDefacer
from brainles_preprocessing.normalization.normalizer_base import Normalizer
from brainles_preprocessing.registration import (  # TODO: this will throw warnings if ANTs or NiftyReg are not installed, not ideal
    ANTsRegistrator,
    NiftyRegRegistrator,
)
from brainles_preprocessing.registration.registrator import Registrator
from brainles_preprocessing.utils.zenodo import fetch_atlases


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
        n4_bias_correction (bool, optional): Indicates whether N4 bias correction should be performed.

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
        n4_bias_correction (bool): Indicates whether N4 bias correction should be performed.
        coregistration_transform_path (str or None): Path to the coregistration transformation matrix, will be set after coregistration.

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
        n4_bias_correction: bool = False,
    ) -> None:
        # Basics
        self.modality_name = modality_name
        self.input_path = Path(input_path)
        self.current = self.input_path
        self.normalizer = normalizer
        self.atlas_correction = atlas_correction
        self.n4_bias_correction = n4_bias_correction
        self.transformation_paths: Dict[PreprocessorSteps, Path | None] = {}

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
        self.steps[PreprocessorSteps.INPUT] = self.input_path

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
            image = read_image(str(self.current))
            normalized_image = self.normalizer.normalize(image=image)
            write_image(
                input_array=normalized_image,
                output_path=str(self.current),
                reference_path=str(self.current),
            )
        else:
            logger.info("No normalizer specified; skipping normalization.")

    def _find_transformation_matrix(
        self, transform_incomplete_path: Path
    ) -> Optional[Path]:
        possible_Files = list(
            transform_incomplete_path.parent.glob(f"{transform_incomplete_path.stem}.*")
        )
        if len(possible_Files) == 0:
            logger.warning(
                f"No transformation matrix found for {transform_incomplete_path}. "
                "Returning None."
            )
            return None
        elif len(possible_Files) > 1:
            # TODO: Handle this case more gracefully, e.g., try to find proper extension based on the registrator
            logger.warning(
                f"Multiple transformation matrices found for {transform_incomplete_path}. "
                "Returning the first one."
            )
        return possible_Files[0]

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
            step (PreprocessorSteps): The current preprocessing step.

        Returns:
            Path: Path to the registration matrix.
        """
        fixed_image_path = Path(fixed_image_path)
        registration_dir = Path(registration_dir)

        registered = registration_dir / f"{moving_image_name}.nii.gz"
        registered_log = registration_dir / f"{moving_image_name}.log"

        # Note, add file ending depending on registration backend!
        registered_matrix = registration_dir / f"M_{moving_image_name}"

        registrator.register(
            fixed_image_path=fixed_image_path,
            moving_image_path=self.current,
            transformed_image_path=registered,
            matrix_path=registered_matrix,
            log_file_path=registered_log,
        )
        self.current = registered
        self.steps[step] = registered

        self.transformation_paths[step] = self._find_transformation_matrix(
            transform_incomplete_path=registered_matrix
        )

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
            bet_img = bet_dir / f"{self.modality_name}_bet.nii.gz"

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
            defaced_img = deface_dir / f"{self.modality_name}_defaced.nii.gz"

            # For Atlas centric preprocessing, we use the atlas corrected or registered image as input
            # For Native space preprocessing, we use the coregistered image as input
            input_img = (
                self.steps[
                    (
                        PreprocessorSteps.ATLAS_CORRECTED
                        if self.atlas_correction
                        else PreprocessorSteps.ATLAS_REGISTERED
                    )
                ]
                or self.steps[PreprocessorSteps.COREGISTERED]
            )

            if input_img is None:
                raise ValueError(
                    "Input image for defacing is missing. Ensure that the required preprocessing steps "
                    "have been performed before defacing."
                )

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
            step (PreprocessorSteps): The current preprocessing step.
        Returns:
            None
        """
        fixed_image_path = Path(fixed_image_path)
        registration_dir_path = Path(registration_dir_path)
        transformation_matrix_path = Path(transformation_matrix_path)

        transformed = registration_dir_path / f"{moving_image_name}.nii.gz"
        transformed_log = registration_dir_path / f"{moving_image_name}.log"

        if (
            isinstance(registrator, (ANTsRegistrator, NiftyRegRegistrator))
            and step == PreprocessorSteps.ATLAS_REGISTERED
        ):
            # we test uniting transforms for these registrators
            assert (
                self.transformation_paths.get(PreprocessorSteps.COREGISTERED, None)
                is not None
            ), "Coregistration must be performed before applying atlas registration."

            registrator.transform(
                fixed_image_path=fixed_image_path,
                moving_image_path=self.steps[PreprocessorSteps.INPUT],
                transformed_image_path=transformed,
                matrix_path=[
                    self.transformation_paths[
                        PreprocessorSteps.COREGISTERED
                    ],  # coregistration matrix
                    transformation_matrix_path,  # atlas registration matrix
                ],
                log_file_path=transformed_log,
            )
        else:
            registrator.transform(
                fixed_image_path=fixed_image_path,
                moving_image_path=self.current,
                transformed_image_path=transformed,
                matrix_path=transformation_matrix_path,
                log_file_path=transformed_log,
            )

        self.current = transformed
        self.steps[step] = transformed
        self.transformation_paths[step] = self._find_transformation_matrix(
            transform_incomplete_path=transformation_matrix_path
        )

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

        bet = bet_dir_path / f"{self.modality_name}_bet.nii.gz"
        mask_path = bet_dir_path / f"{self.modality_name}_brain_mask.nii.gz"

        brain_extractor.extract(
            input_image_path=self.current,
            masked_image_path=bet,
            brain_mask_path=mask_path,
            log_file_path=bet_log,
        )

        # always temporarily store bet image for center modality, since e.g. quickshear defacing could require it
        # down the line even if the user does not wish to save the bet image
        self.steps[PreprocessorSteps.BET] = bet

        if self.bet:
            self.current = bet
        return mask_path

    def deface(
        self,
        defacer,
        defaced_dir_path: Union[str, Path],
        registrator: Optional[Registrator] = None,
    ) -> Path | None:
        """
        WARNING: Legacy method. Please Migrate to use the CenterModality Class. Will be removed in future versions.
        """
        raise RuntimeError(
            "The 'deface' method has been deprecated and moved to the CenterModality class as its only supposed to be called once from the CenterModality. "
            "Please update your code to use the 'CenterModality.deface()' method instead."
        )

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
            image = read_image(str(self.current))
            normalized_image = self.normalizer.normalize(image=image)
            write_image(
                input_array=normalized_image,
                output_path=str(output_path),
                reference_path=str(self.current),
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
        n4_bias_correction (bool, optional): Indicates whether N4 bias correction should be performed.
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
        n4_bias_correction (bool): Indicates whether N4 bias correction should be performed.
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
        n4_bias_correction: bool = False,
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
            n4_bias_correction=n4_bias_correction,
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
        use_gpu: bool = True,
    ) -> Path:
        """

        Extract the brain region using the specified brain extractor.

        Args:
            brain_extractor (BrainExtractor): The brain extractor object.
            bet_dir_path (str or Path): Directory to store brain extraction results.
            use_gpu (bool): Whether to use GPU for brain extraction if available.

        Returns:
            Path: Path to the extracted brain mask.
        """

        bet_dir_path = Path(bet_dir_path)
        bet_log = bet_dir_path / "brain-extraction.log"

        bet = bet_dir_path / f"{self.modality_name}_bet.nii.gz"
        mask_path = bet_dir_path / f"{self.modality_name}_brain_mask.nii.gz"

        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        brain_extractor.extract(
            input_image_path=self.current,
            masked_image_path=bet,
            brain_mask_path=mask_path,
            log_file_path=bet_log,
            device=device,
        )

        # always temporarily store bet image for center modality, since e.g. quickshear defacing could require it
        # down the line even if the user does not wish to save the bet image
        self.steps[PreprocessorSteps.BET] = bet

        if self.bet:
            self.current = bet
        return mask_path

    def deface(
        self,
        defacer: Defacer,
        defaced_dir_path: Union[str, Path],
        registrator: Optional[Registrator] = None,
    ) -> Path | None:
        """
        Deface the current modality using the specified defacer.

        Args:
            defacer (Defacer): The defacer object.
            defaced_dir_path (str or Path): Directory to store defacing results.
            registrator (Registrator, optional): The registrator object for atlas registration.

        Returns:
            Path | None: Path to the defacing mask if successful, None otherwise.
        """
        if isinstance(defacer, QuickshearDefacer):
            defaced_dir_path = Path(defaced_dir_path)
            mask_path = defaced_dir_path / f"{self.modality_name}_deface_mask.nii.gz"

            if self.steps.get(PreprocessorSteps.BET, None) is None:
                raise ValueError(
                    "Brain extraction must be performed before defacing. "
                    "Please run brain extraction first."
                )

            if defacer.force_atlas_registration and registrator is not None:
                logger.info("Forcing atlas registration before defacing as requested.")
                atlas_bet = defaced_dir_path / "atlas_bet.nii.gz"
                atlas_bet_M = defaced_dir_path / "M_atlas_bet"

                # resolve atlas image path
                if isinstance(defacer.atlas_image_path, Atlas):
                    atlas_folder = fetch_atlases()
                    atlas_image_path = atlas_folder / defacer.atlas_image_path.value
                else:
                    atlas_image_path = Path(defacer.atlas_image_path)

                registrator.register(
                    fixed_image_path=atlas_image_path,
                    moving_image_path=self.steps[PreprocessorSteps.BET],
                    transformed_image_path=atlas_bet,
                    matrix_path=atlas_bet_M,
                    log_file_path=defaced_dir_path / "atlas_bet.log",
                )

                deface_mask_atlas = defaced_dir_path / "deface_mask_atlas.nii.gz"
                defacer.deface(
                    input_image_path=atlas_bet,
                    mask_image_path=deface_mask_atlas,
                )

                registrator.inverse_transform(
                    fixed_image_path=self.steps[PreprocessorSteps.BET],
                    moving_image_path=deface_mask_atlas,
                    transformed_image_path=mask_path,
                    matrix_path=atlas_bet_M,
                    log_file_path=defaced_dir_path / "inverse_transform.log",
                )
            else:
                defacer.deface(
                    input_image_path=self.steps[PreprocessorSteps.BET],
                    mask_image_path=mask_path,
                )

            return mask_path
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
