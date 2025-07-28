import shutil
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

from brainles_preprocessing.brain_extraction.brain_extractor import BrainExtractor
from brainles_preprocessing.constants import Atlas, PreprocessorSteps
from brainles_preprocessing.defacing import Defacer, QuickshearDefacer
from brainles_preprocessing.modality import CenterModality, Modality
from brainles_preprocessing.n4_bias_correction import N4BiasCorrector
from brainles_preprocessing.preprocessor.preprocessor import BasePreprocessor
from brainles_preprocessing.registration.registrator import Registrator
from brainles_preprocessing.utils.zenodo import fetch_atlases


class AtlasCentricPreprocessor(BasePreprocessor):
    """
    Preprocesses medical image modalities using coregistration, atlas-registration, atlas-correction, normalization, brain extraction, and more.

    Args:
        center_modality (CenterModality): The central modality for coregistration.
        moving_modalities (List[Modality]): List of modalities to be coregistered to the central modality.
        registrator (Registrator): The registrator object for coregistration and registration to the atlas.
        brain_extractor (Optional[BrainExtractor]): The brain extractor object for brain extraction.
        defacer (Optional[Defacer]): The defacer object for defacing images.
        atlas_image_path (Optional[str or Path]): Path to the atlas image for registration (default is the T1 atlas).
        n4_bias_corrector (Optional[N4BiasCorrector]): The N4 bias corrector object for bias field correction. Defaults to SitkN4BiasCorrector with Otsu Thresholding.
        temp_folder (Optional[str or Path]): Path to a folder for storing intermediate results.
        use_gpu (Optional[bool]): Use GPU for processing if True, CPU if False, or automatically detect if None.
        limit_cuda_visible_devices (Optional[str]): Limit CUDA visible devices to a specific GPU ID.

    """

    def __init__(
        self,
        center_modality: CenterModality,
        moving_modalities: List[Modality],
        registrator: Registrator = None,
        brain_extractor: Optional[BrainExtractor] = None,
        defacer: Optional[Defacer] = None,
        atlas_image_path: Union[str, Path, Atlas] = Atlas.BRATS_SRI24,
        n4_bias_corrector: Optional[N4BiasCorrector] = None,
        temp_folder: Optional[Union[str, Path]] = None,
        use_gpu: bool = True,
        limit_cuda_visible_devices: Optional[str] = None,
    ):
        super().__init__(
            center_modality=center_modality,
            moving_modalities=moving_modalities,
            registrator=registrator,
            brain_extractor=brain_extractor,
            defacer=defacer,
            n4_bias_corrector=n4_bias_corrector,
            temp_folder=temp_folder,
            use_gpu=use_gpu,
            limit_cuda_visible_devices=limit_cuda_visible_devices,
        )

        if isinstance(atlas_image_path, Atlas):
            atlas_folder = fetch_atlases()
            self.atlas_image_path = atlas_folder / atlas_image_path.value
        else:
            self.atlas_image_path = Path(atlas_image_path)

    def run(
        self,
        save_dir_coregistration: Optional[Union[str, Path]] = None,
        save_dir_atlas_registration: Optional[Union[str, Path]] = None,
        save_dir_atlas_correction: Optional[Union[str, Path]] = None,
        save_dir_n4_bias_correction: Optional[Union[str, Path]] = None,
        save_dir_brain_extraction: Optional[Union[str, Path]] = None,
        save_dir_defacing: Optional[Union[str, Path]] = None,
        save_dir_transformations: Optional[Union[str, Path]] = None,
        log_file: Optional[Union[str, Path]] = None,
    ):
        """
        Execute the atlas-centric preprocessing pipeline

        Args:
            save_dir_coregistration (str or Path, optional): Directory path to save intermediate coregistration results.
            save_dir_atlas_registration (str or Path, optional): Directory path to save intermediate atlas registration results.
            save_dir_atlas_correction (str or Path, optional): Directory path to save intermediate atlas correction results.
            save_dir_n4_bias_correction (str or Path, optional): Directory path to save intermediate N4 bias correction results.
            save_dir_brain_extraction (str or Path, optional): Directory path to save intermediate brain extraction results.
            save_dir_defacing (str or Path, optional): Directory path to save intermediate defacing results.
            save_dir_transformations (str or Path, optional): Directory path to save transformation matrices. Defaults to None.
            log_file (str or Path, optional): Path to save the log file. Defaults to a timestamped file in the current directory.

        This method orchestrates the entire preprocessing workflow by sequentially performing:

        1. Co-registration: Aligning moving modalities to the central modality.
        2. Atlas Registration: Aligning the central modality to a predefined atlas.
        3. (Optional) Atlas Correction: Applying additional correction in atlas space if specified.
        4. (Optional) N4 Bias Correction: Applying N4 bias field correction if specified.
        5. (Optional) Brain Extraction: Optionally extracting brain regions using specified masks. Only executed if any modality requires a brain extraction output (or a defacing output that requires prior brain extraction).
        6. (Optional) Defacing: Optionally deface images to remove facial features. Only executed if any modality requires a defacing output.

        Results are saved in the specified directories, allowing for modular and configurable output storage.
        """

        logger_id = self._add_log_file_handler(log_file)
        try:
            logger.info(f"{' Starting preprocessing ':=^80}")
            modality_names = ", ".join(
                [modality.modality_name for modality in self.moving_modalities]
            )
            logger.info(
                f"Received center modality: {self.center_modality.modality_name} "
                f"and moving modalities: {modality_names}"
            )

            # Co-register moving modalities to center modality
            logger.info(f"{' Starting Coregistration ':-^80}")
            self.run_coregistration(
                save_dir_coregistration=save_dir_coregistration,
            )
            logger.info(
                f"Coregistration complete. Output saved to {save_dir_coregistration}"
            )

            # Register center modality to atlas
            logger.info(f"{' Starting atlas registration ':-^80}")
            self.run_atlas_registration(
                save_dir_atlas_registration=save_dir_atlas_registration,
            )
            logger.info(
                f"Transformations complete. Output saved to {save_dir_atlas_registration}"
            )

            # Optional: additional correction in atlas space
            logger.info(f"{' Checking optional atlas correction ':-^80}")
            self.run_atlas_correction(
                save_dir_atlas_correction=save_dir_atlas_correction,
            )

            # Optional: N4 bias correction
            logger.info(f"{' Checking optional N4 bias correction ':-^80}")
            self.run_n4_bias_correction(
                save_dir_n4_bias_correction=save_dir_n4_bias_correction,
            )

            # Now we save images that are not skullstripped (current image = atlas registered or atlas registered + corrected)
            logger.info("Saving non skull-stripped images...")
            for modality in self.all_modalities:
                if modality.raw_skull_output_path:
                    modality.save_current_image(
                        modality.raw_skull_output_path,
                        normalization=False,
                    )
                if modality.normalized_skull_output_path:
                    modality.save_current_image(
                        modality.normalized_skull_output_path,
                        normalization=True,
                    )

            # Optional: Brain extraction
            logger.info(f"{' Checking optional brain extraction ':-^80}")
            self.run_brain_extraction(
                save_dir_brain_extraction=save_dir_brain_extraction,
            )

            # Defacing
            logger.info(f"{' Checking optional defacing ':-^80}")
            self.run_defacing(
                save_dir_defacing=save_dir_defacing,
            )

            # move to separate method
            if save_dir_transformations:
                save_dir_transformations = Path(save_dir_transformations)

                # Save transformation matrices
                logger.info(
                    f"Saving transformation matrices to {save_dir_transformations}"
                )
                for modality in self.all_modalities:

                    modality_transformations_dir = (
                        save_dir_transformations / modality.modality_name
                    )
                    modality_transformations_dir.mkdir(exist_ok=True, parents=True)
                    for step, path in modality.transformation_paths.items():
                        if path is not None:
                            shutil.copyfile(
                                src=str(path.absolute()),
                                dst=str(
                                    modality_transformations_dir
                                    / f"{step.value}_{path.name}"
                                ),
                            )

            # End
            logger.info(f"{' Preprocessing complete ':=^80}")
        finally:
            # Remove log file handler if it was added
            logger.remove(logger_id)

    def run_atlas_registration(
        self, save_dir_atlas_registration: Optional[Union[str, Path]] = None
    ) -> None:
        """Register center modality to atlas.

        Args:
            save_dir_atlas_registration (Optional[str or Path], optional): Directory path to save intermediate atlas registration results. Defaults to None.
        """
        atlas_dir = self.temp_folder / "atlas-space"
        atlas_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Registering center modality to atlas...")
        center_file_name = f"atlas__{self.center_modality.modality_name}"
        transformation_matrix = self.center_modality.register(
            registrator=self.registrator,
            fixed_image_path=self.atlas_image_path,
            registration_dir=atlas_dir,
            moving_image_name=center_file_name,
            step=PreprocessorSteps.ATLAS_REGISTERED,
        )
        logger.info(f"Atlas registration complete. Output saved to {atlas_dir}")

        # Transform moving modalities to atlas
        logger.info(
            f"Transforming {len(self.moving_modalities)} moving modalities to atlas space..."
        )
        for moving_modality in self.moving_modalities:
            moving_file_name = f"atlas__{moving_modality.modality_name}"
            logger.info(
                f"Transforming modality {moving_modality.modality_name} (file={moving_file_name}) to atlas space..."
            )
            moving_modality.transform(
                registrator=self.registrator,
                fixed_image_path=self.atlas_image_path,
                registration_dir_path=Path(atlas_dir),
                moving_image_name=moving_file_name,
                transformation_matrix_path=transformation_matrix,
                step=PreprocessorSteps.ATLAS_REGISTERED,
            )
        self._save_output(
            src=atlas_dir,
            save_dir=save_dir_atlas_registration,
        )

    def run_atlas_correction(
        self,
        save_dir_atlas_correction: Optional[Union[str, Path]] = None,
    ) -> None:
        """Apply optional atlas correction to moving modalities.

        Args:
            save_dir_atlas_correction (Optional[str or Path], optional): Directory path to save intermediate atlas correction results. Defaults to None.
        """
        atlas_correction_dir = self.temp_folder / "atlas-correction"
        atlas_correction_dir.mkdir(exist_ok=True, parents=True)

        for moving_modality in self.moving_modalities:
            if moving_modality.atlas_correction:
                logger.info(
                    f"Applying optional atlas correction for modality {moving_modality.modality_name}"
                )
                moving_file_name = f"atlas_corrected__{self.center_modality.modality_name}__{moving_modality.modality_name}"
                moving_modality.register(
                    registrator=self.registrator,
                    fixed_image_path=self.center_modality.current,
                    registration_dir=atlas_correction_dir,
                    moving_image_name=moving_file_name,
                    step=PreprocessorSteps.ATLAS_CORRECTED,
                )
            else:
                logger.info(
                    f"Skipping optional atlas correction for Modality {moving_modality.modality_name}."
                )

        if self.center_modality.atlas_correction:
            center_atlas_corrected_path = (
                atlas_correction_dir
                / f"atlas_corrected__{self.center_modality.modality_name}.nii.gz"
            )

            shutil.copyfile(
                src=str(self.center_modality.current),
                dst=str(center_atlas_corrected_path),
            )
            # save step result
            self.center_modality.steps[PreprocessorSteps.ATLAS_CORRECTED] = (
                center_atlas_corrected_path
            )

        self._save_output(
            src=atlas_correction_dir,
            save_dir=save_dir_atlas_correction,
        )

    def run_defacing(
        self, save_dir_defacing: Optional[Union[str, Path]] = None
    ) -> None:
        """Deface images to remove facial features using specified Defacer.

        Args:
            save_dir_defacing (Optional[str or Path], optional): Directory path to save intermediate defacing results. Defaults to None.
        """

        # Skip if no defacing is required
        if not self.requires_defacing:
            logger.info("Skipping optional defacing.")
            return

        logger.info("Starting defacing...")

        # Setup output dir
        deface_dir = self.temp_folder / "deface"
        deface_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Defacing center modality...")

        # Assert that a defacer is specified (since the arg is optional)
        if self.defacer is None:
            logger.warning(
                "Requested defacing but no defacer was specified during class initialization."
                + " Using default `brainles_preprocessing.defacing.QuickshearDefacer`"
            )
            self.defacer = QuickshearDefacer()

        atlas_mask = self.center_modality.deface(
            defacer=self.defacer, defaced_dir_path=deface_dir
        )
        # looping over _all_ modalities since .deface is no applying the computed mask
        for moving_modality in self.all_modalities:
            logger.info(f"Applying deface mask to {moving_modality.modality_name}...")
            moving_modality.apply_deface_mask(
                defacer=self.defacer,
                mask_path=atlas_mask,
                deface_dir=deface_dir,
            )

        self._save_output(
            src=deface_dir,
            save_dir=save_dir_defacing,
        )
        # now we save images that are skull-stripped
        logger.info("Saving defaced images...")
        for modality in self.all_modalities:
            if modality.raw_defaced_output_path:
                modality.save_current_image(
                    modality.raw_defaced_output_path,
                    normalization=False,
                )
            if modality.normalized_defaced_output_path:
                modality.save_current_image(
                    modality.normalized_defaced_output_path,
                    normalization=True,
                )
