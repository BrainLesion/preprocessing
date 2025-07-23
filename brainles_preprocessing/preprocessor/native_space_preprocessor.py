import shutil
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from brainles_preprocessing.defacing import QuickshearDefacer
from brainles_preprocessing.preprocessor.preprocessor import BasePreprocessor


class NativeSpacePreprocessor(BasePreprocessor):
    """
    Preprocesses medical image modalities using coregistration, atlas-registration, atlas-correction, normalization, brain extraction, and more.

    Args:
        center_modality (CenterModality): The central modality for coregistration.
        moving_modalities (List[Modality]): List of modalities to be coregistered to the central modality.
        registrator (Registrator): The registrator object for coregistration and registration to the atlas.
        brain_extractor (Optional[BrainExtractor]): The brain extractor object for brain extraction.
        defacer (Optional[Defacer]): The defacer object for defacing images.
        n4_bias_corrector (Optional[N4BiasCorrector]): The N4 bias corrector object for bias field correction. Defaults to SitkN4BiasCorrector with Otsu Thresholding.
        temp_folder (Optional[str or Path]): Path to a folder for storing intermediate results.
        use_gpu (Optional[bool]): Use GPU for processing if True, CPU if False, or automatically detect if None.
        limit_cuda_visible_devices (Optional[str]): Limit CUDA visible devices to a specific GPU ID.

    """

    def run(
        self,
        save_dir_coregistration: Optional[Union[str, Path]] = None,
        save_dir_n4_bias_correction: Optional[Union[str, Path]] = None,
        save_dir_brain_extraction: Optional[Union[str, Path]] = None,
        save_dir_defacing: Optional[Union[str, Path]] = None,
        save_dir_transformations: Optional[Union[str, Path]] = None,
        log_file: Optional[Union[str, Path]] = None,
    ):
        """
        Execute the native space preprocessing pipeline

        Args:
            save_dir_coregistration (str or Path, optional): Directory path to save intermediate coregistration results.
            save_dir_n4_bias_correction (str or Path, optional): Directory path to save intermediate N4 bias correction results.
            save_dir_brain_extraction (str or Path, optional): Directory path to save intermediate brain extraction results.
            save_dir_defacing (str or Path, optional): Directory path to save intermediate defacing results.
            save_dir_transformations (str or Path, optional): Directory path to save transformation matrices. Defaults to None.
            log_file (str or Path, optional): Path to save the log file. Defaults to a timestamped file in the current directory.

        This method orchestrates the entire preprocessing workflow by sequentially performing:

        1. Co-registration: Aligning moving modalities to the central modality.
        2. (Optional) N4 Bias Correction: Applying N4 bias field correction if specified.
        3. (Optional) Brain Extraction: Optionally extracting brain regions using specified masks. Only executed if any modality requires a brain extraction output (or a defacing output that requires prior brain extraction).
        4. (Optional) Defacing: Optionally deface images to remove facial features. Only executed if any modality requires a defacing output.

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
            logger.remove(logger_id)

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

        deface_mask = self.center_modality.deface(
            defacer=self.defacer,
            defaced_dir_path=deface_dir,
            registrator=self.registrator,
        )
        if deface_mask is None:
            return

        for mod in self.all_modalities:
            logger.info(f"Applying deface mask to {mod.modality_name}...")

            mod.apply_deface_mask(
                defacer=self.defacer,
                mask_path=deface_mask,
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
