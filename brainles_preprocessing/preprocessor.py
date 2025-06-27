import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import List, Optional, Union
import warnings

from brainles_preprocessing.constants import Atlas, PreprocessorSteps
from brainles_preprocessing.defacing import Defacer, QuickshearDefacer
from brainles_preprocessing.utils.zenodo import verify_or_download_atlases

from .brain_extraction.brain_extractor import BrainExtractor, HDBetExtractor
from .modality import Modality, CenterModality
from .registration import ANTsRegistrator
from .registration.registrator import Registrator

from .utils.logging_utils import LoggingManager

logging_man = LoggingManager(name=__name__)
logger = logging_man.get_logger()


class Preprocessor:
    """
    Preprocesses medical image modalities using coregistration, normalization, brain extraction, and more.

    Args:
        center_modality (CenterModality): The central modality for coregistration.
        moving_modalities (List[Modality]): List of modalities to be coregistered to the central modality.
        registrator (Registrator): The registrator object for coregistration and registration to the atlas.
        brain_extractor (Optional[BrainExtractor]): The brain extractor object for brain extraction.
        defacer (Optional[Defacer]): The defacer object for defacing images.
        atlas_image_path (Optional[str or Path]): Path to the atlas image for registration (default is the T1 atlas).
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
        temp_folder: Optional[Union[str, Path]] = None,
        use_gpu: Optional[bool] = None,
        limit_cuda_visible_devices: Optional[str] = None,
    ):
        logging_man._setup_logger()

        if not isinstance(center_modality, CenterModality):
            warnings.warn(
                "Center modality should be of type CenterModality instead of Modality to allow for more features, e.g. saving bet and deface masks. "
                "Support for using Modality for the Center Modality will be deprecated in future versions. "
                "Note: Moving modalities should still be of type Modality.",
                category=DeprecationWarning,
            )
        self.center_modality = center_modality
        self.moving_modalities = moving_modalities

        if isinstance(atlas_image_path, Atlas):
            atlas_folder = verify_or_download_atlases()
            self.atlas_image_path = atlas_folder / atlas_image_path.value
        else:
            self.atlas_image_path = Path(atlas_image_path)

        self.registrator = registrator
        if self.registrator is None:
            logger.warning(
                "No registrator provided, using default ANTsRegistrator for registration."
            )
            self.registrator = ANTsRegistrator()

        self.brain_extractor = brain_extractor
        self.defacer = defacer

        self._configure_gpu(
            use_gpu=use_gpu, limit_cuda_visible_devices=limit_cuda_visible_devices
        )

        # Create temporary storage
        if temp_folder:
            temp_folder = Path(temp_folder)
            temp_folder.mkdir(parents=True, exist_ok=True)
            self.temp_folder = temp_folder
        else:
            storage = tempfile.TemporaryDirectory()
            self.temp_folder = Path(storage.name)

    def _configure_gpu(
        self, use_gpu: Optional[bool], limit_cuda_visible_devices: Optional[str] = None
    ) -> None:
        """
        Configures the environment for GPU usage based on the `use_gpu` parameter and CUDA availability.

        Args:
            use_gpu (Optional[bool]): Determines the GPU usage strategy.
        """
        if use_gpu or (use_gpu is None and self._cuda_is_available()):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if limit_cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = limit_cuda_visible_devices

    @staticmethod
    def _cuda_is_available() -> bool:
        """
        Checks if CUDA is available on the system by attempting to run 'nvidia-smi'.

        Returns:
            bool: True if 'nvidia-smi' can be executed successfully, indicating CUDA is available.
        """
        try:
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def ensure_remove_log_file_handler(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                self = args[0]
                if isinstance(self, Preprocessor):
                    logging_man.remove_log_file_handler()

        return wrapper

    @property
    def all_modalities(self) -> List[Modality]:
        """
        Returns a list of all modalities including the center modality.
        """
        return [self.center_modality] + self.moving_modalities

    @property
    def requires_defacing(self) -> bool:
        """
        Returns True if any modality requires defacing otherwise returns False.
        """
        return any(modality.requires_deface for modality in self.all_modalities)

    @ensure_remove_log_file_handler
    def run(
        self,
        save_dir_coregistration: Optional[Union[str, Path]] = None,
        save_dir_atlas_registration: Optional[Union[str, Path]] = None,
        save_dir_atlas_correction: Optional[Union[str, Path]] = None,
        save_dir_brain_extraction: Optional[Union[str, Path]] = None,
        save_dir_defacing: Optional[Union[str, Path]] = None,
        save_dir_transformations: Optional[Union[str, Path]] = None,
        log_file: Optional[Union[str, Path]] = None,
    ):
        """
        Execute the preprocessing pipeline, encompassing coregistration, atlas-based registration,
        atlas correction, and optional brain extraction.

        Args:
            save_dir_coregistration (str or Path, optional): Directory path to save intermediate coregistration results.
            save_dir_atlas_registration (str or Path, optional): Directory path to save intermediate atlas registration results.
            save_dir_atlas_correction (str or Path, optional): Directory path to save intermediate atlas correction results.
            save_dir_brain_extraction (str or Path, optional): Directory path to save intermediate brain extraction results.
            save_dir_defacing (str or Path, optional): Directory path to save intermediate defacing results.
            save_dir_transformations (str or Path, optional): Directory path to save transformation matrices. Defaults to None.
            log_file (str or Path, optional): Path to save the log file. Defaults to a timestamped file in the current directory.

        This method orchestrates the entire preprocessing workflow by sequentially performing:

        1. Co-registration: Aligning moving modalities to the central modality.
        2. Atlas Registration: Aligning the central modality to a predefined atlas.
        3. Atlas Correction: Applying additional correction in atlas space if specified.
        4. Brain Extraction: Optionally extracting brain regions using specified masks. Only executed if any modality requires a brain extraction output (or a defacing output that requires prior brain extraction).
        5. Defacing: Optionally deface images to remove facial features. Only executed if any modality requires a defacing output.

        Results are saved in the specified directories, allowing for modular and configurable output storage.
        """

        logging_man._set_log_file(log_file)
        logger.info(f"{' Starting preprocessing ':=^80}")
        logger.info(f"Logs are saved to {logging_man.log_file_handler.baseFilename}")
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
            logger.info(f"Saving transformation matrices to {save_dir_transformations}")
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

    def run_coregistration(
        self, save_dir_coregistration: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Coregister moving modalities to center modality.

        Args:
            save_dir_coregistration (str, optional): Directory path to save intermediate coregistration results.
        """
        coregistration_dir = self.temp_folder / "coregistration"
        coregistration_dir.mkdir(exist_ok=True, parents=True)

        logger.info(
            f"Coregistering {len(self.moving_modalities)} moving modalities to center modality..."
        )
        for moving_modality in self.moving_modalities:
            file_name = f"co__{self.center_modality.modality_name}__{moving_modality.modality_name}"
            logger.info(
                f"Registering modality {moving_modality.modality_name} (file={file_name}) to center modality..."
            )
            moving_modality.register(
                registrator=self.registrator,
                fixed_image_path=self.center_modality.current,
                registration_dir=coregistration_dir,
                moving_image_name=file_name,
                step=PreprocessorSteps.COREGISTERED,
            )

        shutil.copyfile(
            src=str(self.center_modality.input_path),
            dst=str(
                coregistration_dir
                / f"native__{self.center_modality.modality_name}.nii.gz"
            ),
        )

        self._save_output(
            src=coregistration_dir,
            save_dir=save_dir_coregistration,
        )

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

    def run_brain_extraction(
        self, save_dir_brain_extraction: Optional[Union[str, Path]] = None
    ) -> None:
        """Extract brain regions using specified BrainExtractor.

        Args:
            save_dir_brain_extraction (Optional[str or Path], optional): Directory path to save intermediate brain extraction results. Defaults to None.
        """
        # Check if any bet output paths are requested
        brain_extraction = any(modality.bet for modality in self.all_modalities)

        # Check if any downstream task (e.g. QuickShear) requires brain extraction.
        # Quickshear is the default defacer so we also require bet if no defacer is specified
        required_downstream = self.requires_defacing and (
            isinstance(self.defacer, QuickshearDefacer) or self.defacer is None
        )

        # Skip if no brain extraction is required
        if not brain_extraction and not required_downstream:
            logger.info("Skipping brain extraction.")
            return

        logger.info(
            f"Starting brain extraction{' (for downstream defacing task)' if (required_downstream and not brain_extraction) else ''}..."
        )

        # Setup output dirs
        bet_dir = self.temp_folder / "brain-extraction"
        bet_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Extracting brain region for center modality...")

        # Assert that a brain extractor is specified (since the arg is optional)
        if self.brain_extractor is None:
            logger.warning(
                "Brain extraction is required to compute specified outputs but no brain extractor was specified during class initialization."
                + " Using default `brainles_preprocessing.brain_extraction.HDBetExtractor`"
            )
            self.brain_extractor = HDBetExtractor()

        atlas_mask = self.center_modality.extract_brain_region(
            brain_extractor=self.brain_extractor, bet_dir_path=bet_dir
        )
        for moving_modality in self.moving_modalities:
            logger.info(f"Applying brain mask to {moving_modality.modality_name}...")
            moving_modality.apply_bet_mask(
                brain_extractor=self.brain_extractor,
                mask_path=atlas_mask,
                bet_dir=bet_dir,
            )

        self._save_output(
            src=bet_dir,
            save_dir=save_dir_brain_extraction,
        )

        # now we save images that are skullstripped
        logger.info("Saving brain extracted (bet), i.e. skull-stripped images...")
        for modality in self.all_modalities:
            if modality.raw_bet_output_path:
                modality.save_current_image(
                    modality.raw_bet_output_path,
                    normalization=False,
                )
            if modality.normalized_bet_output_path:
                modality.save_current_image(
                    modality.normalized_bet_output_path,
                    normalization=True,
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

    def _save_output(
        self,
        src: Union[str, Path],
        save_dir: Optional[Union[str, Path]],
    ):
        """
        Save the output from a source directory to the specified save directory.
        """
        if save_dir:
            save_dir = Path(save_dir)
            shutil.copytree(
                src=str(src),
                dst=str(save_dir),
                dirs_exist_ok=True,
            )
