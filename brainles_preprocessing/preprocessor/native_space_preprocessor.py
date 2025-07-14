import os
import shutil
import subprocess
import tempfile
import warnings
from abc import ABC
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import List, Optional, Union

from brainles_preprocessing.brain_extraction.brain_extractor import (
    BrainExtractor,
    HDBetExtractor,
)
from brainles_preprocessing.constants import Atlas, PreprocessorSteps
from brainles_preprocessing.defacing import Defacer, QuickshearDefacer
from brainles_preprocessing.modality import CenterModality, Modality
from brainles_preprocessing.n4_bias_correction import (
    N4BiasCorrector,
    SitkN4BiasCorrector,
)
from brainles_preprocessing.preprocessor.preprocessor import BasePreprocessor
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.registration.registrator import Registrator
from brainles_preprocessing.utils.logging_utils import LoggingManager
from brainles_preprocessing.utils.zenodo import verify_or_download_atlases

logging_man = LoggingManager(name=__name__)
logger = logging_man.get_logger()


class NativeSpacePreprocessor(BasePreprocessor):

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
        Execute the non-atlas centric native space preprocessing pipeline

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

        for mod in self.all_modalities:
            mask = mod.extract_brain_region(
                brain_extractor=self.brain_extractor, bet_dir_path=bet_dir
            )
            logger.info(f"Applying brain mask to {mod.modality_name}...")
            mod.apply_bet_mask(
                brain_extractor=self.brain_extractor,
                mask_path=mask,
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

        for mod in self.all_modalities:
            logger.info(f"Applying deface mask to {mod.modality_name}...")

            mask = mod.deface(defacer=self.defacer, defaced_dir_path=deface_dir)
            mod.apply_deface_mask(
                defacer=self.defacer,
                mask_path=mask,
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
