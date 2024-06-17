from functools import wraps
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from typing import List, Optional

from auxiliary.turbopath import turbopath

from .brain_extraction.brain_extractor import BrainExtractor
from .modality import Modality
from .registration.registrator import Registrator

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocesses medical image modalities using coregistration, normalization, brain extraction, and more.

    Args:
        center_modality (Modality): The central modality for coregistration.
        moving_modalities (List[Modality]): List of modalities to be coregistered to the central modality.
        registrator (Registrator): The registrator object for coregistration and registration to the atlas.
        brain_extractor (BrainExtractor): The brain extractor object for brain extraction.
        atlas_image_path (str, optional): Path to the atlas image for registration (default is the T1 atlas).
        temp_folder (str, optional): Path to a temporary folder for storing intermediate results.
        use_gpu (Optional[bool]): Use GPU for processing if True, CPU if False, or automatically detect if None.
        limit_cuda_visible_devices (Optional[str]): Limit CUDA visible devices to a specific GPU ID.

    """

    def __init__(
        self,
        center_modality: Modality,
        moving_modalities: List[Modality],
        registrator: Registrator,
        brain_extractor: BrainExtractor,
        atlas_image_path: str = turbopath(__file__).parent
        + "/registration/atlas/t1_brats_space.nii",
        temp_folder: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        limit_cuda_visible_devices: Optional[str] = None,
    ):
        self._setup_logger()

        self.center_modality = center_modality
        self.moving_modalities = moving_modalities
        self.atlas_image_path = turbopath(atlas_image_path)
        self.registrator = registrator
        self.brain_extractor = brain_extractor

        self._configure_gpu(
            use_gpu=use_gpu, limit_cuda_visible_devices=limit_cuda_visible_devices
        )

        # Create temporary storage
        if temp_folder:
            os.makedirs(temp_folder, exist_ok=True)
            self.temp_folder = turbopath(temp_folder)
        else:
            storage = tempfile.TemporaryDirectory()
            self.temp_folder = turbopath(storage.name)

        self.atlas_dir = os.path.join(self.temp_folder, "atlas-space")
        os.makedirs(self.atlas_dir, exist_ok=True)

    def _configure_gpu(
        self, use_gpu: Optional[bool], limit_cuda_visible_devices: Optional[str] = None
    ):
        """
        Configures the environment for GPU usage based on the `use_gpu` parameter and CUDA availability.

        Args:
            use_gpu (Optional[bool]): Determines the GPU usage strategy.
        """
        if use_gpu is True or (use_gpu is None and self._cuda_is_available()):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            if limit_cuda_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = limit_cuda_visible_devices

    @staticmethod
    def _cuda_is_available():
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
                if isinstance(self, Preprocessor) and self.log_file_handler:
                    logging.getLogger().removeHandler(self.log_file_handler)

        return wrapper

    def _set_log_file(self, log_file: Optional[str | Path]) -> None:
        """Set the log file and remove the file handler from a potential previous run.

        Args:
            log_file (str | Path): log file path
        """
        if self.log_file_handler:
            logging.getLogger().removeHandler(self.log_file_handler)

        # ensure parent directories exists
        log_file = Path(
            log_file
            if log_file
            else f"brainles_preprocessing_{datetime.now().isoformat()}.log"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

        self.log_file_handler = logging.FileHandler(log_file)
        self.log_file_handler.setFormatter(
            logging.Formatter(
                "[%(levelname)-8s | %(module)-15s | L%(lineno)-5d] %(asctime)s: %(message)s",
                "%Y-%m-%dT%H:%M:%S%z",
            )
        )

        # Add the file handler to the !root! logger
        logging.getLogger().addHandler(self.log_file_handler)

    def _setup_logger(self):
        """Setup the logger and overwrite system hooks to add logging for exceptions and signals."""

        logging.basicConfig(
            format="[%(levelname)s] %(asctime)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
            level=logging.INFO,
        )
        self.log_file_handler = None

        # overwrite system hooks to log exceptions and signals (SIGINT, SIGTERM)
        #! NOTE: This will note work in Jupyter Notebooks, (Without extra setup) see https://stackoverflow.com/a/70469055:
        def exception_handler(exception_type, value, tb):
            """Handle exceptions

            Args:
                exception_type (Exception): Exception type
                exception (Exception): Exception
                traceback (Traceback): Traceback
            """
            logger.error("".join(traceback.format_exception(exception_type, value, tb)))

            if issubclass(exception_type, SystemExit):
                # add specific code if exception was a system exit
                sys.exit(value.code)

        def signal_handler(sig, frame):
            signame = signal.Signals(sig).name
            logger.error(f"Received signal {sig} ({signame}), exiting...")
            sys.exit(0)

        sys.excepthook = exception_handler

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    @property
    def all_modalities(self):
        return [self.center_modality] + self.moving_modalities

    @ensure_remove_log_file_handler
    def run(
        self,
        save_dir_coregistration: Optional[str] = None,
        save_dir_atlas_registration: Optional[str] = None,
        save_dir_atlas_correction: Optional[str] = None,
        save_dir_brain_extraction: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        """
        Execute the preprocessing pipeline, encompassing coregistration, atlas-based registration,
        atlas correction, and optional brain extraction.

        Args:
            save_dir_coregistration (str, optional): Directory path to save coregistration results.
            save_dir_atlas_registration (str, optional): Directory path to save atlas registration results.
            save_dir_atlas_correction (str, optional): Directory path to save atlas correction results.
            save_dir_brain_extraction (str, optional): Directory path to save brain extraction results.
            log_file (str, optional): Path to save the log file. Defaults to a timestamped file in the current directory.

        This method orchestrates the entire preprocessing workflow by sequentially performing:

        1. Coregistration: Aligning moving modalities to the central modality.
        2. Atlas Registration: Aligning the central modality to a predefined atlas.
        3. Atlas Correction: Applying additional correction in atlas space if specified.
        4. Brain Extraction: Optionally extracting brain regions using specified masks.

        Results are saved in the specified directories, allowing for modular and configurable output storage.
        """
        self._set_log_file(log_file=log_file)
        logger.info(f"{' Starting preprocessing ':=^80}")
        logger.info(f"Logs are saved to {self.log_file_handler.baseFilename}")
        logger.info(
            f"Received center modality: {self.center_modality.modality_name} and moving modalities: {', '.join([modality.modality_name for modality in self.moving_modalities])}"
        )

        logger.info(f"{' Starting Coregistration ':-^80}")

        # Coregister moving modalities to center modality
        coregistration_dir = os.path.join(self.temp_folder, "coregistration")
        os.makedirs(coregistration_dir, exist_ok=True)
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
            )

        shutil.copyfile(
            src=self.center_modality.input_path,
            dst=os.path.join(
                coregistration_dir,
                f"native__{self.center_modality.modality_name}.nii.gz",
            ),
        )

        self._save_output(
            src=coregistration_dir,
            save_dir=save_dir_coregistration,
        )
        logger.info(
            f"Coregistration complete. Output saved to {save_dir_coregistration}"
        )

        # Register center modality to atlas
        logger.info(f"{' Starting atlas registration ':-^80}")
        logger.info(f"Registering center modality to atlas...")
        center_file_name = f"atlas__{self.center_modality.modality_name}"
        transformation_matrix = self.center_modality.register(
            registrator=self.registrator,
            fixed_image_path=self.atlas_image_path,
            registration_dir=self.atlas_dir,
            moving_image_name=center_file_name,
        )
        logger.info(f"Atlas registration complete. Output saved to {self.atlas_dir}")

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
                registration_dir_path=self.atlas_dir,
                moving_image_name=moving_file_name,
                transformation_matrix_path=transformation_matrix,
            )
        self._save_output(
            src=self.atlas_dir,
            save_dir=save_dir_atlas_registration,
        )
        logger.info(
            f"Transformations complete. Output saved to {save_dir_atlas_registration}"
        )

        # Optional: additional correction in atlas space
        logger.info(f"{' Checking optional atlas correction ':-^80}")
        atlas_correction_dir = os.path.join(self.temp_folder, "atlas-correction")
        os.makedirs(atlas_correction_dir, exist_ok=True)

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
                )
            else:
                logger.info("Skipping optional atlas correction.")

        if self.center_modality.atlas_correction:
            shutil.copyfile(
                src=self.center_modality.current,
                dst=os.path.join(
                    atlas_correction_dir,
                    f"atlas_corrected__{self.center_modality.modality_name}.nii.gz",
                ),
            )
            logger.info(
                f"Atlas correction complete. Output saved to {save_dir_atlas_correction}"
            )

        self._save_output(
            src=atlas_correction_dir,
            save_dir=save_dir_atlas_correction,
        )

        # now we save images that are not skullstripped
        logger.info("Saving non skull-stripped images...")
        for modality in self.all_modalities:
            if modality.raw_skull_output_path is not None:
                modality.save_current_image(
                    modality.raw_skull_output_path,
                    normalization=False,
                )
            if modality.normalized_skull_output_path is not None:
                modality.save_current_image(
                    modality.normalized_skull_output_path,
                    normalization=True,
                )
        # Optional: Brain extraction
        logger.info(f"{' Checking optional brain extraction ':-^80}")
        brain_extraction = any(modality.bet for modality in self.all_modalities)
        # print("brain extraction: ", brain_extraction)

        if brain_extraction:
            logger.info("Starting brain extraction...")
            bet_dir = os.path.join(self.temp_folder, "brain-extraction")
            os.makedirs(bet_dir, exist_ok=True)
            brain_masked_dir = os.path.join(bet_dir, "brain_masked")
            os.makedirs(brain_masked_dir, exist_ok=True)
            logger.info("Extracting brain region for center modality...")
            atlas_mask = self.center_modality.extract_brain_region(
                brain_extractor=self.brain_extractor, bet_dir_path=bet_dir
            )
            for moving_modality in self.moving_modalities:
                logger.info(
                    f"Applying brain mask to {moving_modality.modality_name}..."
                )
                moving_modality.apply_mask(
                    brain_extractor=self.brain_extractor,
                    brain_masked_dir_path=brain_masked_dir,
                    atlas_mask_path=atlas_mask,
                )

            self._save_output(
                src=bet_dir,
                save_dir=save_dir_brain_extraction,
            )
            logger.info(
                f"Brain extraction complete. Output saved to {save_dir_brain_extraction}"
            )
        else:
            logger.info("Skipping optional brain extraction.")

        # now we save images that are skullstripped
        logger.info("Saving skull-stripped images...")
        for modality in self.all_modalities:
            if modality.raw_bet_output_path is not None:
                modality.save_current_image(
                    modality.raw_bet_output_path,
                    normalization=False,
                )
            if modality.normalized_bet_output_path is not None:
                modality.save_current_image(
                    modality.normalized_bet_output_path,
                    normalization=True,
                )
        logger.info(f"{' Preprocessing complete ':=^80}")

    def _save_output(
        self,
        src: str,
        save_dir: Optional[str],
    ):
        if save_dir is not None:
            save_dir = turbopath(save_dir)
            shutil.copytree(
                src=src,
                dst=save_dir,
                dirs_exist_ok=True,
            )
