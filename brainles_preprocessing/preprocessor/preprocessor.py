import os
import shutil
import subprocess
import tempfile
import warnings
from abc import ABC, abstractmethod
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
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.registration.registrator import Registrator
from brainles_preprocessing.utils.logging_utils import LoggingManager
from brainles_preprocessing.utils.zenodo import verify_or_download_atlases

logging_man = LoggingManager(name=__name__)
logger = logging_man.get_logger()


class BasePreprocessor(ABC):
    """
    Base class to preprocesses medical image modalities.

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

    def __init__(
        self,
        center_modality: CenterModality,
        moving_modalities: List[Modality],
        registrator: Registrator = None,
        brain_extractor: Optional[BrainExtractor] = None,
        defacer: Optional[Defacer] = None,
        n4_bias_corrector: Optional[N4BiasCorrector] = None,
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

        self._check_for_name_conflicts()

        if n4_bias_corrector is None:
            n4_bias_corrector = SitkN4BiasCorrector()
        self.n4_bias_corrector = n4_bias_corrector

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

    def _check_for_name_conflicts(self):
        """
        Checks for name conflicts in the provided modalities.

        Raises:
            ValueError: If any modality name is non-unique.
        """

        name_counts = Counter(mod.modality_name for mod in self.all_modalities)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate modality names found: {', '.join(duplicates)}")

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

    @abstractmethod
    @ensure_remove_log_file_handler
    def run(self, *args, **kwargs):
        """
        Execute the preprocessing pipeline, encompassing
        """
        pass

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
        # Also set the center modality's coregistered step (same as input) for consistency
        self.center_modality.steps[PreprocessorSteps.COREGISTERED] = (
            self.center_modality.steps[PreprocessorSteps.INPUT]
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

    def run_n4_bias_correction(
        self,
        save_dir_n4_bias_correction: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Apply optional N4 bias correction to modalities.

        Args:
            save_dir_n4_bias_correction (Optional[Union[str, Path]], optional): Directory path to save intermediate N4 bias correction results. Defaults to None.
        """

        n4_bias_correction_dir = self.temp_folder / "n4-bias-correction"
        n4_bias_correction_dir.mkdir(exist_ok=True, parents=True)

        for modality in self.all_modalities:
            if modality.n4_bias_correction:
                logger.info(
                    f"Applying optional N4 bias correction for modality {modality.modality_name}"
                )

                output_path = (
                    n4_bias_correction_dir
                    / f"N4_bias_corrected__{modality.modality_name}.nii.gz"
                )

                self.n4_bias_corrector.correct(
                    input_img_path=str(modality.current),
                    output_img_path=str(output_path),
                )

                modality.steps[PreprocessorSteps.N4_BIAS_CORRECTED] = output_path
                modality.current = output_path
            else:
                logger.info(
                    f"Skipping optional N4 bias correction for Modality {modality.modality_name}."
                )

        self._save_output(
            src=n4_bias_correction_dir,
            save_dir=save_dir_n4_bias_correction,
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
