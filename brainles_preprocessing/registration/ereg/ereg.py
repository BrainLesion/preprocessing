# TODO add typing and docs
import os

from auxiliary.runscript import ScriptRunner
from auxiliary.turbopath import turbopath

from brainles_preprocessing.registration.registrator import Registrator

# from auxiliary import ScriptRunner


class eRegRegistrator(Registrator):
    def __init__(
        self,
    ):
        """
        # TODO
        """
        # TODO
        pass

    def register(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: str,
    ) -> None:
        """
        Register images using NiftyReg.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output).
            log_file_path (str): Path to the log file.
        """
        # TODO
        pass

    def transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: str,
    ) -> None:
        """
        Apply a transformation using NiftyReg.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix.
            log_file_path (str): Path to the log file.
        """
        # TODO https://github.com/BrainLesion/eReg/blob/11f8024176a37ee3c7ab8b8e42669d08c3f6f7d0/ereg/registration.py#L317
        pass
