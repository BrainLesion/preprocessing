# TODO add typing and docs
import os

from auxiliary.turbopath import turbopath
from ereg.registration import RegistrationClass

from brainles_preprocessing.registration.registrator import Registrator


class eRegRegistrator(Registrator):
    def __init__(
        self,
    ):
        """
        # TODO
        """
        # TODO
        pass

    # TODO how to deal with the config file and the abstract class
    def register(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: str,
        # TODO default config file
        config_file: str,
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
        # TODO do we need to handle kwargs?
        registrator = RegistrationClass(config_file=config_file)

        # TODO we need a log file
        registrator.register(
            target_image=fixed_image_path,
            moving_image=moving_image_path,
            output_image=transformed_image_path,
            transform_file=matrix_path,
        )

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
        # TODO -> ereg should have a transform method, as this is a very common task! -> https://github.com/BrainLesion/eReg/issues/16
        pass
