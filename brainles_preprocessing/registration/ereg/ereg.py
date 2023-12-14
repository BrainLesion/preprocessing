# TODO add typing and docs
import os

from auxiliary.turbopath import turbopath
from ereg.registration import RegistrationClass

from brainles_preprocessing.registration.registrator import Registrator


class eRegRegistrator(Registrator):
    def __init__(
        self,
        # TODO define default
        config_file: str,
    ):
        """
        # TODO
        """
        self.config_file = config_file

    # TODO how to deal with the config file and the abstract class
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
        # config_file = kwargs["config_file"]
        # TODO do we need to handle kwargs?
        registrator = RegistrationClass(
            config_file=self.config_file,
        )
        
        registrator.register(
            target_image=fixed_image_path,
            moving_image=moving_image_path,
            output_image=transformed_image_path,
            transform_file=matrix_path,
            # TODO we need a log file
        )

        # registrator.config_file

    def transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: str,
        # TODO default config file
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
        # TODO do we need to handle kwargs?
        registrator = RegistrationClass(
            config_file=self.config_file,
        )

        registrator.resample_image(
            target_image=fixed_image_path,
            moving_image=moving_image_path,
            output_image=transformed_image_path,
            # TODO how can this default to none
            transform_file=matrix_path,
            # TODO we need to deal with logging
        )
