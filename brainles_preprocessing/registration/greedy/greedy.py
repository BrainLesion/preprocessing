# TODO add typing and docs
from typing import Optional
import contextlib
import os

from picsl_greedy import Greedy3D

from brainles_preprocessing.registration.registrator import Registrator
from brainles_preprocessing.utils import check_and_add_suffix


class GreedyRegistrator(Registrator):
    def __init__(
        self,
    ):
        pass

    def register(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: Optional[str] = None,
    ) -> None:
        """
        Register images using greedy. Ref: https://pypi.org/project/picsl-greedy/ and https://greedy.readthedocs.io/en/latest/reference.html#greedy-usage

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output). This gets overwritten if it already exists.
            log_file_path (Optional[str]): Path to the log file, which is not used.
        """
        # add .txt suffix to the matrix path if it doesn't have any extension
        matrix_path = check_and_add_suffix(matrix_path, ".mat")

        registor = Greedy3D()
        # these parameters are taken from the OG BraTS Pipeline [https://github.com/CBICA/CaPTk/blob/master/src/applications/BraTSPipeline.cxx]
        command_to_run = f"-i {fixed_image_path} {moving_image_path} -o {matrix_path} -a -dof 6 -m NMI -n 100x50x5 -ia-image-centers"

        if log_file_path is not None:
            with open(log_file_path, "a+") as f:
                with contextlib.redirect_stdout(f):
                    registor.execute(command_to_run)
        else:
            registor.execute(command_to_run)

        self.transform(
            fixed_image_path, moving_image_path, transformed_image_path, matrix_path
        )

    def transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: Optional[str] = None,
        interpolator: str = "LINEAR",
        **kwargs: Optional[dict],
    ) -> None:
        """
        Apply a transformation using greedy.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output). This gets overwritten if it already exists.
            log_file_path (Optional[str]): Path to the log file, which is not used.
            interpolator (Optional[str]): The interpolator to use; one of NN, LINEAR or LABEL. Defaults to LINEAR.
        """
        registor = Greedy3D()
        interpolator_upper = interpolator.upper()
        if "LABEL" in interpolator_upper:
            interpolator_upper += " 0.3vox"

        matrix_path = check_and_add_suffix(matrix_path, ".mat")

        if not os.path.exists(matrix_path):
            self.register(
                fixed_image_path,
                moving_image_path,
                transformed_image_path,
                matrix_path,
                log_file_path,
            )

        command_to_run = f"-rf {fixed_image_path} -rm {moving_image_path} {transformed_image_path} -r {matrix_path} -ri {interpolator_upper}"
        if log_file_path is not None:
            with open(log_file_path, "a+") as f:
                with contextlib.redirect_stdout(f):
                    registor.execute(command_to_run)
        else:
            registor.execute(command_to_run)

    def inverse_transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: Optional[str] = None,
        interpolator: str = "linear",
    ) -> None:
        raise NotImplementedError(
            "Inverse transform is not yet implemented for greedy."
        )
