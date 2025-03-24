# TODO add typing and docs
from typing import Optional

from picsl_greedy import Greedy3D

from brainles_preprocessing.registration.registrator import Registrator
from brainles_preprocessing.utils import check_and_add_suffix


class greedyRegistrator(Registrator):
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
        Register images using greedy. Ref: https://pypi.org/project/picsl-greedy/

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output). This gets overwritten if it already exists.
            log_file_path (Optional[str]): Path to the log file, which is not used.
        """
        # initialize parameter object
        if parameter_object is None:
            parameter_object = self.__initialize_parameter_object()
        # add .txt suffix to the matrix path if it doesn't have any extension
        matrix_path = check_and_add_suffix(matrix_path, ".mat")

        registor = Greedy3D()
        registor.execute(
            f"-i {fixed_image_path} {moving_image_path} "
            "-a -dof 6 -n 40x10 -m NMI "
            "-o {matrix_path}"
        )

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
    ) -> None:
        """
        Apply a transformation using greedy.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output). This gets overwritten if it already exists.
            log_file_path (Optional[str]): Path to the log file, which is not used.
        """
        registor = Greedy3D()
        registor.execute(
            f"-rf {fixed_image_path} -rm {moving_image_path} {transformed_image_path} -r {matrix_path}"
        )
