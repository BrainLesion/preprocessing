# TODO add typing and docs
import os

import itk

from brainles_preprocessing.registration.registrator import Registrator


class elastixRegistrator(Registrator):
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
        log_file_path: str = None,
        parameter_object: itk.ParameterObject = None,
    ) -> None:
        """
        Register images using elastix.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output). This gets overwritten if it already exists.
            log_file_path (str): Path to the log file.
            parameter_object (itk.ParameterObject): Parameter object for elastix registration.
        """
        # initialize parameter object
        if parameter_object is None:
            parameter_object = self.__initialize_parameter_object()
        # add .txt suffix to the matrix path if it doesn't have any extension
        matrix_path = _add_txt_suffix(matrix_path)

        # read images as itk images
        fixed_image = itk.imread(fixed_image_path)
        moving_image = itk.imread(moving_image_path)

        if log_file_path is not None:
            result_image, result_transform_params = itk.elastix_registration_method(
                fixed_image,
                moving_image,
                parameter_object=parameter_object,
                log_to_file=True,
                log_file_name=log_file_path,
                output_directory=os.path.dirname(log_file_path),
            )
        else:
            result_image, result_transform_params = itk.elastix_registration_method(
                fixed_image,
                moving_image,
                parameter_object=parameter_object,
                log_to_console=True,
            )

        itk.imwrite(result_image, transformed_image_path)

        if not os.path.exists(matrix_path):
            result_transform_params.WriteParameterFile(
                result_transform_params.GetParameterMap(0),
                matrix_path,
            )

    def transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: str = None,
    ) -> None:
        """
        Apply a transformation using elastix.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output). This gets overwritten if it already exists.
            log_file_path (str): Path to the log file.
        """
        parameter_object = self.__initialize_parameter_object()

        # check if the matrix file exists
        if os.path.exists(matrix_path):
            parameter_object.SetParameter(
                0, "InitialTransformParametersFileName", matrix_path
            )

        self.register(
            fixed_image_path,
            moving_image_path,
            transformed_image_path,
            matrix_path,
            log_file_path,
            parameter_object,
        )

    def __initialize_parameter_object(self):
        """
        Initialize the parameter object for elastix registration.
        """
        parameter_object = itk.ParameterObject.New()
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")
        parameter_object.AddParameterMap(default_rigid_parameter_map)
        return parameter_object


def _add_txt_suffix(filename: str) -> str:
    """
    Adds a ".txt" suffix to the filename if it doesn't have any extension.

    Parameters:
        filename (str): The filename to check and potentially modify.

    Returns:
        str: The filename with ".txt" suffix added if needed.
    """
    if not filename.endswith(".txt"):
        filename += ".txt"
    return filename
