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
    ) -> None:
        """
        Register images using eReg.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output).
            log_file_path (str): Path to the log file.
        """

        fixed_image = itk.imread(fixed_image_path)
        moving_image = itk.imread(moving_image_path)

        elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
        elastix_object.SetParameterObject(self.__initialize_parameter_object())

        if os.path.exists(matrix_path):
            self.result_transform_parameters.LoadParameterFile(matrix_path)
            elastix_object.SetTransformParameterObject(self.result_transform_parameters)

        # Set additional options
        if log_file_path is None:
            elastix_object.SetLogToConsole(True)
        else:
            elastix_object.SetOutputDirectory(os.path.dirname(log_file_path))
            elastix_object.SetLogToFile(True)
            elastix_object.SetLogFileName(os.path.basename(log_file_path))

        # Update filter object (required)
        elastix_object.UpdateLargestPossibleRegion()

        # Results of Registration
        result_image = elastix_object.GetOutput()
        result_transform_parameters = elastix_object.GetTransformParameterObject()

        itk.imwrite(result_image, transformed_image_path)
        result_transform_parameters.WriteParameterFile(matrix_path)

    def transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: str = None,
    ) -> None:
        """
        Apply a transformation using eReg.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix.
            log_file_path (str): Path to the log file.
        """
        self.register(
            fixed_image_path,
            moving_image_path,
            transformed_image_path,
            matrix_path,
            log_file_path,
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
