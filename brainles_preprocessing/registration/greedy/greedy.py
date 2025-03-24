# TODO add typing and docs
from typing import Optional
import os

import itk
from picsl_greedy import Greedy3D

from brainles_preprocessing.registration.registrator import Registrator
from brainles_preprocessing.utils import check_and_add_suffix

"""
from picsl_greedy import Greedy3D

g = Greedy3D()

# Perform rigid registration
g.execute('-i phantom01_fixed.nii.gz phantom01_moving.nii.gz '
          '-gm phantom01_mask.nii.gz '
          '-a -dof 6 -n 40x10 -m NMI '
          '-o phantom01_rigid.mat')
          
# Apply rigid transform to moving image
g.execute('-rf phantom01_fixed.nii.gz '
          '-rm phantom01_moving.nii.gz phantom01_resliced.nii.gz '
          '-r phantom01_rigid.mat')
"""

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
        parameter_object: Optional[itk.ParameterObject] = None,
    ) -> None:
        """
        Register images using greedy.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output). This gets overwritten if it already exists.
            log_file_path (Optional[str]): Path to the log file.
            parameter_object (Optional[itk.ParameterObject]): The parameter object for greedy registration.
        """
        # initialize parameter object
        if parameter_object is None:
            parameter_object = self.__initialize_parameter_object()
        # add .txt suffix to the matrix path if it doesn't have any extension
        matrix_path = check_and_add_suffix(matrix_path, ".mat")
        
        # read images as itk images
        fixed_image = itk.imread(fixed_image_path)
        moving_image = itk.imread(moving_image_path)

        if log_file_path is not None:
            # split log_file_path
            log_path, log_file = os.path.split(log_file_path)
            result_image, result_transform_params = itk.elastix_registration_method(
                fixed_image,
                moving_image,
                parameter_object=parameter_object,
                log_to_file=True,
                log_file_name=log_file,
                output_directory=log_path,
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
        log_file_path: Optional[str] = None,
    ) -> None:
        """
        Apply a transformation using greedy.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output). This gets overwritten if it already exists.
            log_file_path (Optional[str]): Path to the log file.
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
        Initialize the parameter object for greedy registration.
        """
        parameter_object = itk.ParameterObject.New()
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")
        parameter_object.AddParameterMap(default_rigid_parameter_map)
        return parameter_object

