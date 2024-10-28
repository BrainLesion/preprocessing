# TODO add typing and docs
import datetime
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ants

from brainles_preprocessing.registration.registrator import Registrator


class ANTsRegistrator(Registrator):
    def __init__(
        self,
        registration_params: Optional[Dict[str, Any]] = None,
        transformation_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an ANTsRegistrator instance.

        Parameters:
        - registration_params (dict, optional): Dictionary of parameters for the registration method.
          Defaults to None, which implies using default registration parameters with a rigid transformation.
        - transformation_params (dict, optional): Dictionary of parameters for the transformation method.
          Defaults to an empty dictionary.

        The registration_params dictionary may include the following keys:
        - type_of_transform (str, optional): Type of transformation to use (default is "Rigid").

        Example:
        >>> reg_params = {'type_of_transform': 'Affine', 'reg_iterations': (30, 20, 10)}
        >>> transform_params = {'interpolator': 'linear', 'imagetype': 1}
        >>> registrator = ANTsRegistrator(registration_params=reg_params, transformation_params=transform_params)
        """
        # Set default registration parameters
        default_registration_params = {"type_of_transform": "Rigid"}
        self.registration_params = registration_params or default_registration_params

        # Set default transformation parameters
        self.transformation_params = transformation_params or {}

    def register(
        self,
        fixed_image_path: Union[str, Path],
        moving_image_path: Union[str, Path],
        transformed_image_path: Union[str, Path],
        matrix_path: Union[str, Path],
        log_file_path: Union[str, Path],
        **kwargs,
    ) -> None:
        """
        Register images using ANTs.

        Args:
            fixed_image_path (str or Path): Path to the fixed image.
            moving_image_path (str or Path): Path to the moving image.
            transformed_image_path (str or Path): Path to the transformed image (output).
            matrix_path (str or Path): Path to the transformation matrix (output).
            log_file_path (str or Path): Path to the log file.
            **kwargs: Additional registration parameters to update the instantiated defaults.
        """
        start_time = datetime.datetime.now()

        # TODO - self.registration_params
        # We update the registration parameters with the provided kwargs
        registration_kwargs = {**self.registration_params, **kwargs}

        # Convert all paths to Path objects
        fixed_image_path = Path(fixed_image_path)
        moving_image_path = Path(moving_image_path)
        transformed_image_path = Path(transformed_image_path)
        matrix_path = Path(matrix_path)
        log_file_path = Path(log_file_path)

        if not fixed_image_path.is_file():
            raise FileNotFoundError(f"Fixed image not found: {fixed_image_path}")
        if not moving_image_path.is_file():
            raise FileNotFoundError(f"Moving image not found: {moving_image_path}")

        # Ensure matrix_path has .mat suffix
        if matrix_path.suffix != ".mat":
            matrix_path = matrix_path.with_suffix(".mat")

        fixed_image = ants.image_read(str(fixed_image_path))
        moving_image = ants.image_read(str(moving_image_path))
        registration_result = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            **registration_kwargs,
        )
        transformed_image = registration_result["warpedmovout"]

        # Ensure output directories exist
        transformed_image_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)

        ants.image_write(transformed_image, str(transformed_image_path))

        shutil.copyfile(
            src=registration_result["fwdtransforms"][0],
            dst=str(matrix_path),
        )

        end_time = datetime.datetime.now()

        # TODO nicer logging
        # we create a dummy log file for the moment to pass the tests

        self._log_to_file(
            log_file_path=log_file_path,
            fixed_image_path=fixed_image_path,
            moving_image_path=moving_image_path,
            transformed_image_path=transformed_image_path,
            matrix_path=matrix_path,
            operation_name="registration",
            start_time=start_time,
            end_time=end_time,
        )

    def transform(
        self,
        fixed_image_path: Union[str, Path],
        moving_image_path: Union[str, Path],
        transformed_image_path: Union[str, Path],
        matrix_path: Union[str, Path],
        log_file_path: Union[str, Path],
        **kwargs,
    ) -> None:
        """
        Apply a transformation using ANTs.

        Args:
            fixed_image_path (str or Path): Path to the fixed image.
            moving_image_path (str or Path): Path to the moving image.
            transformed_image_path (str or Path): Path to the transformed image (output).
            matrix_path (str or Path): Path to the transformation matrix.
            log_file_path (str or Path): Path to the log file.
            **kwargs: Additional transformation parameters to update the instantiated defaults.
        """
        start_time = datetime.datetime.now()

        # TODO - self.transformation_params
        # we update the transformation parameters with the provided kwargs
        transform_kwargs = {**self.transformation_params, **kwargs}

        # Convert all paths to Path objects
        fixed_image_path = Path(fixed_image_path)
        moving_image_path = Path(moving_image_path)
        transformed_image_path = Path(transformed_image_path)
        matrix_path = Path(matrix_path)
        log_file_path = Path(log_file_path)

        if not fixed_image_path.is_file():
            raise FileNotFoundError(f"Fixed image not found: {fixed_image_path}")
        if not moving_image_path.is_file():
            raise FileNotFoundError(f"Moving image not found: {moving_image_path}")

        fixed_image = ants.image_read(str(fixed_image_path))
        moving_image = ants.image_read(str(moving_image_path))

        # Ensure output directory exist
        transformed_image_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure matrix_path has .mat suffix
        if matrix_path.suffix != ".mat":
            matrix_path = matrix_path.with_suffix(".mat")

        transformed_image = ants.apply_transforms(
            fixed=fixed_image,
            moving=moving_image,
            transformlist=[str(matrix_path)],
            **transform_kwargs,
        )
        ants.image_write(transformed_image, str(transformed_image_path))

        end_time = datetime.datetime.now()

        # TODO nicer logging
        # we create a dummy log file for the moment to pass the tests

        self._log_to_file(
            log_file_path=log_file_path,
            fixed_image_path=fixed_image_path,
            moving_image_path=moving_image_path,
            transformed_image_path=transformed_image_path,
            matrix_path=matrix_path,
            operation_name="transformation",
            start_time=start_time,
            end_time=end_time,
        )

    @staticmethod
    def _log_to_file(
        log_file_path: Union[str, Path],
        fixed_image_path: Union[str, Path],
        moving_image_path: Union[str, Path],
        transformed_image_path: Union[str, Path],
        matrix_path: Union[str, Path],
        operation_name: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        """
        Log the operation details to a file.

        Args:
            log_file_path (str or Path): Path to the log file.
            fixed_image_path (str or Path): Path to the fixed image.
            moving_image_path (str or Path): Path to the moving image.
            transformed_image_path (str or Path): Path to the transformed image.
            matrix_path (str or Path): Path to the transformation matrix.
            operation_name (str): Name of the operation ('registration' or 'transformation').
            start_time (datetime.datetime): Start time of the operation.
            end_time (datetime.datetime): End time of the operation.
        """

        # Calculate the duration and make it human readable
        duration = (end_time - start_time).total_seconds()

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        milliseconds = int((duration - int(duration)) * 1000)

        # Format the duration as "0:0:0:0"
        duration_formatted = f"{hours}h {minutes}m {seconds}s {milliseconds}ms"

        with open(str(log_file_path), "w") as f:
            f.write(f"*** {operation_name} with antspyx ***\n")
            f.write(f"start time: {start_time} \n")
            f.write(f"fixed image: {fixed_image_path} \n")
            f.write(f"moving image: {moving_image_path} \n")
            f.write(f"transformed image: {transformed_image_path} \n")
            f.write(f"matrix: {matrix_path} \n")
            f.write(f"end time: {end_time} \n")
            f.write(f"duration: {duration_formatted}\n")


if __name__ == "__main__":
    # TODO move this into unit tests
    reg = ANTsRegistrator()

    reg.register(
        fixed_image_path="example/example_data/TCGA-DU-7294/AX_T1_POST_GD_FLAIR_TCGA-DU-7294_TCGA-DU-7294_GE_TCGA-DU-7294_AX_T1_POST_GD_FLAIR_RM_13_t1c.nii.gz",
        moving_image_path="example/example_data/TCGA-DU-7294/AX_T2_FR-FSE_RF2_150_TCGA-DU-7294_TCGA-DU-7294_GE_TCGA-DU-7294_AX_T2_FR-FSE_RF2_150_RM_4_t2.nii.gz",
        transformed_image_path="example/example_ants/transformed_image.nii.gz",
        matrix_path="example/example_ants_matrix/matrix",
        log_file_path="example/example_ants/log.txt",
    )

    reg.transform(
        fixed_image_path="example/example_data/TCGA-DU-7294/AX_T1_POST_GD_FLAIR_TCGA-DU-7294_TCGA-DU-7294_GE_TCGA-DU-7294_AX_T1_POST_GD_FLAIR_RM_13_t1c.nii.gz",
        moving_image_path="example/example_data/OtherEXampleFromTCIA/T1_AX_OtherEXampleTCIA_TCGA-FG-6692_Si_TCGA-FG-6692_T1_AX_SE_10_se2d1_t1.nii.gz",
        transformed_image_path="example/example_ants_transformed/transformed_image.nii.gz",
        matrix_path="example/example_ants_matrix/matrix.mat",
        log_file_path="example/example_ants/log.txt",
    )
