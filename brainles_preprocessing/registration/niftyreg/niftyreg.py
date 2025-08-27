from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from auxiliary.runscript import ScriptRunner
from auxiliary.turbopath import turbopath
from auxiliary.io import read_image

from brainles_preprocessing.registration.registrator import Registrator

VALID_INTERPOLATORS = {
    "0": "nearestNeighbor",
    "1": "linear",
    "3": "cubicSpline",
    "4": "sinc",
}


class NiftyRegRegistrator(Registrator):
    def __init__(
        self,
        registration_abspath: str = os.path.dirname(os.path.abspath(__file__)),
        registration_script: str | None = None,
        transformation_script: str | None = None,
    ):
        """
        Initialize the NiftyRegRegistrator.

        Args:
            registration_abspath (str): Absolute path to the registration directory.
            registration_script (str, optional): Path to the registration script. If None, a default script will be used.
            transformation_script (str, optional): Path to the transformation script. If None, a default script will be used.
        """
        # Set default registration script
        if registration_script is None:
            self.registration_script = os.path.join(
                registration_abspath, "niftyreg_scripts", "rigid_reg.sh"
            )
        else:
            self.registration_script = registration_script

        # Set default transformation script
        if transformation_script is None:
            self.transformation_script = os.path.join(
                registration_abspath, "niftyreg_scripts", "transform.sh"
            )
        else:
            self.transformation_script = transformation_script

    def register(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str | Path,
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
        runner = ScriptRunner(
            script_path=self.registration_script,
            log_path=log_file_path,
        )

        niftyreg_executable = str(
            Path(__file__).parent.absolute() / "niftyreg_scripts" / "reg_aladin",
        )

        matrix_path = Path(matrix_path).with_suffix(".txt")

        # read moving image to get padding value
        padding_value = float(read_image(moving_image_path).min())

        input_params = [
            turbopath(niftyreg_executable),
            turbopath(fixed_image_path),
            turbopath(moving_image_path),
            turbopath(transformed_image_path),
            str(matrix_path),
            str(padding_value),
        ]

        # Call the run method to execute the script and capture the output in the log file
        success, error = runner.run(input_params)

        # if success:
        #     print("Script executed successfully. Check the log file for details.")
        # else:
        #     print("Script execution failed:", error)

    def _compose_affine_transforms(
        self, transform_paths: List[str | Path], output_path: str | Path
    ) -> None:
        """
        Compose a list of affine transform matrices (4x4), applied in order.
        i.e., output = Tn @ Tn-1 @ ... @ T1

        Args:
            transform_paths (list of str or Path): Paths to .txt files with 4x4 affine matrices.
            output_path (str or Path): Where to save the composed transform.
        Returns:
            None
        """
        # Ensure all paths are Path objects
        transform_paths = [Path(p).with_suffix(".txt") for p in transform_paths]

        # Load all matrices
        matrices = [np.loadtxt(p) for p in transform_paths]

        # Compose in order: Tn @ ... @ T1
        composed = matrices[0]
        for mat in matrices[1:]:
            composed = mat @ composed

        # Save
        np.savetxt(output_path, composed, fmt="%.12f")

    def _invert_affine_transform(
        self, transform_path: str | Path, output_path: str | Path
    ) -> None:
        """
        Invert a single affine transform matrix (4x4) and save it.

        Args:
            transform_path (str or Path): Path to the .txt file with the 4x4 affine matrix.
            output_path (str or Path): Where to save the inverted transform.
        Returns:
            None
        """
        # Load the matrix
        matrix = np.loadtxt(transform_path)

        # Invert the matrix
        inverted_matrix = np.linalg.inv(matrix)

        # Save the inverted matrix
        np.savetxt(output_path, inverted_matrix, fmt="%.12f")

    def transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str | Path | List[str | Path],
        log_file_path: str,
        interpolator: str = "0",
        **kwargs: dict,
    ) -> None:
        """
        Apply a transformation using NiftyReg.
        By default the padding value corresponds to the minimum of the moving image.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str | Path | List[str | Path]): Path(s) to the transformation matrix(es).
            log_file_path (str): Path to the log file.
            interpolator (str): Interpolation order (0, 1, 3, 4) (0=NN, 1=LIN; 3=CUB, 4=SINC). Default is '1' (linear).
        Raises:
            AssertionError: If the interpolator is not valid.
        """
        assert (
            interpolator in VALID_INTERPOLATORS
        ), f"Invalid interpolator: {interpolator}. Valid options are: {', '.join([f'{k} ({v})' for k, v in VALID_INTERPOLATORS.items()])}"

        runner = ScriptRunner(
            script_path=self.transformation_script,
            log_path=log_file_path,
        )

        niftyreg_executable = str(
            Path(__file__).parent.absolute() / "niftyreg_scripts" / "reg_resample"
        )

        transform_path = None
        is_tmpfile = False
        if isinstance(matrix_path, list):
            # If matrix_path is a list,we need to compute the combined transformation
            if len(matrix_path) == 1:
                # If only one matrix is provided, we can use it directly
                transform_path = Path(matrix_path[0]).with_suffix(".txt")
            else:
                is_tmpfile = True
                temp_File = tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".txt", delete=False
                )
                transform_path = Path(temp_File.name)
                self._compose_affine_transforms(
                    transform_paths=matrix_path,
                    output_path=transform_path,
                )
        else:
            transform_path = Path(matrix_path).with_suffix(".txt")

        # read moving image to get padding value
        padding_value = float(read_image(moving_image_path).min())

        input_params = [
            turbopath(niftyreg_executable),
            turbopath(fixed_image_path),
            turbopath(moving_image_path),
            turbopath(transformed_image_path),
            str(transform_path),
            str(interpolator),  # interpolation method, 3 is Cubic
            str(padding_value),
        ]

        # Call the run method to execute the script and capture the output in the log file
        success, error = runner.run(input_params)

        if is_tmpfile:
            transform_path.unlink(missing_ok=True)

        # if success:
        #     print("Script executed successfully. Check the log file for details.")
        # else:
        #     print("Script execution failed:", error)

    def inverse_transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str | Path | List[str | Path],
        log_file_path: str,
        interpolator: str = "0",
    ) -> None:
        """
        Apply inverse transformation using NiftyReg.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path  (str | Path | List[str | Path]): Path(s) to the transformation matrix(es) in inverse order.
            log_file_path (str): Path to the log file.
            interpolator (str): Interpolation order (0, 1, 3, 4) (0=NN, 1=LIN; 3=CUB, 4=SINC), Default is '1' (linear).
        """
        if not isinstance(matrix_path, list):
            matrix_path = [matrix_path]
        matrix_path = matrix_path[
            ::-1
        ]  # revert back to forward order to compute composite and then the inverse of it
        temp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)
        transform_path = Path(temp_file.name)

        self._compose_affine_transforms(
            transform_paths=matrix_path,
            output_path=transform_path,
        )
        self._invert_affine_transform(
            transform_path=transform_path,
            output_path=transform_path,
        )

        self.transform(
            fixed_image_path=fixed_image_path,
            moving_image_path=moving_image_path,
            transformed_image_path=transformed_image_path,
            matrix_path=transform_path,
            log_file_path=log_file_path,
            interpolator=interpolator,
        )

        transform_path.unlink(missing_ok=True)
