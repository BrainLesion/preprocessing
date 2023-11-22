# TODO add typing and docs
import os

from auxiliary.runscript import ScriptRunner
from auxiliary.turbopath import turbopath

from brainles_preprocessing.registration.registrator import Registrator

# from auxiliary import ScriptRunner


class NiftyRegRegistrator(Registrator):
    def __init__(
        self,
        registration_abspath=os.path.dirname(os.path.abspath(__file__)),
        registration_script=None,
        transformation_script=None,
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
        fixed_image,
        moving_image,
        transformed_image,
        matrix,
        log_file,
    ):
        """
        Register images using NiftyReg.

        Args:
            fixed_image (str): Path to the fixed image.
            moving_image (str): Path to the moving image.
            transformed_image (str): Path to the transformed image (output).
            matrix (str): Path to the transformation matrix (output).
            log_file (str): Path to the log file.
        """
        runner = ScriptRunner(
            script_path=self.registration_script,
            log_path=log_file,
        )

        niftyreg_executable = str(
            turbopath(__file__).parent + "/niftyreg_scripts/reg_aladin",
        )

        input_params = [
            turbopath(niftyreg_executable),
            turbopath(fixed_image),
            turbopath(moving_image),
            turbopath(transformed_image),
            turbopath(matrix),
        ]

        # Call the run method to execute the script and capture the output in the log file
        success, error = runner.run(input_params)

        # if success:
        #     print("Script executed successfully. Check the log file for details.")
        # else:
        #     print("Script execution failed:", error)

    def transform(
        self,
        fixed_image,
        moving_image,
        transformed_image,
        matrix,
        log_file,
    ):
        """
        Apply a transformation using NiftyReg.

        Args:
            fixed_image (str): Path to the fixed image.
            moving_image (str): Path to the moving image.
            transformed_image (str): Path to the transformed image (output).
            matrix (str): Path to the transformation matrix.
            log_file (str): Path to the log file.
        """
        runner = ScriptRunner(
            script_path=self.transformation_script,
            log_path=log_file,
        )

        niftyreg_executable = str(
            turbopath(__file__).parent + "/niftyreg_scripts/reg_resample",
        )

        input_params = [
            turbopath(niftyreg_executable),
            turbopath(fixed_image),
            turbopath(moving_image),
            turbopath(transformed_image),
            turbopath(matrix),
        ]

        # Call the run method to execute the script and capture the output in the log file
        success, error = runner.run(input_params)

        # if success:
        #     print("Script executed successfully. Check the log file for details.")
        # else:
        #     print("Script execution failed:", error)
