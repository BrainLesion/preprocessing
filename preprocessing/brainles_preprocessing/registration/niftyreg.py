import os
import shlex
import datetime
from ttictoc import Timer
import subprocess
import logging

from .reg import Registrator

logging.basicConfig(level=logging.INFO)


class NiftyRegRegistrator(Registrator):
    def __init__(self, registration_script=None, transformation_script=None):
        """
        Initialize the NiftyRegRegistrator.

        Args:
            registration_abspath (str): Absolute path to the registration directory.
            registration_script (str, optional): Path to the registration script. If None, a default script will be used.
            transformation_script (str, optional): Path to the transformation script. If None, a default script will be used.
        """
        super().__init__(backend="niftyreg")

        registration_abspath = os.path.dirname(os.path.abspath(__file__))

        # Set default registration script
        self.registration_script = registration_script or os.path.join(
            registration_abspath, "niftyreg_scripts", "rigid_reg.sh"
        )

        # Set default transformation script
        self.transformation_script = transformation_script or os.path.join(
            registration_abspath, "niftyreg_scripts", "transform.sh"
        )

    def register(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        self._run_script(
            self.registration_script,
            fixed_image,
            moving_image,
            transformed_image,
            matrix,
            log_file,
        )

    def transform(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        self._run_script(
            self.transformation_script,
            fixed_image,
            moving_image,
            transformed_image,
            matrix,
            log_file,
        )

    def _run_script(
        self,
        script_path,
        fixed_image,
        moving_image,
        transformed_image,
        matrix,
        log_file,
    ):
        the_shell = os.getenv("SHELL", "/bin/bash")
        command = shlex.split(
            f"{the_shell} {script_path} {fixed_image} {moving_image} {transformed_image} {matrix}"
        )

        try:
            logging.info(f"Running: {' '.join(command)}")
            with open(log_file, "w") as outfile:
                subprocess.run(
                    command,
                    stdout=outfile,
                    stderr=outfile,
                    cwd=os.path.dirname(script_path),
                )
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            logging.error(f"Error for: {moving_image}")
