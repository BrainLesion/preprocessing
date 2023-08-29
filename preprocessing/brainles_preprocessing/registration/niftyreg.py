import os
import shlex
import datetime
from ttictoc import Timer

import subprocess
import os

from brainles_preprocessing.registration.reg import Registrator

from auxiliary.runscript import ScriptRunner

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

        input_params = [fixed_image, moving_image, transformed_image, matrix]
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

        input_params = [fixed_image, moving_image, transformed_image, matrix]
        # Call the run method to execute the script and capture the output in the log file
        success, error = runner.run(input_params)


def run_bash_script_in_subprocess_and_log():
    pass


# TODO consider removing this legacy function
def niftyreg_caller(
    fixed_image,
    moving_image,
    transformed_image,
    matrix,
    log_file,
    mode,
):
    """calls niftyreg for registration and transforms"""

    the_shell = "/bin/bash"
    registration_abspath = os.path.dirname(os.path.abspath(__file__))

    if mode == "registration":
        shell_script = os.path.join(
            registration_abspath, "niftyreg_scripts", "rigid_reg.sh"
        )
    elif mode == "transformation":
        shell_script = os.path.join(
            registration_abspath, "niftyreg_scripts", "transform.sh"
        )
    else:
        raise NotImplementedError("this mode is not implemented:", mode)

    # let's try to call it
    try:
        starttime = str(datetime.datetime.now())
        print("** starting: " + moving_image.name + " at: " + starttime)
        t = Timer()  # TicToc("name")
        t.start()
        # your code ...
        # first we create the output dir
        # os.makedirs(output_dir, exist_ok=True)

        # generate subprocess call
        readableCmd = (
            the_shell,
            shell_script,
            fixed_image,
            moving_image,
            transformed_image,
            matrix,
        )
        readableCmd = " ".join(readableCmd)
        print(readableCmd)
        command = shlex.split(readableCmd)
        print(command)

        # cwd = pathlib.Path(__file__).resolve().parent
        cwd = registration_abspath
        print("*** cwd:", cwd)

        with open(log_file, "w") as outfile:
            subprocess.run(command, stdout=outfile, stderr=outfile, cwd=cwd)

        endtime = str(datetime.datetime.now().time())

        elapsed = t.stop("call")
        print(elapsed)

        with open(log_file, "a") as file:
            file.write("\n" + "************************************************" + "\n")
            file.write("cwd: " + str(cwd) + "\n")
            file.write("CALL: " + readableCmd + "\n")
            file.write("************************************************" + "\n")
            file.write("************************************************" + "\n")
            file.write("start time: " + starttime + "\n")
            file.write("end time: " + endtime + "\n")
            file.write("time elapsed: " + str(int(elapsed) / 60) + " minutes" + "\n")
            file.write("************************************************" + "\n")

    except Exception as e:
        print("error: " + str(e))
        print("registration error for: " + moving_image.name)

    endtime = str(datetime.datetime.now())
    print("** finished: " + moving_image.name + " at: " + endtime)
