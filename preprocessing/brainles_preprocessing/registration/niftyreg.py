import pathlib
import shlex
import datetime
from ttictoc import Timer

import subprocess
import os

from preprocessing.brainles_preprocessing.registration import Registrator


class NiftyRegRegistrator(Registrator):
    def register(
        self,
        fixed_image,
        moving_image,
        transformed_image,
        matrix,
        log_file,
        mode,
    ):
        niftyreg_caller(
            fixed_image=fixed_image,
            moving_image=moving_image,
            transformed_image=transformed_image,
            matrix=matrix,
            log_file=log_file,
            mode=mode,
        )
        # TODO

    def transform(
        self,
        fixed_image,
        moving_image,
        transformed_image,
        matrix,
        log_file,
    ):
        # TODO
        pass


def run_bash_script_in_subprocess_and_log():
    pass


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
