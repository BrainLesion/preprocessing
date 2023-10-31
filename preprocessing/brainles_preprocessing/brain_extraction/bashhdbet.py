import pathlib
import shlex
import datetime
from ttictoc import Timer
import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO)


def bash_hdbet_caller(input_image, masked_image, log_file, mode):
    """Skull-strips images with HD-BET.

    Parameters:
    - input_image: Path to the input image
    - masked_image: Path to the output masked image
    - log_file: Path to the log file
    - mode: Mode of operation (gpu/cpu/cpu-fast)

    Returns:
    None
    """

    brain_extraction_abspath = os.path.dirname(os.path.abspath(__file__))

    # Shell script selection using a dictionary
    modes_to_scripts = {
        "gpu": "hd-bet_gpu.sh",
        "cpu": "hd-bet_cpu.sh",
        "cpu-fast": "hd-bet_cpu-fast.sh",
    }
    if mode not in modes_to_scripts:
        raise NotImplementedError(f"This mode is not implemented: {mode}")

    shell_script = os.path.join(
        brain_extraction_abspath, "hdbet_scripts", modes_to_scripts[mode]
    )

    # Using default shell
    the_shell = os.getenv("SHELL", "/bin/bash")

    # Forming the command
    readableCmd = (the_shell, shell_script, input_image, masked_image)
    readableCmd = " ".join(readableCmd)
    command = shlex.split(readableCmd)

    cwd = brain_extraction_abspath

    # Execute command
    try:
        logging.info(f"Starting skullstripping with {mode} for {input_image.name}")
        timer = Timer()
        timer.start()

        with open(log_file, "w") as outfile:
            subprocess.run(command, stdout=outfile, stderr=outfile, cwd=cwd)

        elapsed_time = timer.stop("call")
        _write_log(log_file, readableCmd, cwd, elapsed_time)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        logging.error(f"Skullstripping error for: {input_image.name}")

    logging.info(f"Finished: {input_image.name}")


def _write_log(log_file, command, cwd, elapsed):
    with open(log_file, "a") as file:
        file.write("\n" + "*" * 50 + "\n")
        file.write(f"CALL: {command}\n")
        file.write(f"cwd: {str(cwd)}\n")
        file.write(f"Start time: {str(datetime.datetime.now())}\n")
        file.write(f"End time: {str(datetime.datetime.now().time())}\n")
        file.write(f"Time elapsed: {str(int(elapsed) / 60)} minutes\n")
        file.write("*" * 50 + "\n")
