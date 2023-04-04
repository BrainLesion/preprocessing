import pathlib
import shlex
import datetime
from ttictoc import Timer
import subprocess
import nibabel as nib
import numpy as np


def brain_extractor(
    input_image,
    masked_image,
    log_file,
    mode,
    backend="HD-BET",
):
    if backend == "HD-BET":
        hdbet_caller(
            input_image=input_image,
            masked_image=masked_image,
            log_file=log_file,
            mode=mode,
        )
    else:
        raise NotImplementedError("no other brain extration backend implemented yet")


def hdbet_caller(
    input_image,
    masked_image,
    log_file,
    mode,
):
    """skullstrips images with HD-BET generates a skullstripped file and mask"""
    the_shell = "/bin/bash"

    if mode == "gpu":
        shell_script = "skullstripping_scripts/hd-bet_gpu.sh"
    elif mode == "cpu":
        shell_script = "skullstripping_scripts/hd-bet_cpu.sh"
    elif mode == "cpu-fast":
        shell_script = "skullstripping_scripts/hd-bet_cpu-fast.sh"
    else:
        raise NotImplementedError("this mode is not implemented:", mode)
    # let's try to call it
    try:
        starttime = str(datetime.datetime.now())
        print(
            "** starting skullstripping with:",
            mode,
            "for:",
            input_image.name,
            "at:",
            starttime,
        )
        t = Timer()  # TicToc("name")
        t.start()

        # generate subprocess call
        readableCmd = (
            the_shell,
            shell_script,
            input_image,
            masked_image,
        )
        readableCmd = " ".join(readableCmd)
        print(readableCmd)
        command = shlex.split(readableCmd)
        print(command)

        cwd = pathlib.Path(__file__).resolve().parent
        print(cwd)

        with open(log_file, "w") as outfile:
            subprocess.run(command, stdout=outfile, stderr=outfile, cwd=cwd)

        endtime = str(datetime.datetime.now().time())

        elapsed = t.stop("call")
        print(elapsed)

        with open(log_file, "a") as file:
            file.write("\n" + "************************************************" + "\n")
            file.write("CALL: " + readableCmd + "\n")
            file.write("************************************************" + "\n")
            file.write("************************************************" + "\n")
            file.write("start time: " + starttime + "\n")
            file.write("end time: " + endtime + "\n")
            file.write("time elapsed: " + str(int(elapsed) / 60) + " minutes" + "\n")
            file.write("************************************************" + "\n")

    except Exception as e:
        print("error: " + str(e))
        print("skullstripping error for: " + input_image.name)

    endtime = str(datetime.datetime.now())
    print("** finished: " + input_image.name + " at: " + endtime)


def apply_mask(
    input_image,
    mask_image,
    output_image,
):
    """masks images with brain masks"""
    inputnifti = nib.load(input_image)
    mask = nib.load(mask_image)

    # mask it
    masked_file = np.multiply(inputnifti.get_fdata(), mask.get_fdata())
    masked_file = nib.Nifti1Image(masked_file, inputnifti.affine, inputnifti.header)

    # save it
    nib.save(masked_file, output_image)
