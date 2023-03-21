from path import Path
import pathlib
import tempfile
import os
import shlex
import datetime
from ttictoc import Timer
import subprocess
import shutil
import nibabel as nib
import numpy as np


def niftyreg_caller(
    fixed_image, moving_image, transformed_image, matrix, log_file, mode
):
    """calls niftyreg for registration and transforms"""

    the_shell = "/bin/bash"

    if mode == "registration":
        shell_script = "registration_scripts/rigid_reg.sh"
    elif mode == "transformation":
        shell_script = "registration_scripts/transform.sh"
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

        cwd = pathlib.Path(__file__).resolve().parent
        print(cwd)

        with open(log_file, "w") as outfile:
            subprocess.run(command, stdout=outfile, stderr=outfile, cwd=cwd)

        endtime = str(datetime.datetime.now().time())

        elapsed = t.stop("call")
        print(elapsed)

        with open(log_file, "a") as file:
            file.write("\n" + "************************************************" + "\n")
            file.write("niftyreg CALL: " + readableCmd + "\n")
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


def skullstrip(input_image, masked_image, log_file, mode):
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
            file.write("HD-BET CALL: " + readableCmd + "\n")
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


def apply_mask(input_image, mask_image, output_image):
    """masks images with brain masks"""
    inputnifti = nib.load(input_image)
    mask = nib.load(mask_image)

    # mask it
    masked_file = np.multiply(inputnifti.get_fdata(), mask.get_fdata())
    masked_file = nib.Nifti1Image(masked_file, inputnifti.affine, inputnifti.header)

    # save it
    nib.save(masked_file, output_image)


def brats_preprocessing(input_dict, output_dict, options_dict=None):
    """runs the BraTS inspired preprocessing"""

    if options_dict is None:
        options_dict = {}

    # get input data
    i_t1 = Path(os.path.abspath(input_dict["t1"]))
    i_t1c = Path(os.path.abspath(input_dict["t1c"]))
    i_t2 = Path(os.path.abspath(input_dict["t2"]))
    i_fla = Path(os.path.abspath(input_dict["fla"]))

    output_dir = Path(os.path.abspath(options_dict.get("output_dir", "output")))

    # create temporary storage
    storage = tempfile.TemporaryDirectory()
    temp_folder = Path(os.path.abspath(storage.name))
    print(temp_folder)

    # C O R E G I S T R A T I O N
    # coregister everything to t1
    # prepare directory
    coregistration_dir = temp_folder + "/coregistration"
    os.makedirs(coregistration_dir, exist_ok=True)

    # 1. t1c
    co_t1c = coregistration_dir + "/co-t1_t1c.nii.gz"
    co_t1c_log = coregistration_dir + "/co-t1_t1c.log"
    co_t1c_matrix = coregistration_dir + "/co-t1_t1c_matrix.txt"

    niftyreg_caller(
        fixed_image=i_t1,
        moving_image=i_t1c,
        transformed_image=co_t1c,
        matrix=co_t1c_matrix,
        log_file=co_t1c_log,
        mode="registration",
    )

    # 2. t2
    co_t2 = coregistration_dir + "/co-t1_t2.nii.gz"
    co_t2_log = coregistration_dir + "/co-t1_t2.log"
    co_t2_matrix = coregistration_dir + "/co-t1_t2_matrix.txt"

    niftyreg_caller(
        fixed_image=i_t1,
        moving_image=i_t2,
        transformed_image=co_t2,
        matrix=co_t2_matrix,
        log_file=co_t2_log,
        mode="registration",
    )

    # 3. fla
    co_fla = coregistration_dir + "/co-t1_fla.nii.gz"
    co_fla_log = coregistration_dir + "/co-t1_fla.log"
    co_fla_matrix = coregistration_dir + "/co-t1_fla_matrix.txt"

    niftyreg_caller(
        fixed_image=i_t1,
        moving_image=i_fla,
        transformed_image=co_fla,
        matrix=co_fla_matrix,
        log_file=co_fla_log,
        mode="registration",
    )

    # 4. last copy t1 itself and export folder
    keep_coregistration = options_dict.get("keep_coregistration", False)
    if keep_coregistration == True:
        native_t1 = coregistration_dir + "/native_t1.nii.gz"
        shutil.copyfile(i_t1, native_t1)
        # and copy the folder to keep it
        shutil.copytree(
            coregistration_dir, output_dir + "/coregistration", dirs_exist_ok=True
        )

    # T o   B r a T S   s p a c e !
    # prepare directory
    brats_dir = temp_folder + "/brats-space"
    os.makedirs(brats_dir, exist_ok=True)

    # register T1
    brats_atlas_image = (
        "atlas/t1_brats_space.nii"  # consider registering to the skullstripped atlas
        # "atlas/t1_skullstripped_brats_space.nii"
    )
    brats_t1_matrix = brats_dir + "/br_t1.txt"
    br_t1_log = brats_dir + "/br_t1.log"

    br_t1 = brats_dir + "/br_t1.nii.gz"

    niftyreg_caller(
        fixed_image=brats_atlas_image,
        moving_image=i_t1,
        transformed_image=br_t1,
        matrix=brats_t1_matrix,
        log_file=br_t1_log,
        mode="registration",
    )

    # transform T1c
    br_t1c = brats_dir + "/br_t1c.nii.gz"
    br_t1c_log = brats_dir + "/br_t1c.log"

    niftyreg_caller(
        fixed_image=brats_atlas_image,
        moving_image=co_t1c,
        transformed_image=br_t1c,
        matrix=brats_t1_matrix,
        log_file=br_t1c_log,
        mode="transformation",
    )

    # transform T2
    br_t2 = brats_dir + "/br_t2.nii.gz"
    br_t2_log = brats_dir + "/br_t2.log"

    niftyreg_caller(
        fixed_image=brats_atlas_image,
        moving_image=co_t2,
        transformed_image=br_t2,
        matrix=brats_t1_matrix,
        log_file=br_t2_log,
        mode="transformation",
    )

    # transform FLAIR
    br_fla = brats_dir + "/br_fla.nii.gz"
    br_fla_log = brats_dir + "/br_fla.log"

    niftyreg_caller(
        fixed_image=brats_atlas_image,
        moving_image=co_fla,
        transformed_image=br_fla,
        matrix=brats_t1_matrix,
        log_file=br_fla_log,
        mode="transformation",
    )

    # copy folder to output
    keep_brats_space = options_dict.get("keep_brats_space", False)
    if keep_brats_space == True:
        shutil.copytree(brats_dir, output_dir + "/brats-space", dirs_exist_ok=True)

    # S K U L L S T R I P P I N G
    # prepare
    ss_dir = temp_folder + "/skullstripping"
    os.makedirs(ss_dir, exist_ok=True)

    # get output data
    o_t1 = Path(os.path.abspath(output_dict["t1"]))
    os.makedirs(o_t1.parent, exist_ok=True)

    o_t1c = Path(os.path.abspath(output_dict["t1c"]))
    os.makedirs(o_t1c.parent, exist_ok=True)

    o_t2 = Path(os.path.abspath(output_dict["t2"]))
    os.makedirs(o_t2.parent, exist_ok=True)

    o_fla = Path(os.path.abspath(output_dict["fla"]))
    os.makedirs(o_fla.parent, exist_ok=True)

    bet_mode = options_dict.get("bet_mode", "gpu")
    # skullstrip t1 and obtain mask
    hd_bet_log = ss_dir + "/hd-bet.log"
    br_ss_t1 = ss_dir + "/br_ss_t1.nii.gz"
    brats_mask = br_ss_t1[:-7] + "_mask.nii.gz"

    skullstrip(
        input_image=br_t1,
        masked_image=br_ss_t1,
        log_file=hd_bet_log,
        mode=bet_mode,
    )

    # copy files and folders to output
    keep_skullstripping = options_dict.get("keep_skullstripping", False)
    if keep_skullstripping == True:
        shutil.copytree(ss_dir, output_dir + "/skullstripping", dirs_exist_ok=True)

    shutil.copyfile(br_ss_t1, o_t1)

    # mask t1c
    apply_mask(input_image=br_t1c, mask_image=brats_mask, output_image=o_t1c)
    # mask t2
    apply_mask(input_image=br_t2, mask_image=brats_mask, output_image=o_t2)
    # mask flair
    apply_mask(input_image=br_fla, mask_image=brats_mask, output_image=o_fla)


def t1c_focused_preprocessing(input_dict, output_dict, options_dict=None):
    """runs the BraTS inspired preprocessing"""

    if options_dict is None:
        options_dict = {}

    # get input data
    i_t1 = Path(os.path.abspath(input_dict["t1"]))
    i_t1c = Path(os.path.abspath(input_dict["t1c"]))
    i_t2 = Path(os.path.abspath(input_dict["t2"]))
    i_fla = Path(os.path.abspath(input_dict["fla"]))

    output_dir = Path(os.path.abspath(options_dict.get("output_dir", "output")))

    # create temporary storage
    storage = tempfile.TemporaryDirectory()
    temp_folder = Path(os.path.abspath(storage.name))
    print(temp_folder)

    # C O R E G I S T R A T I O N
    # coregister everything to t1
    # prepare directory
    coregistration_dir = temp_folder + "/coregistration"
    os.makedirs(coregistration_dir, exist_ok=True)

    # 1. t1c
    co_t1 = coregistration_dir + "/co-t1c_t1.nii.gz"
    co_t1_log = coregistration_dir + "/co-t1c_t1.log"
    co_t1c_matrix = coregistration_dir + "/co-t1c_t1_matrix.txt"

    niftyreg_caller(
        fixed_image=i_t1c,
        moving_image=i_t1,
        transformed_image=co_t1,
        matrix=co_t1c_matrix,
        log_file=co_t1_log,
        mode="registration",
    )

    # 2. t2
    co_t2 = coregistration_dir + "/co-t1c_t2.nii.gz"
    co_t2_log = coregistration_dir + "/co-t1c_t2.log"
    co_t2_matrix = coregistration_dir + "/co-t1c_t2_matrix.txt"

    niftyreg_caller(
        fixed_image=i_t1c,
        moving_image=i_t2,
        transformed_image=co_t2,
        matrix=co_t2_matrix,
        log_file=co_t2_log,
        mode="registration",
    )

    # 3. fla
    co_fla = coregistration_dir + "/co-t1c_fla.nii.gz"
    co_fla_log = coregistration_dir + "/co-t1c_fla.log"
    co_fla_matrix = coregistration_dir + "/co-t1c_fla_matrix.txt"

    niftyreg_caller(
        fixed_image=i_t1c,
        moving_image=i_fla,
        transformed_image=co_fla,
        matrix=co_fla_matrix,
        log_file=co_fla_log,
        mode="registration",
    )

    # 4. last copy t1c itself and export folder
    keep_coregistration = options_dict.get("keep_coregistration", False)
    if keep_coregistration == True:
        native_t1c = coregistration_dir + "/native_t1c.nii.gz"
        shutil.copyfile(i_t1c, native_t1c)
        # and copy the folder to keep it
        shutil.copytree(
            coregistration_dir, output_dir + "/coregistration", dirs_exist_ok=True
        )

    # T o   B r a T S   s p a c e !
    # prepare directory
    brats_dir = temp_folder + "/brats-space"
    os.makedirs(brats_dir, exist_ok=True)

    # register T1c
    brats_atlas_image = (
        "atlas/t1_brats_space.nii"  # consider registering to the skullstripped atlas
        # "atlas/t1_skullstripped_brats_space.nii"
    )
    brats_t1c_matrix = brats_dir + "/br_t1c.txt"
    br_t1c_log = brats_dir + "/br_t1c.log"

    br_t1c = brats_dir + "/br_t1c.nii.gz"

    niftyreg_caller(
        fixed_image=brats_atlas_image,
        moving_image=i_t1c,
        transformed_image=br_t1c,
        matrix=brats_t1c_matrix,
        log_file=br_t1c_log,
        mode="registration",
    )

    # transform T1
    br_t1 = brats_dir + "/br_t1.nii.gz"
    br_t1_log = brats_dir + "/br_t1.log"

    niftyreg_caller(
        fixed_image=brats_atlas_image,
        moving_image=co_t1,
        transformed_image=br_t1,
        matrix=brats_t1c_matrix,
        log_file=br_t1_log,
        mode="transformation",
    )

    # transform T2
    br_t2 = brats_dir + "/br_t2.nii.gz"
    br_t2_log = brats_dir + "/br_t2.log"

    niftyreg_caller(
        fixed_image=brats_atlas_image,
        moving_image=co_t2,
        transformed_image=br_t2,
        matrix=brats_t1c_matrix,
        log_file=br_t2_log,
        mode="transformation",
    )

    # transform FLAIR
    br_fla = brats_dir + "/br_fla.nii.gz"
    br_fla_log = brats_dir + "/br_fla.log"

    niftyreg_caller(
        fixed_image=brats_atlas_image,
        moving_image=co_fla,
        transformed_image=br_fla,
        matrix=brats_t1c_matrix,
        log_file=br_fla_log,
        mode="transformation",
    )

    # copy folder to output
    keep_brats_space = options_dict.get("keep_brats_space", False)
    if keep_brats_space == True:
        shutil.copytree(brats_dir, output_dir + "/brats-space", dirs_exist_ok=True)

    # S K U L L S T R I P P I N G
    # prepare
    ss_dir = temp_folder + "/skullstripping"
    os.makedirs(ss_dir, exist_ok=True)

    # get output data
    o_t1 = Path(os.path.abspath(output_dict["t1"]))
    os.makedirs(o_t1.parent, exist_ok=True)

    o_t1c = Path(os.path.abspath(output_dict["t1c"]))
    os.makedirs(o_t1c.parent, exist_ok=True)

    o_t2 = Path(os.path.abspath(output_dict["t2"]))
    os.makedirs(o_t2.parent, exist_ok=True)

    o_fla = Path(os.path.abspath(output_dict["fla"]))
    os.makedirs(o_fla.parent, exist_ok=True)

    bet_mode = options_dict.get("bet_mode", "gpu")
    # skullstrip t1c and obtain mask
    hd_bet_log = ss_dir + "/hd-bet.log"
    br_ss_t1c = ss_dir + "/br_ss_t1c.nii.gz"
    brats_mask = br_ss_t1c[:-7] + "_mask.nii.gz"

    skullstrip(
        input_image=br_t1c,
        masked_image=br_ss_t1c,
        log_file=hd_bet_log,
        mode=bet_mode,
    )

    # copy files and folders to output
    keep_skullstripping = options_dict.get("keep_skullstripping", False)
    if keep_skullstripping == True:
        shutil.copytree(ss_dir, output_dir + "/skullstripping", dirs_exist_ok=True)

    shutil.copyfile(br_ss_t1c, o_t1c)

    # mask t1
    apply_mask(input_image=br_t1, mask_image=brats_mask, output_image=o_t1)
    # mask t2
    apply_mask(input_image=br_t2, mask_image=brats_mask, output_image=o_t2)
    # mask flair
    apply_mask(input_image=br_fla, mask_image=brats_mask, output_image=o_fla)
