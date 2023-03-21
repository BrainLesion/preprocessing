from utils import turbopath, name_extractor
from lib import niftyreg_caller, skullstrip, apply_mask
import tempfile
import os
import shutil


def modality_centric_atlas_preprocessing(
    io_dict: dict,
    options_dict: dict,
):
    # get input data
    all_modalities = list(io_dict.keys())
    primary_modality = all_modalities[0]
    moving_modalities = all_modalities[1:]

    # create temporary storage
    storage = tempfile.TemporaryDirectory()
    temp_folder = turbopath(storage.name)
    print(temp_folder)

    # C O R E G I S T R A T I O N
    # coregister everything to center_modality
    # prepare directory
    coregistration_dir = temp_folder + "/coregistration"
    os.makedirs(coregistration_dir, exist_ok=True)

    primary_input_image = turbopath(io_dict[primary_modality]["input"])
    coregistered_modalities = []
    for mm in moving_modalities:
        reg_name = "/co__" + primary_modality + "__" + mm

        co_registered = coregistration_dir + reg_name + ".nii.gz"
        co_registered_log = coregistration_dir + reg_name + ".log"
        co_registered_matrix = coregistration_dir + reg_name + ".txt"

        moving_input_image = turbopath(io_dict[mm]["input"])

        niftyreg_caller(
            fixed_image=primary_input_image,
            moving_image=moving_input_image,
            transformed_image=co_registered,
            matrix=co_registered_matrix,
            log_file=co_registered_log,
            mode="registration",
        )
        coregistered_modalities.append(co_registered)

    # also copy center_modality itself and export folder
    keep_coregistration = options_dict.get("keep_coregistration", None)
    if keep_coregistration is not None:
        keep_coregistration = turbopath(keep_coregistration)
        native_cm = coregistration_dir + "/native__" + primary_modality + ".nii.gz"
        shutil.copyfile(primary_modality, native_cm)
        # and copy the folder to keep it
        shutil.copytree(coregistration_dir, keep_coregistration, dirs_exist_ok=True)

    # T o   a t l a s   s p a c e !
    # prepare directory
    atlas_dir = temp_folder + "/atlas-space"
    os.makedirs(atlas_dir, exist_ok=True)

    # register center_modality
    atlas_pm_matrix = atlas_dir + "/atlas__" + primary_modality + ".txt"

    atlas_pm_log = atlas_dir + "/atlas__" + primary_modality + ".log"

    atlas_pm = atlas_dir + "/atlas__" + primary_modality + ".nii.gz"

    atlas_image = turbopath(options_dict.get("atlas_image", "atlas/t1_brats_space.nii"))

    niftyreg_caller(
        fixed_image=atlas_image,
        moving_image=primary_input_image,
        transformed_image=atlas_pm,
        matrix=atlas_pm_matrix,
        log_file=atlas_pm_log,
        mode="registration",
    )

    # transform moving modalities
    for coreg, mm in zip(coregistered_modalities, moving_modalities):
        atlas_coreg = atlas_dir + "/atlas__" + mm + ".nii.gz"
        atlas_coreg_log = atlas_dir + "/atlas__" + mm + ".log"

        niftyreg_caller(
            fixed_image=atlas_image,
            moving_image=coreg,
            transformed_image=atlas_coreg,
            matrix=atlas_pm_matrix,
            log_file=atlas_coreg_log,
            mode="transformation",
        )

    # copy folder to output
    keep_atlas_registration = options_dict.get("keep_atlas_registration", None)
    if keep_atlas_registration is not None:
        keep_atlas_registration = turbopath(keep_atlas_registration)
        shutil.copytree(atlas_dir, keep_atlas_registration, dirs_exist_ok=True)

    # S K U L L S T R I P P I N G
    bet_mode = options_dict.get("bet_mode", None)
    if bet_mode is not None:
        # prepare
        bet_dir = temp_folder + "/brainextraction"
        os.makedirs(bet_dir, exist_ok=True)

        # get output data

        # skullstrip t1c and obtain mask
        hd_bet_log = bet_dir + "/hd-bet.log"
        atlas_bet_pm = bet_dir + "/atlas_bet_pm.nii.gz"
        atlas_mask = atlas_bet_pm[:-7] + "_mask.nii.gz"

        skullstrip(
            input_image=atlas_pm,
            masked_image=atlas_bet_pm,
            log_file=hd_bet_log,
            mode=bet_mode,
        )

        # copy files and folders to output
        keep_brainextraction = options_dict.get("keep_brainextraction", None)
        if keep_brainextraction is not None:
            keep_brainextraction = turbopath(keep_brainextraction)
            shutil.copytree(bet_dir, keep_brainextraction, dirs_exist_ok=True)

    # O U T P U T S
    pm_output = turbopath(io_dict[primary_modality]["output"])
    os.makedirs(pm_output.parent, exist_ok=True)
    if io_dict[primary_modality]["bet"] == False:
        shutil.copyfile(
            atlas_pm,
            pm_output,
        )
    else:
        shutil.copyfile(
            atlas_bet_pm,
            pm_output,
        )

    # now mask the rest or copy the non masked images
    for mm in moving_modalities:
        atlas_coreg = atlas_dir + "/atlas__" + mm + ".nii.gz"
        mm_output = io_dict[mm]["output"]
        os.makedirs(mm_output.parent, exist_ok=True)

        if io_dict[mm]["bet"] == False:
            shutil.copyfile(atlas_coreg, mm_output)
        else:
            apply_mask(
                input_image=atlas_coreg,
                mask_image=atlas_mask,
                output_image=mm_output,
            )


if __name__ == "__main__":
    test_path = turbopath(
        "/home/koflerf/niftyreg_preprocessor/module/atlas/t1_brats_space.nii"
    )
    print(name_extractor(test_path))

    io_dict = {
        "t1c": {
            "input": "input_path",
            "output": "output_path",
            "bet": True,
        },
        "t1": {
            "input": "input_path",
            "output": "output_path",
            "bet": True,
        },
        "t2": {
            "input": "input_path",
            "output": "output_path",
            "bet": True,
        },
        "flair": {
            "input": "input_path",
            "output": "output_path",
            "bet": True,
        },
    }

    options_dict = {
        "atlas_image": "atlas/t1_brats_space.nii",
        "bet_mode": "gpu",
        "keep_coregistration": None,
        "keep_atlas_registration": None,
        "keep_brainextraction": None,
    }

    modality_centric_atlas_preprocessing(
        io_dict=io_dict,
        options_dict=options_dict,
    )
