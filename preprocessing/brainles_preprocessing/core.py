from auxiliary import turbopath
from preprocessing.brainles_preprocessing.registration.functional import (
    register,
    transform,
)
from brainles_preprocessing.brain_extraction import brain_extractor, apply_mask

import tempfile
import os
import shutil

core_abspath = os.path.dirname(os.path.abspath(__file__))


class Modality:
    def __init__(
        self,
        modality_name: str,
        input_path: str,
        output_path: str,
        bet: bool,
    ) -> None:
        self.modality_name = modality_name
        self.input_path = turbopath(input_path)
        self.output_path = turbopath(output_path)
        self.bet = bet


def preprocess_modality_centric_to_atlas_space(
    center_modality: Modality,
    moving_modalities: list[Modality],
    atlas_image: str = os.path.join(
        core_abspath, "registration/atlas/t1_brats_space.nii"
    ),
    bet_mode: str = "gpu",
    limit_cuda_visible_devices: str = None,
    keep_coregistration: str = None,
    keep_atlas_registration: str = None,
    keep_brainextraction: str = None,
):
    # CUDA devices
    if limit_cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = limit_cuda_visible_devices

    # create temporary storage
    storage = tempfile.TemporaryDirectory()
    temp_folder = turbopath(storage.name)
    print(temp_folder)

    # C O R E G I S T R A T I O N
    # coregister everything to center_modality
    # prepare directory
    coregistration_dir = temp_folder + "/coregistration"
    os.makedirs(coregistration_dir, exist_ok=True)

    coregistered_modalities = []
    for mm in moving_modalities:
        reg_name = "/co__" + center_modality.modality_name + "__" + mm.modality_name

        co_registered = coregistration_dir + reg_name + ".nii.gz"
        co_registered_log = coregistration_dir + reg_name + ".log"
        co_registered_matrix = coregistration_dir + reg_name + ".txt"

        register(
            fixed_image=center_modality.input_path,
            moving_image=center_modality.input_path,
            transformed_image=co_registered,
            matrix=co_registered_matrix,
            log_file=co_registered_log,
        )
        coregistered_modalities.append(co_registered)

    # also copy center_modality itself and export folder
    if keep_coregistration is not None:
        keep_coregistration = turbopath(keep_coregistration)
        native_cm = (
            coregistration_dir + "/native__" + center_modality.modality_name + ".nii.gz"
        )

        shutil.copyfile(center_modality.input_path, native_cm)
        # and copy the folder to keep it
        shutil.copytree(coregistration_dir, keep_coregistration, dirs_exist_ok=True)

    # T o   a t l a s   s p a c e !
    # prepare directory
    atlas_dir = temp_folder + "/atlas-space"
    os.makedirs(atlas_dir, exist_ok=True)

    # register center_modality
    atlas_pm_matrix = atlas_dir + "/atlas__" + center_modality.modality_name + ".txt"

    atlas_pm_log = atlas_dir + "/atlas__" + center_modality.modality_name + ".log"

    atlas_pm = atlas_dir + "/atlas__" + center_modality.modality_name + ".nii.gz"

    atlas_image = turbopath(atlas_image)

    register(
        fixed_image=atlas_image,
        moving_image=center_modality.input_path,
        transformed_image=atlas_pm,
        matrix=atlas_pm_matrix,
        log_file=atlas_pm_log,
    )

    # transform moving modalities
    for coreg, mm in zip(coregistered_modalities, moving_modalities):
        atlas_coreg = atlas_dir + "/atlas__" + mm.modality_name + ".nii.gz"
        atlas_coreg_log = atlas_dir + "/atlas__" + mm.modality_name + ".log"

        transform(
            fixed_image=atlas_image,
            moving_image=coreg,
            transformed_image=atlas_coreg,
            matrix=atlas_pm_matrix,
            log_file=atlas_coreg_log,
        )

    # copy folder to output
    if keep_atlas_registration is not None:
        keep_atlas_registration = turbopath(keep_atlas_registration)
        shutil.copytree(atlas_dir, keep_atlas_registration, dirs_exist_ok=True)

    # S K U L L S T R I P P I N G
    if bet_mode is not None:
        # prepare
        bet_dir = temp_folder + "/brainextraction"
        os.makedirs(bet_dir, exist_ok=True)

        # skullstrip t1c and obtain mask
        hd_bet_log = bet_dir + "/hd-bet.log"
        atlas_bet_pm = bet_dir + "/atlas_bet_pm.nii.gz"
        atlas_mask = atlas_bet_pm[:-7] + "_mask.nii.gz"

        brain_extractor(
            input_image=atlas_pm,
            masked_image=atlas_bet_pm,
            log_file=hd_bet_log,
            mode=bet_mode,
        )

        # copy files and folders to output
        if keep_brainextraction is not None:
            keep_brainextraction = turbopath(keep_brainextraction)
            shutil.copytree(bet_dir, keep_brainextraction, dirs_exist_ok=True)

    # O U T P U T S
    os.makedirs(center_modality.output_path.parent, exist_ok=True)
    if center_modality.bet == False:
        shutil.copyfile(
            atlas_pm,
            center_modality.output_path,
        )
    else:
        shutil.copyfile(
            atlas_bet_pm,
            center_modality.output_path,
        )

    # now mask the rest or copy the non masked images
    for mm in moving_modalities:
        atlas_coreg = atlas_dir + "/atlas__" + mm.modality_name + ".nii.gz"
        os.makedirs(mm.output_path.parent, exist_ok=True)

        if mm.bet == False:
            shutil.copyfile(atlas_coreg, mm.output_path)
        else:
            apply_mask(
                input_image=atlas_coreg,
                mask_image=atlas_mask,
                output_image=mm.output_path,
            )

        # TODO introduce channel-wise normalization


if __name__ == "__main__":
    pass
