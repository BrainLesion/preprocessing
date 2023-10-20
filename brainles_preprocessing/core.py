from auxiliary.normalization.normalizer_base import Normalizer
from auxiliary.turbopath import turbopath
from brainles_preprocessing.registration.functional import (
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
        normalizer: Normalizer = None,
    ) -> None:
        self.modality_name = modality_name
        self.input_path = turbopath(input_path)
        self.output_path = turbopath(output_path)
        self.bet = bet
        self.normalizer = normalizer


# TODO citation reminder decorator here
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
    cm = center_modality
    all_modalities = [cm] + moving_modalities

    # CUDA devices
    if limit_cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = limit_cuda_visible_devices

    # create temporary storage
    storage = tempfile.TemporaryDirectory()
    temp_folder = turbopath(storage.name)
    print(temp_folder)

    # COREGISTRATION # TODO think about moving this to a sub-function - think about being back-end agnostic
    # coregister everything to center_modality
    # prepare directory
    coregistration_dir = temp_folder + "/coregistration"
    os.makedirs(coregistration_dir, exist_ok=True)

    coregistered_modalities = []  # TODO think about saving this to the mm instead
    for mm in moving_modalities:
        reg_name = "/co__" + cm.modality_name + "__" + mm.modality_name

        co_registered = coregistration_dir + reg_name + ".nii.gz"
        co_registered_log = coregistration_dir + reg_name + ".log"
        co_registered_matrix = coregistration_dir + reg_name + ".txt"

        register(
            fixed_image=cm.input_path,
            moving_image=mm.input_path,
            transformed_image=co_registered,
            matrix=co_registered_matrix,
            log_file=co_registered_log,
        )
        coregistered_modalities.append(co_registered)

    # also copy center_modality itself and export folder
    if keep_coregistration is not None:
        keep_coregistration = turbopath(keep_coregistration)
        native_cm = coregistration_dir + "/native__" + cm.modality_name + ".nii.gz"

        shutil.copyfile(cm.input_path, native_cm)
        # and copy the folder to keep it
        shutil.copytree(coregistration_dir, keep_coregistration, dirs_exist_ok=True)

    # TO ATLAS SPACE # TODO again this should probably be a function - think about being back-end agnostic
    # prepare directory
    atlas_dir = temp_folder + "/atlas-space"
    os.makedirs(atlas_dir, exist_ok=True)

    # register center_modality
    atlas_cm_matrix = atlas_dir + "/atlas__" + cm.modality_name + ".txt"

    atlas_cm_log = atlas_dir + "/atlas__" + cm.modality_name + ".log"

    atlas_cm = atlas_dir + "/atlas__" + cm.modality_name + ".nii.gz"

    atlas_image = turbopath(atlas_image)

    register(
        fixed_image=atlas_image,
        moving_image=cm.input_path,
        transformed_image=atlas_cm,
        matrix=atlas_cm_matrix,
        log_file=atlas_cm_log,
    )
    cm.current = atlas_cm

    # transform moving modalities
    for coreg, mm in zip(coregistered_modalities, moving_modalities):
        atlas_coreg = atlas_dir + "/atlas__" + mm.modality_name + ".nii.gz"
        atlas_coreg_log = atlas_dir + "/atlas__" + mm.modality_name + ".log"

        transform(
            fixed_image=atlas_image,
            moving_image=coreg,
            transformed_image=atlas_coreg,
            matrix=atlas_cm_matrix,
            log_file=atlas_coreg_log,
        )
        mm.current = atlas_coreg

    # copy folder to output
    if keep_atlas_registration is not None:
        keep_atlas_registration = turbopath(keep_atlas_registration)
        shutil.copytree(atlas_dir, keep_atlas_registration, dirs_exist_ok=True)

    # o p t i o n a l  s t e p s
    # BRAINEXTRACTION # TODO make this a function - think about being back-end agnostic
    if bet_mode is not None:
        # prepare
        bet_dir = temp_folder + "/brain-extraction"
        os.makedirs(bet_dir, exist_ok=True)

        # skullstrip cm and obtain mask
        bet_log = bet_dir + "/brain-extraction.log"
        atlas_bet_cm = bet_dir + "/atlas_bet_" + cm.modality_name + ".nii.gz"
        atlas_mask = (
            atlas_bet_cm[:-7] + "_mask.nii.gz"
        )  # TODO change this, the skullstripper should define here

        brain_extractor(
            input_image=atlas_cm,
            masked_image=atlas_bet_cm,
            log_file=bet_log,
            mode=bet_mode,
        )

    # masking
    os.makedirs(cm.output_path.parent, exist_ok=True)
    if not cm.bet:
        cm.current = atlas_cm
    elif cm.bet:
        cm.current = atlas_bet_cm

    # now mask the rest or copy the non masked images
    brain_masked_dir = bet_dir + "/brain_masked"
    os.makedirs(brain_masked_dir, exist_ok=True)

    for mm in moving_modalities:
        atlas_coreg = atlas_dir + "/atlas__" + mm.modality_name + ".nii.gz"

        if not mm.bet:
            mm.current = atlas_coreg
        elif mm.bet:
            mm.brain_masked = (
                brain_masked_dir + "/brain_masked__" + mm.modality_name + ".nii.gz"
            )
            apply_mask(
                input_image=atlas_coreg,
                mask_image=atlas_mask,
                output_image=mm.brain_masked,
            )
            mm.current = mm.brain_masked

    # copy files and folders to output
    if keep_brainextraction is not None:
        keep_brainextraction = turbopath(keep_brainextraction)
        shutil.copytree(bet_dir, keep_brainextraction, dirs_exist_ok=True)

        # TODO introduce channel-wise normalization

    # FINAL OUTPUTS
    for mod in all_modalities:
        os.makedirs(mod.output_path.parent, exist_ok=True)
        shutil.copyfile(
            mod.current,
            mod.output_path,
        )


if __name__ == "__main__":
    pass
