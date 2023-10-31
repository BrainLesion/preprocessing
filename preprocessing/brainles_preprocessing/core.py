import os
import shutil
import tempfile
from auxiliary.normalization.normalizer_base import Normalizer
from auxiliary.turbopath import turbopath
from brainles_preprocessing.registration.functional import register, transform
from brainles_preprocessing.brain_extraction import brain_extractor, apply_mask

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
        self.current_path = turbopath(input_path)
        self.output_path = turbopath(output_path)
        self.bet = bet
        self.normalizer = normalizer


def set_cuda_devices(limit_cuda_visible_devices: str = None):
    if limit_cuda_visible_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = limit_cuda_visible_devices


def coregister_modalities(
    center_modality, moving_modalities, temp_folder, keep_coregistration
):
    """
    Coregister a list of moving modalities to a center modality.

    Args:
    - center_modality (Modality): The reference modality.
    - moving_modalities (list[Modality]): List of modalities to be coregistered.
    - temp_folder (str): Path to the temporary directory for intermediate files.

    Returns:
    - list: List of paths of coregistered modalities.
    """
    coregistration_dir = os.path.join(temp_folder, "coregistration")
    os.makedirs(coregistration_dir, exist_ok=True)

    coregistered_modalities = []
    for moving_modality in moving_modalities:
        reg_name = (
            f"/co__{center_modality.modality_name}__{moving_modality.modality_name}"
        )
        co_registered = os.path.join(coregistration_dir, f"{reg_name}.nii.gz")
        co_registered_matrix = os.path.join(coregistration_dir, f"{reg_name}.txt")
        co_registered_log = os.path.join(coregistration_dir, f"{reg_name}.log")

        register(
            fixed_image=center_modality.input_path,
            moving_image=center_modality.input_path,
            transformed_image=co_registered,
            matrix=co_registered_matrix,
            log_file=co_registered_log,
        )
        coregistered_modalities.append(co_registered)

    if keep_coregistration:
        keep_coregistration = turbopath(keep_coregistration)
        native_cm = os.path.join(
            coregistration_dir, f"native__{center_modality.modality_name}.nii.gz"
        )

        shutil.copyfile(center_modality.input_path, native_cm)
        shutil.copytree(coregistration_dir, keep_coregistration, dirs_exist_ok=True)

    return coregistered_modalities


def atlas_registration(
    center_modality,
    moving_modalities,
    atlas_image,
    coregistered_modalities,
    temp_folder,
    keep_atlas_registration,
):
    """
    Register modalities to an atlas space.

    Args:
    - center_modality (Modality): The reference modality.
    - moving_modalities (list[Modality]): List of modalities to be registered.
    - atlas_image (str): Path to the atlas image.
    - coregistered_modalities (list): List of paths of coregistered modalities.
    - temp_folder (str): Path to the temporary directory for intermediate files.
    """
    atlas_dir = os.path.join(temp_folder, "atlas-space")
    os.makedirs(atlas_dir, exist_ok=True)

    atlas_center_modality = os.path.join(
        atlas_dir, f"atlas__{center_modality.modality_name}.nii.gz"
    )
    atlas_center_modality_matrix = os.path.join(
        atlas_dir, f"atlas__{center_modality.modality_name}.txt"
    )
    atlas_center_modality_log = os.path.join(
        atlas_dir, f"atlas__{center_modality.modality_name}.log"
    )

    register(
        fixed_image=atlas_image,
        moving_image=center_modality.input_path,
        transformed_image=atlas_center_modality,
        matrix=atlas_center_modality_matrix,
        log_file=atlas_center_modality_log,
    )
    center_modality.current_path = atlas_center_modality

    for coreg, moving_modality in zip(coregistered_modalities, moving_modalities):
        atlas_coreg = os.path.join(
            atlas_dir, f"atlas__{moving_modality.modality_name}.nii.gz"
        )
        atlas_coreg_log = os.path.join(
            atlas_dir, f"atlas__{moving_modality.modality_name}.log"
        )

        transform(
            fixed_image=atlas_image,
            moving_image=coreg,
            transformed_image=atlas_coreg,
            matrix=atlas_center_modality_matrix,
            log_file=atlas_coreg_log,
        )
        moving_modality.current_path = atlas_coreg

    # copy folder to output
    if keep_atlas_registration:
        keep_atlas_registration = turbopath(keep_atlas_registration)
        shutil.copytree(atlas_dir, keep_atlas_registration, dirs_exist_ok=True)


def brain_extraction(temp_folder, center_modality, bet_mode):
    """
    Extract the brain region from the center modality.

    Args:
    - temp_folder (str): Path to the temporary directory for intermediate files.
    - center_modality (Modality): The modality to be processed.
    - bet_mode (str): Mode for brain extraction. E.g., 'gpu'.

    Returns:
    - tuple: Directory containing the brain-extracted image and the brain mask.
    """
    bet_dir = os.path.join(temp_folder, "/brain-extraction")
    os.makedirs(bet_dir, exist_ok=True)

    bet_log = os.path.join(bet_dir, "brain-extraction.log")
    atlas_bet_cm = os.path.join(
        bet_dir, f"atlas_bet_{center_modality.modality_name}.nii.gz"
    )
    atlas_mask = os.path.join(
        bet_dir, f"atlas_bet_{center_modality.modality_name}_mask.nii.gz"
    )

    brain_extractor(
        input_image=center_modality.current_path,
        masked_image=atlas_bet_cm,
        log_file=bet_log,
        mode=bet_mode,
    )
    # Is this check necessary?
    if center_modality.bet:
        center_modality.current_path = atlas_bet_cm
    return bet_dir, atlas_mask


def mask_moving_modalities(moving_modalities, atlas_mask, bet_dir):
    """
    Apply the mask to the moving modalities, or copy the non masked images.

    Args:
    - moving_modalities (list[Modality]): List of modalities to be masked.
    - atlas_mask (str): Path to the brain mask.
    - bet_dir (str): Directory containing brain-extraction results.
    """
    brain_masked_dir = os.path.join(bet_dir, "brain_masked")
    os.makedirs(brain_masked_dir, exist_ok=True)

    for moving_modality in moving_modalities:
        if moving_modality.bet:
            brain_masked = os.path.join(
                brain_masked_dir,
                f"brain_masked__{moving_modality.modality_name}.nii.gz",
            )
            apply_mask(
                input_image=moving_modality.current_path,
                mask_image=atlas_mask,
                output_image=brain_masked,
            )
            moving_modality.current_path = brain_masked

    # copy files and folders to output
    if keep_brainextraction is not None:
        keep_brainextraction = turbopath(keep_brainextraction)
        shutil.copytree(bet_dir, keep_brainextraction, dirs_exist_ok=True)


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
    """
    Process imaging modalities to bring them to atlas space with various preprocessing steps.

    Args:
    - center_modality (Modality): The reference modality.
    - moving_modalities (list[Modality]): List of modalities to be processed.
    - atlas_image (str): Path to the atlas image. Default is the T1 image in BRATS space.
    - bet_mode (str): Mode for brain extraction. Default is 'gpu'.
    - limit_cuda_visible_devices (str): CUDA device(s) to be made visible. Default is None.
    - keep_coregistration (str): Path to save coregistration results. Default is None.
    - keep_atlas_registration (str): Path to save atlas registration results. Default is None.
    - keep_brainextraction (str): Path to save brain extraction results. Default is None.
    """
    set_cuda_devices(limit_cuda_visible_devices)

    storage = tempfile.TemporaryDirectory()
    temp_folder = turbopath(storage.name)

    bet_dir = os.path.join(temp_folder, "brain-extraction")

    coregistered_modalities = coregister_modalities(
        center_modality, moving_modalities, temp_folder, keep_coregistration
    )

    atlas_registration(
        center_modality,
        moving_modalities,
        atlas_image,
        coregistered_modalities,
        temp_folder,
    )

    bet_dir, atlas_mask = brain_extraction(temp_folder, center_modality, bet_mode)

    mask_moving_modalities(moving_modalities, atlas_mask, bet_dir)

    for modality in [center_modality] + moving_modalities:
        os.makedirs(modality.output_path.parent, exist_ok=True)
        shutil.copyfile(
            modality.current_path,
            modality.output_path,
        )
    # Cleanup
    storage.cleanup()


if __name__ == "__main__":
    pass
