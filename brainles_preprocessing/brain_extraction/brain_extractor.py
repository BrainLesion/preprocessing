# TODO add typing and docs
from abc import abstractmethod
from pathlib import Path
from shutil import copyfile

from auxiliary.nifti.io import read_nifti, write_nifti
from auxiliary.turbopath import name_extractor
from brainles_hd_bet import run_hd_bet


class BrainExtractor:
    @abstractmethod
    def extract(
        self,
        input_image_path: str,
        masked_image_path: str,
        brain_mask_path: str,
        log_file_path: str,
        # TODO convert mode to enum
        mode: str,
    ) -> None:
        pass

    def apply_mask(
        self,
        input_image_path: str,
        mask_image_path: str,
        masked_image_path: str,
    ) -> None:
        """
        Apply a brain mask to an input image.

        Parameters:
        - input_image_path (str): Path to the input image (NIfTI format).
        - mask_image_path (str): Path to the brain mask image (NIfTI format).
        - masked_image_path (str): Path to save the resulting masked image (NIfTI format).

        Returns:
        - str: Path to the saved masked image.
        """

        # read data
        input_data = read_nifti(input_image_path)
        mask_data = read_nifti(mask_image_path)

        # mask and save it
        masked_data = input_data * mask_data

        write_nifti(
            input_array=masked_data,
            output_nifti_path=masked_image_path,
            reference_nifti_path=input_image_path,
            create_parent_directory=True,
        )


class HDBetExtractor(BrainExtractor):
    def extract(
        self,
        input_image_path: str,
        masked_image_path: str,
        brain_mask_path: str,
        log_file_path: str = None,
        # TODO convert mode to enum
        mode: str = "accurate",
        device: int | str = 0,
        do_tta: bool = True,
    ) -> None:
        # GPU + accurate + TTA
        """skullstrips images with HD-BET generates a skullstripped file and mask"""
        run_hd_bet(
            mri_fnames=[input_image_path],
            output_fnames=[masked_image_path],
            # device=0,
            # TODO consider postprocessing
            # postprocess=False,
            mode=mode,
            device=device,
            postprocess=False,
            do_tta=do_tta,
            keep_mask=True,
            overwrite=True,
        )

        hdbet_mask_path = (
            Path(masked_image_path).parent
            / f"{name_extractor(masked_image_path)}_mask.nii.gz"
        )
        if hdbet_mask_path.resolve() != Path(brain_mask_path).resolve():
            copyfile(
                src=hdbet_mask_path,
                dst=brain_mask_path,
            )
