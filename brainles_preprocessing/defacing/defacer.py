from abc import abstractmethod
from pathlib import Path

from auxiliary.nifti.io import read_nifti, write_nifti


class Defacer:
    @abstractmethod
    def deface(
        self,
        input_image_path: Path,
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

        Args:
            input_image_path (str): Path to the input image (NIfTI format).
            mask_image_path (str): Path to the brain mask image (NIfTI format).
            masked_image_path (str): Path to save the resulting masked image (NIfTI format).
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
