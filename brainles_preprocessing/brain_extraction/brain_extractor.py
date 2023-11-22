from abc import abstractmethod

import nibabel as nib
import numpy as np

from brainles_hd_bet import run_hd_bet


class BrainExtractor:
    @abstractmethod
    def extract(self, input_image, output_image, log_file, mode):
        pass

    def apply_mask(
        self,
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


class HDBetExtractor(BrainExtractor):
    def extract(self, input_image, output_image, log_file, mode):
        run_hd_bet(
            mri_fnames=[input_image],
            output_fnames=[output_image],
            mode="accurate",
            device=0,
            postprocess=False,
            do_tta=True,
            keep_mask=True,
            overwrite=True,
        )
