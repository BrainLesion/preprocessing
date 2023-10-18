import nibabel as nib
import numpy as np

from .bashhdbet import bash_hdbet_caller
from .hdbet import hdbet_caller


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
    elif backend == "bashHD-BET":
        bash_hdbet_caller(
            input_image=input_image,
            masked_image=masked_image,
            log_file=log_file,
            mode=mode,
        )

    else:
        raise NotImplementedError("no other brain extration backend implemented yet")


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
