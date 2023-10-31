import nibabel as nib
import numpy as np

from .bashhdbet import bash_hdbet_caller
from .hdbet import hdbet_caller
from .hdbet_implementation import HDBetExtractor, BashHDBetExtractor

# Can be moved to core
def brain_extractor(input_image, masked_image, log_file, mode, backend="hdbet"):
    """Skull-strips images using the specified backend."""

    if backend == "hdbet":
        extractor = HDBetExtractor()
    elif backend == "bashhdbet":
        extractor = BashHDBetExtractor()
    else:
        raise NotImplementedError(f"Unsupported brain extraction backend: {backend}")

    extractor.extract(input_image, masked_image, log_file, mode)


# Is this the best place to put it?
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
