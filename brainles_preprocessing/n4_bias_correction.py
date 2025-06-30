from typing import List, Optional, Union

import SimpleITK as sitk
from numpy.typing import NDArray


def n4_bias_corrector(
    input_image: str,
    input_mask: Optional[str] = None,
    n_fitting_levels: int = 4,
    n_max_iterations: Optional[Union[int, List[int]]] = None,
) -> NDArray:
    """
    Correct the bias field of the input image.

    Args:
        input_image (str): Path to the input image.
        input_mask (str): Path to the input mask.
        n_max_iterations (Optional[Union[int, List[int]]]): The maximum number of iterations. Will be multiplied by the number of fitting levels (default is 4).

    Returns:
        itk.image: The output image with corrected bias field.
    """
    img_itk = sitk.ReadImage(input_image)

    if input_mask is not None:
        mask_itk = sitk.ReadImage(input_mask)
    else:
        mask_itk = sitk.OtsuThreshold(img_itk, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    if n_max_iterations is not None:
        corrector.SetMaximumNumberOfIterations(n_max_iterations * n_fitting_levels)

    corrected_image = corrector.Execute(img_itk, mask_itk)
    return sitk.GetArrayFromImage(corrected_image)
