from typing import List, Optional, Union, Callable

import SimpleITK as sitk
from numpy.typing import NDArray


class N4BiasOptions:
    def __init__(
        self,
        mask_func: Callable,
        n_max_iterations: Optional[int] = None,
    ):
        """
        A class to hold options for N4 bias correction.

        Args
            mask_func (Callable): A function that takes `img_itk` as its first argument. E.g.: `lambda img_itk: sitk.OtsuThreshold(img_itk, 0, 1, 200)
            n_max_iterations (Optional[Union[int, List[int]]]): The maximum number of iterations.
            `
        """
        self.mask_func = mask_func
        self.n_max_iterations = n_max_iterations

    def compute(self, img_itk):
        return self.mask_func(img_itk)


def n4_bias_corrector(
    input_image: str,
    n4_bias_opts: N4BiasOptions,
) -> NDArray:
    """
    Correct the bias field of the input image.

    Args:
        input_image (str): Path to the input image.
        n4_bias_opts (N4BiasOptions): Function to compute the mask for N4 bias correction.
        n_max_iterations (Optional[Union[int, List[int]]]): The maximum number of iterations.

    Returns:
        itk.image: The output image with corrected bias field.
    """
    img_itk = sitk.ReadImage(input_image)

    mask_itk = n4_bias_opts.compute(img_itk)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    if n4_bias_opts.n_max_iterations is not None:
        corrector.SetMaximumNumberOfIterations(n4_bias_opts.n_max_iterations)

    corrected_image = corrector.Execute(img_itk, mask_itk)
    return sitk.GetArrayFromImage(corrected_image)
