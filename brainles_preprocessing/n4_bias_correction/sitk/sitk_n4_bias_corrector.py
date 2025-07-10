from pathlib import Path
from typing import Callable, Optional, Union

import SimpleITK as sitk
from auxiliary.io import write_image

from brainles_preprocessing.n4_bias_correction.n4_bias_corrector import N4BiasCorrector


class SitkN4BiasCorrector(N4BiasCorrector):

    def __init__(
        self,
        mask_func: Optional[Callable[[sitk.Image], sitk.Image]] = None,
        n_max_iterations: Optional[int] = None,
        n_fitting_levels: int = 3,
    ) -> None:
        """
        N4 Bias Corrector using SimpleITK.

        Args:
            mask_func (Optional[Callable[[sitk.Image], sitk.Image]], optional):
                Function that generates a mask from an image.
                Defaults to: `lambda img_itk: sitk.OtsuThreshold(img_itk, 0, 1, 200)`.
            n_max_iterations (Optional[int], optional):
                Maximum number of iterations for bias field correction.
            n_fitting_levels (int, optional):
                Number of fitting levels. Default is 3.
        """

        if mask_func is None:
            mask_func = lambda img_itk: sitk.OtsuThreshold(img_itk, 0, 1, 200)
        self.mask_func = mask_func
        self.n_max_iterations = n_max_iterations
        self.n_fitting_levels = n_fitting_levels

    def compute_mask(self, img_itk: sitk.Image) -> sitk.Image:
        """
        Compute the mask for the input image using the provided mask function.

        Args:
            img_itk (SimpleITK.Image): The input image in SimpleITK format.

        Returns:
            SimpleITK.Image: The computed mask.
        """

        return self.mask_func(img_itk)

    def correct(
        self,
        input_img_path: Union[str, Path],
        output_img_path: Union[str, Path],
    ) -> None:
        """
        Correct the bias field of the input image using SimpleITK.

        Args:
            input_img_path (Union[str, Path]): Path to the input image.
            output_img_path (Union[str, Path]): Path where the corrected image will be saved.

        Returns:
            None
        """
        img_itk = sitk.ReadImage(str(input_img_path))

        mask_itk = self.compute_mask(img_itk)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        if self.n_max_iterations is not None:
            corrector.SetMaximumNumberOfIterations(
                [self.n_max_iterations] * self.n_fitting_levels
            )

        corrected_img = corrector.Execute(img_itk, mask_itk)
        corrected_img = sitk.GetArrayFromImage(corrected_img)

        write_image(
            input_array=corrected_img,
            output_path=str(output_img_path),
            reference_path=str(input_img_path),
        )
