from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional
import numpy as np
from auxiliary.io import read_image, write_image


class Defacer(ABC):
    def __init__(
        self,
        masking_value: Optional[Union[int, float]] = None,
    ):
        """
        Base class for defacing medical images using brain masks.

        Subclasses should implement the `deface` method to generate a defaced image
        based on the provided input image and mask.
        """
        # Here, masking value functions as a global value across all images and modalities
        # If no value is passed, the minimum of a given input image is chosen
        # TODO: Consider extending this to modality-specific masking values in the future, this should
        # probably be implemented as a property of the the specific modality
        self.masking_value = masking_value

    @abstractmethod
    def deface(
        self,
        input_image_path: Union[str, Path],
        mask_image_path: Union[str, Path],
    ) -> None:
        """
        Generate a defacing mask provided an input image.

        Args:
            input_image_path (str or Path): Path to the input image (NIfTI format).
            mask_image_path (str or Path): Path to save the output mask image (NIfTI format).
        """
        pass

    def apply_mask(
        self,
        input_image_path: Union[str, Path],
        mask_path: Union[str, Path],
        defaced_image_path: Union[str, Path],
    ) -> None:
        """
        Apply a brain mask to an input image.

        Args:
            input_image_path (str or Path): Path to the input image (NIfTI format).
            mask_path (str or Path): Path to the brain mask image (NIfTI format).
            defaced_image_path (str or Path): Path to save the resulting defaced image (NIfTI format).
        """

        input_image_path = Path(input_image_path)
        mask_path = Path(mask_path)

        if not input_image_path.is_file():
            raise FileNotFoundError(
                f"Input image file does not exist: {input_image_path}"
            )
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

        try:
            # Read data
            input_data = read_image(str(input_image_path))
            mask_data = read_image(str(mask_path))
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while reading input files: {e}"
            ) from e

        # Check that the input and mask have the same shape
        if input_data.shape != mask_data.shape:
            raise ValueError("Input image and mask must have the same dimensions.")

        # Check whether a global masking value was passed, otherwise choose minimum
        if self.masking_value is None:
            current_masking_value = np.min(input_data)
        else:
            current_masking_value = (
                np.array(self.masking_value).astype(input_data.dtype).item()
            )
        # Apply mask (element-wise either input or masking value)
        masked_data = np.where(
            mask_data.astype(bool), input_data, current_masking_value
        )

        # Save the defaced image
        write_image(
            input_array=masked_data,
            output_path=str(defaced_image_path),
            reference_path=str(input_image_path),
            create_parent_directory=True,
        )
