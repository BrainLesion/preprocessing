# TODO add typing and docs
from abc import abstractmethod
from pathlib import Path
from typing import Union

from auxiliary.io import read_image, write_image


class BrainExtractor:
    @abstractmethod
    def extract(
        self,
        input_image_path: Union[str, Path],
        masked_image_path: Union[str, Path],
        brain_mask_path: Union[str, Path],
        **kwargs,
    ) -> None:
        """
        Abstract method to extract the brain from an input image.

        Args:
            input_image_path (str or Path): Path to the input image.
            masked_image_path (str or Path): Path where the brain-extracted image will be saved.
            brain_mask_path (str or Path): Path where the brain mask will be saved.
            mode (str or Mode): Extraction mode.
            **kwargs: Additional keyword arguments.
        """
        pass

    def apply_mask(
        self,
        input_image_path: Union[str, Path],
        mask_path: Union[str, Path],
        bet_image_path: Union[str, Path],
    ) -> None:
        """
        Apply a brain mask to an input image.

        Args:
            input_image_path (str or Path): Path to the input image (NIfTI format).
            mask_path (str or Path): Path to the brain mask image (NIfTI format).
            bet_image_path (str or Path): Path to save the resulting masked image (NIfTI format).
        """

        try:
            # Read data
            input_data = read_image(str(input_image_path))
            mask_data = read_image(str(mask_path))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {e.filename}") from e
        except Exception as e:
            raise RuntimeError(f"Error reading files: {e}") from e

        # Check that the input and mask have the same shape
        if input_data.shape != mask_data.shape:
            raise ValueError("Input image and mask must have the same dimensions.")

        # Mask and save it
        masked_data = input_data * mask_data

        try:
            write_image(
                input_array=masked_data,
                output_path=str(bet_image_path),
                reference_path=str(input_image_path),
                create_parent_directory=True,
            )
        except Exception as e:
            raise RuntimeError(f"Error writing output file: {e}") from e
