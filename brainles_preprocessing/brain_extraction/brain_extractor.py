# TODO add typing and docs
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
from enum import Enum

from auxiliary.io import read_image, write_image
from brainles_hd_bet import run_hd_bet


class Mode(Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class BrainExtractor:
    @abstractmethod
    def extract(
        self,
        input_image_path: Union[str, Path],
        masked_image_path: Union[str, Path],
        brain_mask_path: Union[str, Path],
        log_file_path: Optional[Union[str, Path]],
        mode: Union[str, Mode],
        **kwargs,
    ) -> None:
        """
        Abstract method to extract the brain from an input image.

        Args:
            input_image_path (str or Path): Path to the input image.
            masked_image_path (str or Path): Path where the brain-extracted image will be saved.
            brain_mask_path (str or Path): Path where the brain mask will be saved.
            log_file_path (str or Path, Optional): Path to the log file.
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


class HDBetExtractor(BrainExtractor):
    def extract(
        self,
        input_image_path: Union[str, Path],
        masked_image_path: Union[str, Path],
        brain_mask_path: Union[str, Path],
        log_file_path: Optional[Union[str, Path]] = None,
        # TODO convert mode to enum
        mode: Union[str, Mode] = Mode.ACCURATE,
        device: Optional[Union[int, str]] = 0,
        do_tta: Optional[bool] = True,
    ) -> None:
        # GPU + accurate + TTA
        """
        Skull-strips images with HD-BET and generates a skull-stripped file and mask.

        Args:
            input_image_path (str or Path): Path to the input image.
            masked_image_path (str or Path): Path where the brain-extracted image will be saved.
            brain_mask_path (str or Path): Path where the brain mask will be saved.
            log_file_path (str or Path, Optional): Path to the log file.
            mode (str or Mode): Extraction mode ('fast' or 'accurate').
            device (str or int): Device to use for computation (e.g., 0 for GPU 0, 'cpu' for CPU).
            do_tta (bool): whether to do test time data augmentation by mirroring along all axes.
        """

        # Ensure mode is a Mode enum instance
        if isinstance(mode, str):
            try:
                mode_enum = Mode(mode.lower())
            except ValueError:
                raise ValueError(f"'{mode}' is not a valid Mode.")
        elif isinstance(mode, Mode):
            mode_enum = mode
        else:
            raise TypeError("Mode must be a string or a Mode enum instance.")

        # Run HD-BET
        run_hd_bet(
            mri_fnames=[str(input_image_path)],
            output_fnames=[str(masked_image_path)],
            mode=mode_enum.value,
            device=device,
            # TODO consider postprocessing
            postprocess=False,
            do_tta=do_tta,
            keep_mask=True,
            overwrite=True,
        )

        # Construct the path to the generated mask
        masked_image_path = Path(masked_image_path)
        hdbet_mask_path = masked_image_path.with_name(
            masked_image_path.name.replace(".nii.gz", "_mask.nii.gz")
        )

        if hdbet_mask_path.resolve() != Path(brain_mask_path).resolve():
            try:
                shutil.copyfile(
                    src=str(hdbet_mask_path),
                    dst=str(brain_mask_path),
                )
            except Exception as e:
                raise RuntimeError(f"Error copying mask file: {e}") from e
