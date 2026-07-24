from pathlib import Path
from typing import Optional, Union
import shutil
from enum import Enum

from brainles_preprocessing.brain_extraction.brain_extractor import BrainExtractor
from brainles_hd_bet import run_hd_bet


class Mode(Enum):
    FAST = "fast"
    ACCURATE = "accurate"


class HDBetExtractor(BrainExtractor):
    def __init__(self, masking_value: Optional[Union[int, float]] = None):
        """
        Brain extraction HDBet implementation.

        Args:
            masking_value (Optional[Union[int, float]], optional): global value to be inserted in the masked areas. Default is None which leads to the minimum of each respective image.
        """
        super().__init__(masking_value=masking_value)

    def extract(
        self,
        input_image_path: Union[str, Path],
        masked_image_path: Union[str, Path],
        brain_mask_path: Union[str, Path],
        mode: Union[str, Mode] = Mode.ACCURATE,
        device: Optional[Union[int, str]] = 0,
        do_tta: bool = True,
        **kwargs,
    ) -> None:
        # GPU + accurate + TTA
        """
        Skull-strips images with HD-BET and generates a skull-stripped file and mask.

        Args:
            input_image_path (str or Path): Path to the input image.
            masked_image_path (str or Path): Path where the brain-extracted image will be saved.
            brain_mask_path (str or Path): Path where the brain mask will be saved.
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
