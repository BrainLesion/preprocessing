from pathlib import Path
from typing import Union

import nibabel as nib
from auxiliary.io import read_image, write_image

from brainles_preprocessing.defacing.defacer import Defacer
from brainles_preprocessing.defacing.quickshear.nipy_quickshear import run_quickshear
from brainles_preprocessing.constants import Atlas


class QuickshearDefacer(Defacer):
    """
    Defacer using Quickshear algorithm.

    Quickshear uses a skull stripped version of an anatomical images as a reference to deface the unaltered anatomical image.

    Base publication:
        - PDF: https://www.researchgate.net/profile/J-Hale/publication/262319696_Quickshear_defacing_for_neuroimages/links/570b97ee08aed09e917516b1/Quickshear-defacing-for-neuroimages.pdf
        - Bibtex:
            ```
            @article{schimke2011quickshear,
                    title={Quickshear Defacing for Neuroimages.},
                    author={Schimke, Nakeisha and Hale, John},
                    journal={HealthSec},
                    volume={11},
                    pages={11},
                    year={2011}
                    }
            ```
    """

    def __init__(
        self,
        buffer: float = 10.0,
        force_atlas_registration: bool = True,
        atlas_image_path: Union[str, Path, Atlas] = Atlas.SRI24,
    ):
        """Initialize Quickshear defacer

        Args:
            buffer (float, optional): buffer parameter from quickshear algorithm. Defaults to 10.0.
            force_atlas_registration (bool, optional): If True, forces atlas registration of the BET mask before defacing to potentially boost quickshear performance. Defaults to True.
            atlas_image_path (Union[str, Path, Atlas], optional): Path to the atlas image or an Atlas enum value that will be used for the optional atlas registrations. Defaults to Atlas.SRI24.
        """
        super().__init__()
        self.buffer = buffer
        self.force_atlas_registration = force_atlas_registration
        self.atlas_image_path = atlas_image_path

    def deface(
        self,
        input_image_path: Union[str, Path],
        mask_image_path: Union[str, Path],
    ) -> None:
        """
        Generate a defacing mask using Quickshear algorithm.

        Note:
        The input image must be a brain-extracted (skull-stripped) image.

        Args:
        input_image_path (str or Path): Path to the brain-extracted input image.
        mask_image_path (str or Path): Path to save the generated mask image.
        """

        bet_img = nib.load(str(input_image_path))
        mask = run_quickshear(bet_img=bet_img, buffer=self.buffer)

        # transpose to match simpletik order
        mask = mask.transpose(2, 1, 0)
        write_image(
            input_array=mask,
            output_path=str(mask_image_path),
            reference_path=str(input_image_path),
        )
