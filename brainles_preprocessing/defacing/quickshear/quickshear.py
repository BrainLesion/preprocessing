from pathlib import Path
from typing import Union

import nibabel as nib

from brainles_preprocessing.defacing.defacer import Defacer
from brainles_preprocessing.defacing.quickshear.nipy_quickshear import run_quickshear
from auxiliary.nifti.io import write_nifti


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

    def __init__(self, buffer: float = 10.0):
        """Initialize Quickshear defacer

        Args:
            buffer (float, optional): buffer parameter from quickshear algorithm. Defaults to 10.0.
        """
        super().__init__()
        self.buffer = buffer

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
        write_nifti(
            input_array=mask,
            output_nifti_path=str(mask_image_path),
            reference_nifti_path=str(input_image_path),
        )
