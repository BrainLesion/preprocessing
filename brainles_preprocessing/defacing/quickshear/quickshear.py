from pathlib import Path

import nibabel as nib
from brainles_preprocessing.defacing.defacer import Defacer
from brainles_preprocessing.defacing.quickshear.nipy_quickshear import run_quickshear
from numpy.typing import NDArray
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

    def deface(self, mask_image_path: Path, bet_img_path: Path) -> None:
        """Deface image using Quickshear algorithm

        Args:
            bet_img_path (Path): Path to the brain extracted image
        """

        bet_img = nib.load(bet_img_path)
        mask = run_quickshear(bet_img=bet_img, buffer=self.buffer)
        write_nifti(
            input_array=mask,
            output_nifti_path=mask_image_path,
            reference_nifti_path=bet_img_path,
        )
