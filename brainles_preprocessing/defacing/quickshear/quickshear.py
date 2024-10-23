from pathlib import Path
from brainles_preprocessing.defacing.quickshear.nipy_quickshear import quickshear
from brainles_preprocessing.defacing.defacer import Defacer
import nibabel as nib
from numpy.typing import NDArray


class QuickshearDefacer(Defacer):

    def __init__(self, buffer: float = 10.0):
        """Initialize Quickshear defacer

        Args:
            buffer (float, optional): buffer parameter from quickshear algorithm. Defaults to 10.0.
        """
        super().__init__()
        self.buffer = buffer

    def deface(self, bet_img_path: Path) -> NDArray:
        """Deface image using Quickshear algorithm

        Args:
            bet_img_path (Path): Path to the brain extracted image

        Returns:
            NDArray: Defaced image mask
        """

        bet_img = nib.load(bet_img_path)
        return quickshear(bet_img=bet_img, buff=self.buffer)
