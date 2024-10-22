import numpy as np
from .normalizer_base import Normalizer


class WindowingNormalizer(Normalizer):
    """
    Normalizer subclass for windowing-based image normalization.
    """

    def __init__(self, center, width):
        """
        Initialize the WindowingNormalizer.

        Parameters:
            center (float): The window center.
            width (float): The window width.
        """
        super().__init__()
        self.center = center
        self.width = width

    def normalize(self, image):
        """
        Normalize the input image using windowing.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The windowed normalized image.
        """
        min_value = self.center - self.width / 2
        max_value = self.center + self.width / 2
        windowed_image = np.clip(image, min_value, max_value)
        return windowed_image
