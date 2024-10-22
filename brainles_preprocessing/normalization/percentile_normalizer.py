import numpy as np
from .normalizer_base import Normalizer


class PercentileNormalizer(Normalizer):
    """
    Normalizer subclass for percentile-based image normalization.
    """

    def __init__(
        self,
        lower_percentile: float = 0.0,
        upper_percentile: float = 100.0,
        lower_limit: float = 0,
        upper_limit: float = 1,
    ):
        """
        Initialize the PercentileNormalizer.

        Parameters:
            lower_percentile (float): The lower percentile for mapping.
            upper_percentile (float): The upper percentile for mapping.
            lower_limit (float): The lower limit for normalized values.
            upper_limit (float): The upper limit for normalized values.
        """
        super().__init__()
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def normalize(self, image: np.ndarray):
        """
        Normalize the input image using percentile-based mapping.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The percentile-normalized image.
        """
        lower_value = np.percentile(image, self.lower_percentile)
        upper_value = np.percentile(image, self.upper_percentile)
        normalized_image = np.clip(
            (image - lower_value) / (upper_value - lower_value), 0, 1
        )
        normalized_image = (
            normalized_image * (self.upper_limit - self.lower_limit) + self.lower_limit
        )
        return normalized_image
