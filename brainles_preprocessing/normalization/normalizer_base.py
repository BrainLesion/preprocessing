from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Normalizer(ABC):
    """
    Abstract base class for image normalization methods.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def normalize(self, image) -> NDArray:
        """
        Normalize the input image based on the chosen method.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The normalized image.
        """
        pass
