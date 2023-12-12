from abc import ABC, abstractmethod
from typing import Any


class Registrator(ABC):
    # TODO probably the init here should be removed?
    # def __init__(self, backend):
    #     self.backend = backend

    @abstractmethod
    def register(
        self,
        fixed_image: Any,
        moving_image: Any,
        transformed_image: Any,
        matrix: Any,
        log_file: str,
    ) -> None:
        """
        Abstract method for registering images.

        Args:
            fixed_image (Any): The fixed image for registration.
            moving_image (Any): The moving image to be registered.
            transformed_image (Any): The resulting transformed image after registration.
            matrix (Any): The transformation matrix applied during registration.
            log_file (str): The path to the log file for recording registration details.

        Returns:
            None
        """
        pass

    @abstractmethod
    def transform(
        self,
        fixed_image: Any,
        moving_image: Any,
        transformed_image: Any,
        matrix: Any,
        log_file: str,
    ) -> None:
        """
        Abstract method for transforming images.

        Args:
            fixed_image (Any): The fixed image to be transformed.
            moving_image (Any): The moving image to be transformed.
            transformed_image (Any): The resulting transformed image.
            matrix (Any): The transformation matrix applied during transformation.
            log_file (str): The path to the log file for recording transformation details.

        Returns:
            None
        """
        pass
