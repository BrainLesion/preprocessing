from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Registrator(ABC):
    # TODO probably the init here should be removed?
    # def __init__(self, backend):
    #     self.backend = backend

    # Maps the shared interpolator names ("linear", "nearestNeighbor") to the backend-specific value.
    # None means the backend does not support choosing an interpolator.
    INTERPOLATOR_MAP: Optional[Dict[str, str]] = None

    @classmethod
    def normalize_interpolator(cls, interpolator: str) -> str:
        """
        Map a shared interpolator name to the backend-specific value. Backend-specific values are passed through unchanged.

        Args:
            interpolator (str): Shared name ("linear", "nearestNeighbor") or backend-specific value.

        Returns:
            str: The backend-specific interpolator value.

        Raises:
            ValueError: If the backend does not support choosing an interpolator.
        """
        if cls.INTERPOLATOR_MAP is None:
            raise ValueError(
                f"{cls.__name__} does not support choosing an interpolator."
            )
        return cls.INTERPOLATOR_MAP.get(interpolator, interpolator)

    @abstractmethod
    def register(
        self,
        fixed_image_path: Any,
        moving_image_path: Any,
        transformed_image_path: Any,
        matrix_path: Any,
        log_file_path: str,
    ):
        """
        Abstract method for registering images.

        Args:
            fixed_image_path (Any): The fixed image for registration.
            moving_image_path (Any): The moving image to be registered.
            transformed_image_path (Any): The resulting transformed image after registration.
            matrix_path (Any): The transformation matrix applied during registration.
            log_file_path (str): The path to the log file for recording registration details.
        """
        pass

    @abstractmethod
    def transform(
        self,
        fixed_image_path: Any,
        moving_image_path: Any,
        transformed_image_path: Any,
        matrix_path: Any,
        log_file_path: str,
        interpolator: str = "default",
        **kwargs
    ):
        """
        Abstract method for transforming images.

        Args:
            fixed_image_path (Any): The fixed image to be transformed.
            moving_image_path (Any): The moving image to be transformed.
            transformed_image_path (Any): The resulting transformed image.
            matrix_path (Any): The transformation matrix applied during transformation.
            log_file_path (str): The path to the log file for recording transformation details.
            interpolator (str): The interpolator to be used during transformation.

        """
        pass

    @abstractmethod
    def inverse_transform(
        self,
        fixed_image_path: Any,
        moving_image_path: Any,
        transformed_image_path: Any,
        matrix_path: Any,
        log_file_path: str,
        interpolator: str = "default",
    ):
        """
        Abstract method for inverse transforming images.

        Args:
            fixed_image_path (Any): The fixed image to be transformed.
            moving_image_path (Any): The moving image to be transformed.
            transformed_image_path (Any): The resulting transformed image.
            matrix_path (Any): The transformation matrix applied during transformation.
            log_file_path (str): The path to the log file for recording transformation details.
            interpolator (str): The interpolator to be used during transformation.
        """
        pass
