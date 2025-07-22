from pathlib import Path
from typing import Optional, Union

from loguru import logger

from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.registration.registrator import Registrator


class Transform:
    """
    Class to apply precomputed transformations based on the registration process.
    Common use case is to apply inverse transformations to transform e.g. segmentations in atlas space back to native space or transform existing labels in native space to atlas space.
    """

    def __init__(
        self,
        transformations_dir: Union[str, Path],
        registrator: Optional[Registrator] = None,
    ) -> None:
        """
        Initialize the Transform class.

        Args:
            transformations_dir (Union[str, Path]): Directory containing precomputed transformations for each modality.
            registrator (Optional[Registrator], optional): Registrator instance to use for applying transformations. If None, defaults to ANTsRegistrator.
        """

        self.transformations_dir = Path(transformations_dir)

        if registrator is None:
            logger.warning(
                "No registrator provided, using default ANTsRegistrator for registration."
            )
        self.registrator: Registrator = registrator or ANTsRegistrator()

    def apply(
        self,
        target_modality_name: str,
        target_modality_img: Union[str, Path],
        moving_image: Union[str, Path],
        output_img_path: Union[str, Path],
        log_file_path: Union[str, Path],
        interpolator: Optional[str] = None,
        inverse: bool = False,
    ) -> None:
        """
        Apply forward/ inverse transformation to a moving image to align it with a target modality.

        Args:
            target_modality_name (str): Name of the target modality. Must match the name used to create the transformations.
            target_modality_img (Union[str, Path]): Path to the target modality image.
            moving_image (Union[str, Path]): Path to the moving image. E.g., this could be a segmentation in atlas space.
            output_img_path (Union[str, Path]): Path where the transformed image will be saved.
            log_file_path (Union[str, Path]): Path to the log file where transformation details will be written.
            interpolator (Optional[str]): Interpolation method used during transformation.
            inverse (bool): If True, applies the inverse transformation. Default is False.
                Available options depend on the chosen registrator:

                - **ANTsRegistrator**:
                    - "linear"
                    - "nearestNeighbor" (default)
                    - "multiLabel" (deprecated, prefer "genericLabel")
                    - "gaussian"
                    - "bSpline"
                    - "cosineWindowedSinc"
                    - "welchWindowedSinc"
                    - "hammingWindowedSinc"
                    - "lanczosWindowedSinc"
                    - "genericLabel" (recommended for label images)

                - **NiftyReg**:
                    - "0": nearest neighbor (default)
                    - "1": linear
                    - "3": cubic spline
                    - "4": sinc

        Raises:
            AssertionError: If the transformations directory for the given modality does not exist.
        """
        logger.info(
            f"Applying {'inverse' if inverse else ''} transformation for {target_modality_name} using {self.registrator.__class__.__name__}."
        )

        # assert modality name eixsts in transformations_dir
        modality_transformations_dir = (
            self.transformations_dir / f"{target_modality_name}"
        )

        assert (
            modality_transformations_dir.exists()
        ), f"Transformations directory for {target_modality_name} does not exist: {modality_transformations_dir}"

        transforms = list(modality_transformations_dir.iterdir())
        transforms.sort()  # sort by name to get order for forward transform

        if inverse:
            transforms = transforms[::-1]  # inverse order for inverse transform

        kwargs = {
            "fixed_image_path": target_modality_img,
            "moving_image_path": moving_image,
            "transformed_image_path": output_img_path,
            "matrix_path": transforms,
            "log_file_path": str(log_file_path),
        }
        if interpolator is not None:
            kwargs["interpolator"] = interpolator

        if inverse:
            self.registrator.inverse_transform(**kwargs)
        else:
            self.registrator.transform(**kwargs)
