# Modified from:
# https://github.com/nipreps/synthstrip/blob/main/nipreps/synthstrip/cli.py
# Original copyright (c) 2024, NiPreps developers
# Licensed under the Apache License, Version 2.0
# Changes made by the BrainLesion Preprocessing team (2025)

from pathlib import Path
from typing import Optional, Union, cast

import nibabel as nib
import numpy as np
import scipy
import torch
from nibabel.nifti1 import Nifti1Image
from nipreps.synthstrip.model import StripModel
from nitransforms.linear import Affine

from brainles_preprocessing.brain_extraction.brain_extractor import BrainExtractor
from brainles_preprocessing.utils.zenodo import fetch_synthstrip


class SynthStripExtractor(BrainExtractor):

    def __init__(self, border: int = 1):
        """
        Brain extraction using SynthStrip with preprocessing conforming to model requirements.

        This is an optional dependency - to use this extractor, you need to install the `brainles_preprocessing` package with the `synthstrip` extra: `pip install brainles_preprocessing[synthstrip]`

        Adapted from https://github.com/nipreps/synthstrip

        Args:
            border (int): Mask border threshold in mm. Defaults to 1.
        """

        super().__init__()
        self.border = border

    def _setup_model(self, device: torch.device) -> StripModel:
        """
        Load SynthStrip model and prepare it for inference on the specified device.

        Args:
            device: Device to load the model onto.

        Returns:
            A configured and ready-to-use StripModel.
        """
        # necessary for speed gains (according to original nipreps authors)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        with torch.no_grad():
            model = StripModel()
            model.to(device)
            model.eval()

        # Load the model weights
        weights_folder = fetch_synthstrip()
        weights = weights_folder / "synthstrip.1.pt"
        checkpoint = torch.load(weights, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def _conform(self, input_nii: Nifti1Image) -> Nifti1Image:
        """
        Resample the input image to match SynthStrip's expected input space.

        Args:
            input_nii (Nifti1Image): Input NIfTI image to conform.

        Raises:
            ValueError: If the input NIfTI image does not have a valid affine.

        Returns:
            A new NIfTI image with conformed shape and affine.
        """

        shape = np.array(input_nii.shape[:3])
        affine = input_nii.affine

        if affine is None:
            raise ValueError("Input NIfTI image must have a valid affine.")

        # Get corner voxel centers in index coords
        corner_centers_ijk = (
            np.array(
                [
                    (i, j, k)
                    for k in (0, shape[2] - 1)
                    for j in (0, shape[1] - 1)
                    for i in (0, shape[0] - 1)
                ]
            )
            + 0.5
        )

        # Get corner voxel centers in mm
        corners_xyz = (
            affine
            @ np.hstack((corner_centers_ijk, np.ones((len(corner_centers_ijk), 1)))).T
        )

        # Target affine is 1mm voxels in LIA orientation
        target_affine = np.diag([-1.0, 1.0, -1.0, 1.0])[:, (0, 2, 1, 3)]

        # Target shape
        extent = corners_xyz.min(1)[:3], corners_xyz.max(1)[:3]
        target_shape = ((extent[1] - extent[0]) / 1.0 + 0.999).astype(int)

        # SynthStrip likes dimensions be multiple of 64 (192, 256, or 320)
        target_shape = np.clip(
            np.ceil(np.array(target_shape) / 64).astype(int) * 64, 192, 320
        )

        # Ensure shape ordering is LIA too
        target_shape[2], target_shape[1] = target_shape[1:3]

        # Coordinates of center voxel do not change
        input_c = affine @ np.hstack((0.5 * (shape - 1), 1.0))
        target_c = target_affine @ np.hstack((0.5 * (target_shape - 1), 1.0))

        # Rebase the origin of the new, plumb affine
        target_affine[:3, 3] -= target_c[:3] - input_c[:3]

        nii = Affine(
            reference=Nifti1Image(
                np.zeros(target_shape),
                target_affine,
                None,
            ),
        ).apply(input_nii)
        return cast(Nifti1Image, nii)

    def _resample_like(
        self,
        image: Nifti1Image,
        target: Nifti1Image,
        output_dtype: Optional[np.dtype] = None,
        cval: Union[int, float] = 0,
    ) -> Nifti1Image:
        """
        Resample the input image to match the target's grid using an identity transform.

        Args:
            image: The image to be resampled.
            target: The reference image.
            output_dtype: Output data type.
            cval: Value to use for constant padding.

        Returns:
            A resampled NIfTI image.
        """
        result = Affine(reference=target).apply(
            image,
            output_dtype=output_dtype,
            cval=cval,
        )
        return cast(Nifti1Image, result)

    def extract(
        self,
        input_image_path: Union[str, Path],
        masked_image_path: Union[str, Path],
        brain_mask_path: Union[str, Path],
        device: Union[torch.device, str] = "cuda",
        num_threads: int = 1,
        **kwargs,
    ) -> None:
        """
        Extract the brain from an input image using SynthStrip.

        Args:
            input_image_path (Union[str, Path]): Path to the input image.
            masked_image_path (Union[str, Path]): Path to the output masked image.
            brain_mask_path (Union[str, Path]): Path to the output brain mask.
            device (Union[torch.device, str], optional): Device to use for computation. Defaults to "cuda".
            num_threads (int, optional): Number of threads to use for computation in CPU mode. Defaults to 1.

        Returns:
            None: The function saves the masked image and brain mask to the specified paths.
        """

        device = torch.device(device) if isinstance(device, str) else device
        model = self._setup_model(device=device)

        if device.type == "cpu" and num_threads > 0:
            torch.set_num_threads(num_threads)

        # normalize intensities
        image = nib.load(input_image_path)
        image = cast(Nifti1Image, image)
        conformed = self._conform(image)
        in_data = conformed.get_fdata(dtype="float32")
        in_data -= in_data.min()
        in_data = np.clip(in_data / np.percentile(in_data, 99), 0, 1)
        in_data = in_data[np.newaxis, np.newaxis]

        # predict the surface distance transform
        input_tensor = torch.from_numpy(in_data).to(device)
        with torch.no_grad():
            sdt = model(input_tensor).cpu().numpy().squeeze()

        # unconform the sdt and extract mask
        sdt_target = self._resample_like(
            Nifti1Image(sdt, conformed.affine, None),
            image,
            output_dtype=np.dtype("int16"),
            cval=100,
        )
        sdt_data = np.asanyarray(sdt_target.dataobj).astype("int16")

        # find largest CC (just do this to be safe for now)
        components = scipy.ndimage.label(sdt_data.squeeze() < self.border)[0]
        bincount = np.bincount(components.flatten())[1:]
        mask = components == (np.argmax(bincount) + 1)
        mask = scipy.ndimage.morphology.binary_fill_holes(mask)

        # write the masked output
        img_data = image.get_fdata()
        bg = np.min([0, img_data.min()])
        img_data[mask == 0] = bg
        Nifti1Image(img_data, image.affine, image.header).to_filename(
            masked_image_path,
        )

        # write the brain mask
        hdr = image.header.copy()
        hdr.set_data_dtype("uint8")
        Nifti1Image(mask, image.affine, hdr).to_filename(brain_mask_path)
