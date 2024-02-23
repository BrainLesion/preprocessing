# TODO add typing and docs
import os
import shutil

import ants
from auxiliary.nifti.io import read_nifti, write_nifti
from auxiliary.runscript import ScriptRunner
from auxiliary.turbopath import turbopath

from brainles_preprocessing.registration.registrator import Registrator

# from auxiliary import ScriptRunner


class ANTsRegistrator(Registrator):
    def __init__(
        self,
        type_of_transform: str = "Rigid",
        # TODO add registration parameters here
    ):
        """
        TODO
        """
        self.type_of_transform = type_of_transform

    def register(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: str,
    ) -> None:
        """
        Register images using NiftyReg.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix (output).
            log_file_path (str): Path to the log file.
        """
        transformed_image_path = turbopath(transformed_image_path)
        # we need to add .mat as this is the format for ANTs matrixes
        matrix_path = turbopath(matrix_path) + ".mat"

        fixed_image = ants.image_read(fixed_image_path)
        moving_image = ants.image_read(moving_image_path)

        # TODO finetune registration parameters
        registration_result = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform=self.type_of_transform,
        )
        transformed_image = registration_result["warpedmovout"]
        # make sure the parent exists
        os.makedirs(transformed_image_path.parent, exist_ok=True)
        ants.image_write(transformed_image, transformed_image_path)

        # write matrix
        os.makedirs(matrix_path.parent, exist_ok=True)
        shutil.copyfile(registration_result["fwdtransforms"][0], matrix_path)

        # TODO logging

    def transform(
        self,
        fixed_image_path: str,
        moving_image_path: str,
        transformed_image_path: str,
        matrix_path: str,
        log_file_path: str,
    ) -> None:
        """
        Apply a transformation using NiftyReg.

        Args:
            fixed_image_path (str): Path to the fixed image.
            moving_image_path (str): Path to the moving image.
            transformed_image_path (str): Path to the transformed image (output).
            matrix_path (str): Path to the transformation matrix.
            log_file_path (str): Path to the log file.
        """

        fixed_image = ants.image_read(fixed_image_path)
        moving_image = ants.image_read(moving_image_path)

        transformed_image_path = turbopath(transformed_image_path)
        os.makedirs(transformed_image_path.parent, exist_ok=True)

        matrix_path = turbopath(matrix_path) + ".mat"

        transformed_image = ants.apply_transforms(
            fixed=fixed_image, moving=moving_image, transformlist=[matrix_path]
        )

        ants.image_write(transformed_image, transformed_image_path)

        # TODO logging


if __name__ == "__main__":
    # TODO move this into unit tests
    reg = ANTsRegistrator()

    reg.register(
        fixed_image_path="example/example_data/TCGA-DU-7294/AX_T1_POST_GD_FLAIR_TCGA-DU-7294_TCGA-DU-7294_GE_TCGA-DU-7294_AX_T1_POST_GD_FLAIR_RM_13_t1c.nii.gz",
        moving_image_path="example/example_data/TCGA-DU-7294/AX_T2_FR-FSE_RF2_150_TCGA-DU-7294_TCGA-DU-7294_GE_TCGA-DU-7294_AX_T2_FR-FSE_RF2_150_RM_4_t2.nii.gz",
        transformed_image_path="example/example_ants/transformed_image.nii.gz",
        matrix_path="example/example_ants_matrix/matrix",
        log_file_path="example/example_ants/log.txt",
    )

    reg.transform(
        fixed_image_path="example/example_data/TCGA-DU-7294/AX_T1_POST_GD_FLAIR_TCGA-DU-7294_TCGA-DU-7294_GE_TCGA-DU-7294_AX_T1_POST_GD_FLAIR_RM_13_t1c.nii.gz",
        moving_image_path="example/example_data/OtherEXampleFromTCIA/T1_AX_OtherEXampleTCIA_TCGA-FG-6692_Si_TCGA-FG-6692_T1_AX_SE_10_se2d1_t1.nii.gz",
        transformed_image_path="example/example_ants_transformed/transformed_image.nii.gz",
        matrix_path="example/example_ants_matrix/matrix",
        log_file_path="example/example_ants/log.txt",
    )
