import unittest
from pathlib import Path

from brainles_preprocessing.constants import PreprocessorSteps
from brainles_preprocessing.modality import Modality
from brainles_preprocessing.registration import (
    ElastixRegistrator,
    NiftyRegRegistrator,
)


class TestModalityInterpolatorGuard(unittest.TestCase):

    def setUp(self):
        self.modality = Modality(
            modality_name="t1",
            input_path=Path("/tmp/fake_t1.nii.gz"),
            raw_skull_output_path=Path("/tmp/fake_output.nii.gz"),
        )
        self.fixed_image_path = Path("/tmp/fake_fixed.nii.gz")
        self.registration_dir = Path("/tmp/fake_registration_dir")

    def test_register_rejects_interpolator_for_non_ants(self):
        with self.assertRaises(ValueError):
            self.modality.register(
                registrator=NiftyRegRegistrator(),
                fixed_image_path=self.fixed_image_path,
                registration_dir=self.registration_dir,
                moving_image_name="t1",
                step=PreprocessorSteps.COREGISTERED,
                interpolator="linear",
            )

    def test_transform_rejects_interpolator_for_elastix(self):
        with self.assertRaises(ValueError):
            self.modality.transform(
                registrator=ElastixRegistrator(),
                fixed_image_path=self.fixed_image_path,
                registration_dir_path=self.registration_dir,
                moving_image_name="t1",
                transformation_matrix_path=self.registration_dir / "M_t1",
                step=PreprocessorSteps.COREGISTERED,
                interpolator="linear",
            )


if __name__ == "__main__":
    unittest.main()
