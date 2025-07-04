import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from brainles_preprocessing.back_to_native import BackToNativeSpace
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.registration.registrator import Registrator


class TestBackToNativeSpace(unittest.TestCase):

    def setUp(self):
        self.transform_dir = Path("/tmp/fake_transform_dir")
        self.target_modality_name = "flair"
        self.target_modality_img = Path("/tmp/fake_target.nii.gz")
        self.moving_image = Path("/tmp/fake_moving.nii.gz")
        self.output_img_path = Path("/tmp/fake_output.nii.gz")
        self.log_file_path = Path("/tmp/fake_log.log")

    def test_default_registrator_is_ants(self):
        back = BackToNativeSpace(transformations_dir=self.transform_dir)
        self.assertIsInstance(back.registrator, ANTsRegistrator)

    @patch("brainles_preprocessing.back_to_native.Path.exists", return_value=False)
    def test_missing_transformation_dir_raises_assertion(self, mock_exists):
        back = BackToNativeSpace(transformations_dir=self.transform_dir)
        with self.assertRaises(AssertionError):
            back.transform(
                target_modality_name=self.target_modality_name,
                target_modality_img=self.target_modality_img,
                moving_image=self.moving_image,
                output_img_path=self.output_img_path,
                log_file_path=self.log_file_path,
            )

    @patch("brainles_preprocessing.back_to_native.Path.iterdir")
    @patch("brainles_preprocessing.back_to_native.Path.exists", return_value=True)
    def test_inverse_transform_called_with_correct_args(
        self, mock_exists, mock_iterdir
    ):
        mock_registrator = MagicMock(spec=Registrator)
        back = BackToNativeSpace(
            transformations_dir=self.transform_dir,
            registrator=mock_registrator,
        )

        # Mock Transformation files
        fake_transforms = [
            Path("/tmp/fake_transform1.txt"),
            Path("/tmp/fake_transform2.txt"),
        ]
        mock_iterdir.return_value = fake_transforms

        back.transform(
            target_modality_name=self.target_modality_name,
            target_modality_img=self.target_modality_img,
            moving_image=self.moving_image,
            output_img_path=self.output_img_path,
            log_file_path=self.log_file_path,
            interpolator="linear",
        )

        # Check if inverse_transform was called with correct parameters
        expected_kwargs = {
            "fixed_image_path": self.target_modality_img,
            "moving_image_path": self.moving_image,
            "transformed_image_path": self.output_img_path,
            "matrix_path": list(reversed(fake_transforms)),  # Inverse order
            "log_file_path": str(self.log_file_path),
            "interpolator": "linear",
        }
        mock_registrator.inverse_transform.assert_called_once_with(**expected_kwargs)


if __name__ == "__main__":
    unittest.main()
