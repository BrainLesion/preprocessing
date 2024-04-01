import os
import shutil
import unittest
from abc import abstractmethod

from auxiliary.turbopath import turbopath

from brainles_preprocessing.registration.ANTs.ANTs import ANTsRegistrator
from brainles_preprocessing.registration.eReg.eReg import eRegRegistrator
from brainles_preprocessing.registration.niftyreg.niftyreg import NiftyRegRegistrator


class RegistratorBase:
    @abstractmethod
    def get_registrator(self):
        pass

    @abstractmethod
    def get_method_and_extension(self):
        pass

    def setUp(self):
        self.registrator = self.get_registrator()
        self.method_name, self.matrix_extension = self.get_method_and_extension()

        test_data_dir = turbopath(__file__).parent + "/test_data"
        input_dir = test_data_dir + "/input"
        self.output_dir = test_data_dir + f"/temp_output_{self.method_name}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.fixed_image = input_dir + "/tcia_example_t1c.nii.gz"
        self.moving_image = input_dir + "/bet_tcia_example_t1c_mask.nii.gz"

        self.matrix = self.output_dir + "/matrix"
        self.transform_matrix = input_dir + f"/matrix.{self.matrix_extension}"

    def tearDown(self):
        # Clean up created files if they exist
        shutil.rmtree(self.output_dir)

    def test_register_creates_output_files(self):
        transformed_image = self.output_dir + "/registered_image.nii.gz"
        log_file = self.output_dir + "/registration.log"

        self.registrator.register(
            fixed_image_path=self.fixed_image,
            moving_image_path=self.moving_image,
            transformed_image_path=transformed_image,
            matrix_path=self.matrix,
            log_file_path=log_file,
        )

        self.assertTrue(
            os.path.exists(transformed_image),
            "transformed file was not created.",
        )

        self.assertTrue(
            os.path.exists(f"{self.matrix}.{self.matrix_extension}"),
            "matrix file was not created.",
        )

        self.assertTrue(
            os.path.exists(log_file),
            "log file was not created.",
        )

    def test_transform_creates_output_files(self):
        transformed_image = self.output_dir + "/transformed_image.nii.gz"
        log_file = self.output_dir + "/transformation.log"

        self.registrator.transform(
            fixed_image_path=self.fixed_image,
            moving_image_path=self.moving_image,
            transformed_image_path=transformed_image,
            matrix_path=self.transform_matrix,
            log_file_path=log_file,
        )

        self.assertTrue(
            os.path.exists(transformed_image),
            "transformed file was not created.",
        )

        self.assertTrue(
            os.path.exists(log_file),
            "log file was not created.",
        )


# TODO also test transform


class TestANTsRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return ANTsRegistrator()

    def get_method_and_extension(self):
        return "ants", "mat"


class TestNiftyRegRegistratorRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return NiftyRegRegistrator()

    def get_method_and_extension(self):
        return "niftyreg", "txt"


class TestEregRegistratorRegistrator(RegistratorBase, unittest.TestCase):
    def get_registrator(self):
        return eRegRegistrator()

    def get_method_and_extension(self):
        return "ereg", "mat"
