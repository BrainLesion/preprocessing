import os
from abc import abstractmethod

from auxiliary.turbopath import turbopath
import shutil


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
        self.moving_image = input_dir + "/tcia_example_t1.nii.gz"

        self.matrix = self.output_dir + f"/{self.method_name}_matrix"
        self.transform_matrix = input_dir + f"/{self.method_name}_matrix"

    def tearDown(self):
        # Clean up created files if they exist
        shutil.rmtree(self.output_dir)
        # pass

    def test_register_creates_output_files(self):
        transformed_image = (
            self.output_dir + f"/{self.method_name}_registered_image.nii.gz"
        )
        log_file = self.output_dir + f"/{self.method_name}_registration.log"

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
        transformed_image = (
            self.output_dir + f"/{self.method_name}_transformed_image.nii.gz"
        )
        log_file = self.output_dir + f"/{self.method_name}_transformation.log"

        print("tf m:", self.transform_matrix)
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
