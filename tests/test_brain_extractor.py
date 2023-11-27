import os
import unittest

from brainles_preprocessing.brain_extraction import HDBetExtractor
from auxiliary import turbopath


class TestHDBetExtractor(unittest.TestCase):
    def setUp(self):
        inputDir = (
            turbopath(__file__).parent.parent + "example_data/OtherEXampleFromTCIA"
        )
        self.brain_extractor = HDBetExtractor()
        self.input_image = inputDir.files("*t1c.nii.gz")[0]
        self.output_image = ...
        self.mask_image = ...
        self.masked_image = ...

    def tearDown(self):
        # Clean up created files if they exist
        for file_path in [
            self.input_image,
            self.output_image,
            self.mask_image,
            self.masked_image,
        ]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_extract_creates_output_files(self):
        self.brain_extractor.extract(
            input_image=self.input_image, output_image=self.output_image, log_file=None
        )

        self.assertTrue(
            os.path.exists(self.output_image), "Output image file was not created."
        )
        self.assertTrue(
            os.path.exists(self.mask_image), "Mask image file was not created."
        )

    def test_apply_mask_creates_output_file(self):
        # self.brain_extractor.apply_mask(
        #     self.input_image, self.mask_image, self.output_image
        # )
        # self.assertTrue(
        #     os.path.exists(self.output_image_path),
        #     "Output image file was not created in apply_mask.",
        # )
