import os
import shutil
import unittest

from auxiliary.turbopath import turbopath

from brainles_preprocessing.brain_extraction import HDBetExtractor


class TestHDBetExtractor(unittest.TestCase):
    def setUp(self):
        test_data_dir = turbopath(__file__).parent + "/test_data"
        input_dir = test_data_dir + "/input"
        self.output_dir = test_data_dir + "/temp_output_hdbet"
        os.makedirs(self.output_dir, exist_ok=True)

        self.brain_extractor = HDBetExtractor()

        self.input_image_path = input_dir + "/tcia_example_t1c.nii.gz"
        self.input_brain_mask_path = input_dir + "/bet_tcia_example_t1c_mask.nii.gz"

        self.masked_image_path = self.output_dir + "/bet_tcia_example_t1c.nii.gz"
        self.brain_mask_path = self.output_dir + "/bet_tcia_example_t1c_mask.nii.gz"
        self.masked_again_image_path = (
            self.output_dir + "/bet_tcia_example_t1c_masked2.nii.gz"
        )

        print(self.input_image_path)
        print(self.masked_image_path)

    def tearDown(self):
        # Clean up created files if they exist
        shutil.rmtree(self.output_dir)

    def test_extract_creates_output_files(self):
        # we try to run the fastest possible skullstripping on CPU
        self.brain_extractor.extract(
            input_image_path=self.input_image_path,
            masked_image_path=self.masked_image_path,
            brain_mask_path=self.brain_mask_path,
            mode="fast",
            device="cpu",
            do_tta=False,
            # TODO generate and also test for presence of log file
        )

        self.assertTrue(
            os.path.exists(self.masked_image_path), "Masked image file was not created."
        )
        self.assertTrue(
            os.path.exists(self.brain_mask_path),
            "Brain mask image file was not created.",
        )

    def test_apply_mask_creates_output_file(self):
        self.brain_extractor.apply_mask(
            input_image_path=self.input_image_path,
            mask_image_path=self.input_brain_mask_path,
            masked_image_path=self.masked_again_image_path,
        )
        self.assertTrue(
            os.path.exists(self.masked_again_image_path),
            "Output image file was not created in apply_mask.",
        )
