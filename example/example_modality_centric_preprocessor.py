# This script is an example of how to use the ModalityCentricPreprocessor class to preprocess a set of MR images. It is only here for quick development and testing purposes. It is not intended to be used in a production environment.
from auxiliary.turbopath import turbopath
from tqdm import tqdm

from brainles_preprocessing.brain_extraction import HDBetExtractor
from brainles_preprocessing.modality import CenterModality, Modality
from brainles_preprocessing.n4_bias_correction.sitk.sitk_n4_bias_corrector import (
    SitkN4BiasCorrector,
)
from brainles_preprocessing.normalization.percentile_normalizer import (
    PercentileNormalizer,
)
from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
from brainles_preprocessing.registration import ANTsRegistrator, NiftyRegRegistrator


def preprocess(inputDir):
    inputDir = turbopath(inputDir)
    print("*** start ***")

    # where are the raw mr files?
    brainles_dir = turbopath(inputDir) + "/" + inputDir.name + "_brainles"

    raw_bet_dir = brainles_dir / "raw_bet"
    norm_bet_dir = brainles_dir / "normalized_bet"
    raw_skull_dir = brainles_dir / "raw_skull"
    norm_skull_dir = brainles_dir / "normalized_skull"
    raw_deface_dir = brainles_dir / "raw_defaced"
    norm_deface_dir = brainles_dir / "normalized_defaced"

    t1_file = inputDir.files("*t1.nii.gz")
    t1c_file = inputDir.files("*t1c.nii.gz")
    t2_file = inputDir.files("*t2.nii.gz")
    flair_file = inputDir.files("*fla.nii.gz")
    if len(t1_file) == len(t1c_file) == len(t2_file) == len(flair_file) == 1:
        # print(t1_file)
        # print(t1c_file)
        # print(t2_file)
        # print(flair_file)
        t1File = t1_file[0]
        t1cFile = t1c_file[0]
        t2File = t2_file[0]
        flaFile = flair_file[0]
        # normalizer
        percentile_normalizer = PercentileNormalizer(
            lower_percentile=0.1,
            upper_percentile=99.9,
            lower_limit=0,
            upper_limit=1,
        )
        # define modalities
        center = CenterModality(
            modality_name="t1c",
            input_path=t1cFile,
            raw_bet_output_path=raw_bet_dir / f"{inputDir.name}_t1c_bet_raw.nii.gz",
            raw_skull_output_path=raw_skull_dir
            / f"{inputDir.name}_t1c_skull_raw.nii.gz",
            # normalized_bet_output_path=norm_bet_dir / inputDir.name
            # + "_t1c_bet_normalized.nii.gz",
            normalized_skull_output_path=norm_skull_dir
            / f"{inputDir.name}_t1c_skull_normalized.nii.gz",
            raw_defaced_output_path=raw_deface_dir
            / f"{inputDir.name}_t1c_defaced_raw.nii.gz",
            # normalized_defaced_output_path=norm_deface_dir
            # / "_t1c_defaced_normalized.nii.gz",
            atlas_correction=True,
            # n4_bias_correction=True,
            normalizer=percentile_normalizer,
        )
        moving_modalities = [
            Modality(
                modality_name="t1",
                input_path=t1File,
                raw_bet_output_path=raw_bet_dir / f"{inputDir.name}_t1_bet_raw.nii.gz",
                raw_skull_output_path=raw_skull_dir
                / f"{inputDir.name}_t1_skull_raw.nii.gz",
                # normalized_bet_output_path=norm_bet_dir / inputDir.name
                # + "_t1_bet_normalized.nii.gz",
                normalized_skull_output_path=norm_skull_dir
                / f"{inputDir.name}_t1_skull_normalized.nii.gz",
                raw_defaced_output_path=raw_deface_dir
                / f"{inputDir.name}_t1_defaced_raw.nii.gz",
                # normalized_defaced_output_path=norm_deface_dir / inputDir.name
                # + "_t1_defaced_normalized.nii.gz",
                atlas_correction=True,
                # n4_bias_correction=True,
                normalizer=percentile_normalizer,
            ),
            Modality(
                modality_name="t2",
                input_path=t2File,
                raw_bet_output_path=raw_bet_dir / f"{inputDir.name}_t2_bet_raw.nii.gz",
                raw_skull_output_path=raw_skull_dir
                / f"{inputDir.name}_t2_skull_raw.nii.gz",
                # normalized_bet_output_path=norm_bet_dir / inputDir.name
                # + "_t2_bet_normalized.nii.gz",
                normalized_skull_output_path=norm_skull_dir
                / f"{inputDir.name}_t2_skull_normalized.nii.gz",
                # raw_defaced_output_path=raw_deface_dir / inputDir.name
                # + "_t2_defaced_raw.nii.gz",
                # normalized_defaced_output_path=norm_deface_dir / inputDir.name
                # + "_t2_defaced_normalized.nii.gz",
                atlas_correction=True,
                normalizer=percentile_normalizer,
            ),
            Modality(
                modality_name="flair",
                input_path=flaFile,
                raw_bet_output_path=raw_bet_dir / f"{inputDir.name}_fla_bet_raw.nii.gz",
                raw_skull_output_path=raw_skull_dir
                / f"{inputDir.name}_fla_skull_raw.nii.gz",
                # normalized_bet_output_path=norm_bet_dir / inputDir.name
                # + "_fla_bet_normalized.nii.gz",
                normalized_skull_output_path=norm_skull_dir
                / f"{inputDir.name}_fla_skull_normalized.nii.gz",
                # raw_defaced_output_path=raw_deface_dir / inputDir.name
                # + "_fla_defaced_raw.nii.gz",
                # normalized_defaced_output_path=norm_deface_dir / inputDir.name
                # + "_fla_defaced_normalized.nii.gz",
                atlas_correction=True,
                normalizer=percentile_normalizer,
            ),
        ]

        preprocessor = AtlasCentricPreprocessor(
            center_modality=center,
            moving_modalities=moving_modalities,
            # choose the registration backend you want to use
            # registrator=NiftyRegRegistrator(),
            registrator=ANTsRegistrator(),
            # n4_bias_corrector=SitkN4BiasCorrector(),
            # registrator=NiftyRegRegistrator(),
            # brain_extractor=HDBetExtractor(),
            # temp_folder="temporary_directory",
            limit_cuda_visible_devices="0",
        )

        preprocessor.run(
            save_dir_coregistration=brainles_dir + "/co-registration",
            save_dir_atlas_registration=brainles_dir + "/atlas-registration",
            save_dir_atlas_correction=brainles_dir + "/atlas-correction",
            save_dir_n4_bias_correction=brainles_dir + "/n4-bias-correction",
            save_dir_brain_extraction=brainles_dir + "/brain-extraction",
            save_dir_defacing=brainles_dir + "/defacing",
            save_dir_transformations=brainles_dir + "/transformations",
        )


### *** GOGOGO *** ###
if __name__ == "__main__":
    EXAMPLE_DATA_DIR = turbopath("example/example_data")

    exams = EXAMPLE_DATA_DIR.dirs()

    for exam in tqdm(exams[1:]):
        print("processing:", exam)
        preprocess(exam)
        break  # remove this break to process all exams in the example_data directory
