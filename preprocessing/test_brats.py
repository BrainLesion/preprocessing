from brats import brats_style_t1_centric_preprocessing
from utils import turbopath


from path import Path
import os
import datetime
from tqdm import tqdm


def preprocess(inputDir):
    inputDir = Path(os.path.abspath(inputDir))
    try:
        print("*** start ***")

        # where are the raw mr files?
        btk_raw_dir = Path(inputDir)

        # is the exam already processed?
        brainles_dir = Path(inputDir) + "/" + inputDir.name + "_brainles"
        prep_dir = brainles_dir + "/preprocessed"

        if not os.path.exists(prep_dir):
            t1_file = btk_raw_dir.files("*t1.nii.gz")
            t1c_file = btk_raw_dir.files("*t1c.nii.gz")
            t2_file = btk_raw_dir.files("*t2.nii.gz")
            flair_file = btk_raw_dir.files("*fla.nii.gz")

            if len(t1_file) == len(t1c_file) == len(t2_file) == len(flair_file) == 1:
                print(t1_file)
                print(t1c_file)
                print(t2_file)
                print(flair_file)

                t1File = t1_file[0]
                t1cFile = t1c_file[0]
                t2File = t2_file[0]
                flaFile = flair_file[0]

                # execute it
                brats_style_t1_centric_preprocessing(
                    input_t1=t1File,
                    output_t1=prep_dir + "/" + inputDir.name + "_t1.nii.gz",
                    input_t1c=t1cFile,
                    output_t1c=prep_dir + "/" + inputDir.name + "_t1c.nii.gz",
                    input_t2=t2File,
                    output_t2=prep_dir + "/" + inputDir.name + "_t2.nii.gz",
                    input_flair=flaFile,
                    output_flair=prep_dir + "/" + inputDir.name + "_fla.nii.gz",
                    bet_mode="gpu",
                    limit_cuda_visible_devices="0",
                    keep_coregistration=brainles_dir + "/co-registration",
                    keep_atlas_registration=brainles_dir + "/atlas-registration",
                    keep_brainextraction=brainles_dir + "/brain-extraction",
                )

    except Exception as e:
        print("error: " + str(e))
        print("conversion error for:", inputDir)

        time = str(datetime.datetime.now().time())

        print("** finished:", inputDir.name, "at:", time)


### *** GOGOGO *** ###
if __name__ == "__main__":
    EXAMPLE_DATA_DIR = turbopath("example_data")

    exams = EXAMPLE_DATA_DIR.dirs()

    for exam in tqdm(exams):
        print(exam)
        preprocess(exam)
