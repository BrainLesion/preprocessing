from pathlib import Path
from brainles_preprocessing.back_to_native import BackToNativeSpace
from brainles_preprocessing.registration import ANTsRegistrator, NiftyRegRegistrator


# registrator, abr = ANTsRegistrator(), "ants"
registrator, abr = NiftyRegRegistrator(), "niftyreg"


transforms_dir = Path(
    f"/home/marcelrosier/preprocessing/example/example_data/TCGA-DU-7294/{abr}_TCGA-DU-7294_brainles/transformations"
)

segmentation_path = Path(
    f"/home/marcelrosier/preprocessing/example/example_data/{abr}_TCGA-DU-7294_segmentation.nii.gz"
)
flair_img_path = Path(
    "/home/marcelrosier/preprocessing/example/example_data/TCGA-DU-7294/AXIAL_FLAIR_RF2_150_TCGA-DU-7294_TCGA-DU-7294_GE_TCGA-DU-7294_AXIAL_FLAIR_RF2_150_IR_7_fla.nii.gz"
)
t1c_img_path = Path(
    "/home/marcelrosier/preprocessing/example/example_data/TCGA-DU-7294/AX_T1_POST_GD_FLAIR_TCGA-DU-7294_TCGA-DU-7294_GE_TCGA-DU-7294_AX_T1_POST_GD_FLAIR_RM_13_t1c.nii.gz"
)


back = BackToNativeSpace(
    registrator=registrator,
    transformations_dir=transforms_dir,
)


target = "t1c"  # "flair"
back.transform(
    target_modality_name=target,
    target_modality_img=t1c_img_path if target == "t1c" else flair_img_path,
    moving_image=segmentation_path,
    output_img_path=Path(
        f"/home/marcelrosier/preprocessing/example/{abr}_inverse_test-TCGA-DU-7294_segmentation_{target}.nii.gz"
    ),
    log_file_path="inverse.log",
    # interpolator="linear",
)
