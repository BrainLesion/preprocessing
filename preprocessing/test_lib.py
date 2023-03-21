from lib import brats_preprocessing
from lib import t1c_focused_preprocessing

if __name__ == "__main__":
    # keep in mind that tests need to be executed from the main directory not from the test subfolder!
    in_dict = {
        "t1": "module/testing/TCGA-DU-7294/TCGA-DU-7294-T1.nii.gz",
        "t1c": "module/testing/TCGA-DU-7294/TCGA-DU-7294-T1c.nii.gz",
        "t2": "module/testing/TCGA-DU-7294/TCGA-DU-7294-T2.nii.gz",
        "fla": "module/testing/TCGA-DU-7294/TCGA-DU-7294-FLAIR.nii.gz",
    }

    out_dict_brats = {
        "t1": "module/testing/test_output_brats/t1.nii.gz",
        "t1c": "module/testing/test_output_brats/t1c.nii.gz",
        "t2": "module/testing/test_output_brats/t2.nii.gz",
        "fla": "module/testing/test_output_brats/fla.nii.gz",
    }

    out_dict_t1c = {
        "t1": "module/testing/test_output_t1c/t1.nii.gz",
        "t1c": "module/testing/test_output_t1c/t1c.nii.gz",
        "t2": "module/testing/test_output_t1c/t2.nii.gz",
        "fla": "module/testing/test_output_t1c/fla.nii.gz",
    }

    opt_dict_brats = {
        "output_dir": "module/testing/test_output_brats",
        "keep_coregistration": True,
        "keep_brats_space": True,
        "bet_mode": "cpu",
        "keep_skullstripping": True,
    }

    opt_dict_t1c = {
        "output_dir": "module/testing/test_output_t1c",
        "keep_coregistration": True,
        "keep_brats_space": True,
        "bet_mode": "cpu",
        "keep_skullstripping": True,
    }

    brats_preprocessing(
        input_dict=in_dict, output_dict=out_dict_brats, options_dict=opt_dict_brats,
    )
    t1c_focused_preprocessing(
        input_dict=in_dict, output_dict=out_dict_t1c, options_dict=opt_dict_t1c,
    )
