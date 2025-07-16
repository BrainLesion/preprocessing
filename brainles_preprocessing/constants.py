from enum import Enum, IntEnum


class PreprocessorSteps(IntEnum):
    INPUT = 0
    COREGISTERED = 1
    ATLAS_REGISTERED = 2
    ATLAS_CORRECTED = 3
    N4_BIAS_CORRECTED = 4
    BET = 5
    DEFACED = 6


class Atlas(str, Enum):
    BRATS_SRI24 = "brats_sri24.nii"
    """Slightly modified SRI24 atlas as found in the BraTS challenges"""
    BRATS_SRI24_SKULLSTRIPPED = "brats_sri24_skullstripped.nii"
    """Slightly modified SRI24 skull stripped atlas as found in the BraTS challenges"""

    SRI24 = "sri24.nii"
    """SRI24 atlas from https://www.nitrc.org/frs/download.php/4502/sri24_anatomy_unstripped_nifti.zip"""
    SRI24_SKULLSTRIPPED = "sri24_skullstripped.nii"
    """SRI24 skull stripped atlas from https://www.nitrc.org/frs/download.php/4499/sri24_anatomy_nifti.zip"""

    BRATS_MNI152 = "brats_MNI152lin_T1_1mm.nii.gz"
    """Slightly modified MNI152 atlas as found in the BraTS challenges"""

    MNI152 = "mni_icbm152_t1_tal_nlin_sym_09c.nii"
    """MNI_ICBM_2009c_Nonlinear_Symmetric atlas from https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/"""
