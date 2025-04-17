from enum import Enum, IntEnum


class PreprocessorSteps(IntEnum):
    INPUT = 0
    COREGISTERED = 1
    ATLAS_REGISTERED = 2
    ATLAS_CORRECTED = 3
    BET = 4
    DEFACED = 5


class Atlas(str, Enum):
    BRATS_SRI24 = "brats_sri24.nii"
    BRATS_SRI24_SKULLSTRIPPED = "brats_sri24_skullstripped.nii"
    SRI24 = "sri24.nii"
    SRI24_SKULLSTRIPPED = "sri24_skullstripped.nii"
    BRATS_MNI152 = "brats_MNI152lin_T1_1mm.nii.gz"
