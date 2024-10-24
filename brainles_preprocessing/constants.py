from enum import IntEnum


class PreprocessorSteps(IntEnum):
    INPUT = 0
    COREGISTERED = 1
    ATLAS_REGISTERED = 2
    ATLAS_CORRECTED = 3
    BET = 4
    DEFACED = 5
