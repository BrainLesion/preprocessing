import warnings
from .atlas_centric_preprocessor import AtlasCentricPreprocessor
from .native_space_preprocessor import NativeSpacePreprocessor


# Deprecation warning for Preprocessor alias, added to ensure backward compatibility.
class Preprocessor(AtlasCentricPreprocessor):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Preprocessor has been renamed to AtlasCentricPreprocessor and is deprecated."
            "The alias will be removed in future releases, please migrate to AtlasCentricPreprocessor.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
