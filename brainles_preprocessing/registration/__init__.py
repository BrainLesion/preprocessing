import warnings


try:
    from .ANTs.ANTs import ANTsRegistrator
except ImportError:
    warnings.warn(
        "ANTS package not found. If you want to use it, please install it using 'pip install antspyx'"
    )


from .niftyreg.niftyreg import NiftyRegRegistrator


try:
    from .elastix.elastix import ElastixRegistrator
except ImportError:
    warnings.warn(
        "itk-elastix package not found. If you want to use it, please install it using 'pip install brainles_preprocessing[itk-elastix]'"
    )

try:
    from .greedy.greedy import GreedyRegistrator
except ImportError:
    warnings.warn(
        "picsl_greedy  package not found. If you want to use it, please install it using 'pip install brainles_preprocessing[picsl_greedy]'"
    )
