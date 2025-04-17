import warnings


try:
    from .ANTs.ANTs import ANTsRegistrator
except ImportError:
    warnings.warn(
        "ANTS package not found. If you want to use it, please install it using 'pip install antspyx'"
    )


from .niftyreg.niftyreg import NiftyRegRegistrator

from .elastix.elastix import ElastixRegistrator

from .greedy.greedy import GreedyRegistrator
