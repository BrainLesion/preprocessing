# Can be moved to core
from .niftyreg import NiftyRegRegistrator
import logging

logging.basicConfig(level=logging.INFO)


def register(
    fixed_image, moving_image, transformed_image, matrix, log_file, backend="niftyreg"
):
    registrator = _get_registrator(backend)
    registrator.register(fixed_image, moving_image, transformed_image, matrix, log_file)


def transform(
    fixed_image, moving_image, transformed_image, matrix, log_file, backend="niftyreg"
):
    registrator = _get_registrator(backend)
    registrator.transform(
        fixed_image, moving_image, transformed_image, matrix, log_file
    )


def _get_registrator(backend):
    if backend == "niftyreg":
        return NiftyRegRegistrator()
    else:
        raise NotImplementedError(f"Unsupported backend: {backend}")
