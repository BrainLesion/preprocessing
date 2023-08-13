from .niftyreg import niftyreg_caller


class Registrator:
    def __init__(self, backend="niftyreg"):
        self.backend = backend

    def register(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        raise NotImplementedError("Subclasses must implement the register method")

    def transform(self, fixed_image, moving_image, transformed_image, matrix, log_file):
        raise NotImplementedError("Subclasses must implement the transform method")


def register(
    fixed_image,
    moving_image,
    transformed_image,
    matrix,
    log_file,
    mode,
    backend="niftyreg",
):
    if backend == "niftyreg":
        niftyreg_caller(
            fixed_image=fixed_image,
            moving_image=moving_image,
            transformed_image=transformed_image,
            matrix=matrix,
            log_file=log_file,
            mode=mode,
        )
    else:
        raise NotImplementedError("this backend is not implemented:", backend)
