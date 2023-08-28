from .niftyreg import niftyreg_caller
from .niftyreg import NiftyRegRegistrator


def register(
    fixed_image,
    moving_image,
    transformed_image,
    matrix,
    log_file,
    backend="niftyreg",
):
    if backend == "niftyreg":
        # niftyreg_caller(
        #     fixed_image=fixed_image,
        #     moving_image=moving_image,
        #     transformed_image=transformed_image,
        #     matrix=matrix,
        #     log_file=log_file,
        #     mode=mode,
        # )
        registrator = NiftyRegRegistrator()
        registrator.register(
            fixed_image=fixed_image,
            moving_image=moving_image,
            transformed_image=transformed_image,
            matrix=matrix,
            log_file=log_file,
        )

    else:
        raise NotImplementedError("this backend is not implemented:", backend)


def transform(
    fixed_image,
    moving_image,
    transformed_image,
    matrix,
    log_file,
    backend="niftyreg",
):
    if backend == "niftyreg":
        # niftyreg_caller(
        #     fixed_image=fixed_image,
        #     moving_image=moving_image,
        #     transformed_image=transformed_image,
        #     matrix=matrix,
        #     log_file=log_file,
        #     mode=mode,
        # )

        registrator = NiftyRegRegistrator()
        registrator.transform(
            fixed_image=fixed_image,
            moving_image=moving_image,
            transformed_image=transformed_image,
            matrix=matrix,
            log_file=log_file,
        )

    else:
        raise NotImplementedError("this backend is not implemented:", backend)
