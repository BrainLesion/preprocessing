from .niftyreg import niftyreg_caller
from .niftyreg import NiftyRegRegistrator


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
        # niftyreg_caller(
        #     fixed_image=fixed_image,
        #     moving_image=moving_image,
        #     transformed_image=transformed_image,
        #     matrix=matrix,
        #     log_file=log_file,
        #     mode=mode,
        # )
        if mode == "registration":
            registrator = NiftyRegRegistrator()
            registrator.register(
                fixed_image=fixed_image,
                moving_image=moving_image,
                transformed_image=transformed_image,
                matrix=matrix,
                log_file=log_file,
            )
        elif mode == "transformation":
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
