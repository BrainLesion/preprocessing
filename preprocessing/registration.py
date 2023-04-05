from niftyreg import niftyreg_caller


def registrator(
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
