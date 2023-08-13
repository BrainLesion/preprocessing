from brainles_hd_bet import run_hd_bet

TODO = "TODO"


def hdbet_caller(
    input_image,
    masked_image,
    # TODO implement logging!
    log_file,
    mode,
):
    # GPU + accurate + TTA
    """skullstrips images with HD-BET generates a skullstripped file and mask"""
    run_hd_bet(
        mri_fnames=[input_image],
        output_fnames=[masked_image],
        # device=0,
        # TODO consider postprocessing
        # postprocess=False,
        mode="accurate",
        device=0,
        postprocess=False,
        do_tta=True,
        keep_mask=True,
        overwrite=True,
    )
