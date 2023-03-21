from modality_centric import modality_centric_atlas_preprocessing


def brats_style_t1c_centric_preprocessing(
    input_t1c: str,
    output_t1c: str,
    input_t1: str,
    output_t1: str,
    input_t2: str,
    output_t2: str,
    input_flair: str,
    output_flair: str,
    bet_mode: str = "gpu",
    keep_coregistration: str = None,
    keep_atlas_registration: str = None,
    keep_brainextraction: str = None,
):
    io_dict = {
        "t1c": {
            "input": input_t1c,
            "output": output_t1c,
            "bet": True,
        },
        "t1": {
            "input": input_t1,
            "output": output_t1,
            "bet": True,
        },
        "t2": {
            "input": input_t2,
            "output": output_t2,
            "bet": True,
        },
        "flair": {
            "input": input_flair,
            "output": output_flair,
            "bet": True,
        },
    }

    options_dict = {
        "bet_mode": bet_mode,
        "keep_coregistration": keep_coregistration,
        "keep_atlas_registration": keep_atlas_registration,
        "keep_brainextraction": keep_brainextraction,
    }

    modality_centric_atlas_preprocessing(
        io_dict=io_dict,
        options_dict=options_dict,
    )


def brats_style_t1_centric_preprocessing(
    input_t1: str,
    output_t1: str,
    input_t1c: str,
    output_t1c: str,
    input_t2: str,
    output_t2: str,
    input_flair: str,
    output_flair: str,
    bet_mode: str = "gpu",
    keep_coregistration: str = None,
    keep_atlas_registration: str = None,
    keep_brainextraction: str = None,
):
    io_dict = {
        "t1": {
            "input": input_t1,
            "output": output_t1,
            "bet": True,
        },
        "t1c": {
            "input": input_t1c,
            "output": output_t1c,
            "bet": True,
        },
        "t2": {
            "input": input_t2,
            "output": output_t2,
            "bet": True,
        },
        "flair": {
            "input": input_flair,
            "output": output_flair,
            "bet": True,
        },
    }

    options_dict = {
        "bet_mode": bet_mode,
        "keep_coregistration": keep_coregistration,
        "keep_atlas_registration": keep_atlas_registration,
        "keep_brainextraction": keep_brainextraction,
    }

    modality_centric_atlas_preprocessing(
        io_dict=io_dict,
        options_dict=options_dict,
    )
